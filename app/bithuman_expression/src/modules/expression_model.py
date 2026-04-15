"""
bithuman_expression/src/modules/expression_model.py
Python replacement for expression_model.cpython-310-x86_64-linux-gnu.so

Audio-conditioned Diffusion Transformer (DiT) for expression avatar generation.

Architecture:
  - Video latent tokens are patchified and fed into a sequence of DiTAudioBlocks
  - Each block performs: AdaLN self-attention → audio cross-attention → MLP
  - Positional encoding via 3-D RoPE (time × height × width)
  - Audio features are projected by AudioProjModel before cross-attention

Classes:
  RMSNorm, MLP, SelfAttention, CrossAttention,
  DiTAudioBlock, AudioProjModel, Head, ExpressionModel

Free functions:
  flash_attention, sinusoidal_embedding_1d,
  precompute_freqs_cis, precompute_freqs_cis_3d, pad_freqs, rope_apply
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim: int, seq_len: int) -> torch.Tensor:
    """
    Create a 1-D sinusoidal positional embedding.

    Args:
        dim:     embedding dimension (must be even)
        seq_len: number of positions

    Returns:
        (seq_len, dim) float tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half
    )
    positions = torch.arange(seq_len, dtype=torch.float32)
    args = positions.unsqueeze(1) * freqs.unsqueeze(0)   # (seq_len, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def precompute_freqs_cis(freq_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponentials for 1-D RoPE.

    Args:
        freq_dim: half of the head dimension
        seq_len:  maximum sequence length
        theta:    RoPE base frequency

    Returns:
        (seq_len, freq_dim) complex64 tensor
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, freq_dim, 2, dtype=torch.float32) / freq_dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)                       # (seq_len, freq_dim//2)
    freqs = torch.polar(torch.ones_like(freqs), freqs)  # complex
    return freqs


def precompute_freqs_cis_3d(
    freq_dim: int,
    t_len: int,
    h_len: int,
    w_len: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Precompute complex exponentials for 3-D RoPE (time × height × width).

    Each spatial-temporal position (t, h, w) gets a combined frequency by
    splitting freq_dim equally among the three axes.

    Args:
        freq_dim: total frequency dimension (split evenly across T/H/W)
        t_len, h_len, w_len: grid extents
        theta:    RoPE base

    Returns:
        (t_len * h_len * w_len, freq_dim) complex64 tensor
    """
    dim_each = freq_dim // 3

    def _1d(length, d):
        freqs = 1.0 / (
            theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
        )
        t = torch.arange(length, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)   # (length, d//2) complex

    ft = _1d(t_len, dim_each)   # (T, D/6) complex
    fh = _1d(h_len, dim_each)
    fw = _1d(w_len, dim_each)

    # Broadcast over a 3-D grid
    ft = ft.unsqueeze(1).unsqueeze(1).expand(-1, h_len, w_len, -1)
    fh = fh.unsqueeze(0).unsqueeze(2).expand(t_len, -1, w_len, -1)
    fw = fw.unsqueeze(0).unsqueeze(0).expand(t_len, h_len, -1, -1)

    freqs = torch.cat([ft, fh, fw], dim=-1)    # (T, H, W, freq_dim//2) complex
    return freqs.reshape(t_len * h_len * w_len, -1)


def pad_freqs(freqs: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Pad a frequency tensor along the last dimension to target_dim by repeating.

    Args:
        freqs:      (..., D) complex tensor
        target_dim: desired last dimension

    Returns:
        (..., target_dim) complex tensor
    """
    current = freqs.shape[-1]
    if current >= target_dim:
        return freqs[..., :target_dim]
    repeat = math.ceil(target_dim / current)
    return freqs.repeat(*([1] * (freqs.ndim - 1)), repeat)[..., :target_dim]


def rope_apply(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to query/key tensors.

    Args:
        x:         (B, L, num_heads, head_dim) float tensor
        freqs_cis: (L, head_dim // 2) complex tensor

    Returns:
        same shape as x, with RoPE applied
    """
    # View as complex: (B, L, num_heads, head_dim//2) complex
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)      # (1, L, 1, head_dim//2)
    x_out = torch.view_as_real(x_c * freqs).flatten(-2)
    return x_out.to(x.dtype)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Scaled dot-product attention (uses torch SDPA which dispatches to
    FlashAttention when available).

    Args:
        q, k, v:   (B, num_heads, L, head_dim)
        attn_mask: optional attention bias
        dropout_p: dropout probability (training only)

    Returns:
        (B, num_heads, L, head_dim)
    """
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.float()).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, in_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(in_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, in_dim: int, num_heads: int):
        super().__init__()
        assert in_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, 3 * in_dim, bias=False)
        self.out = nn.Linear(in_dim, in_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, L, C)
            freqs_cis: (L, head_dim//2) complex — RoPE frequencies

        Returns:
            (B, L, C)
        """
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                          # (B, L, H, D)

        if freqs_cis is not None:
            q = rope_apply(q, freqs_cis)
            k = rope_apply(k, freqs_cis)

        # (B, H, L, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = flash_attention(q, k, v)                   # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention (query from video, key/value from audio)."""

    def __init__(self, in_dim: int, num_heads: int, text_dim: int):
        super().__init__()
        assert in_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = in_dim // num_heads
        self.q_proj  = nn.Linear(in_dim,   in_dim, bias=False)
        self.kv_proj = nn.Linear(text_dim, 2 * in_dim, bias=False)
        self.out     = nn.Linear(in_dim,   in_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, L1, C)  — video tokens
            context: (B, L2, text_dim) — audio context
            mask:    optional (B, 1, L1, L2) attention bias

        Returns:
            (B, L1, C)
        """
        B, L1, C = x.shape
        _, L2, _ = context.shape

        q  = self.q_proj(x).reshape(B, L1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, L2, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = flash_attention(q, k, v, attn_mask=mask)   # (B, H, L1, D)
        out = out.transpose(1, 2).reshape(B, L1, C)
        return self.out(out)


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------

class DiTAudioBlock(nn.Module):
    """
    Single Diffusion Transformer block with audio cross-attention.

    Processing order (AdaLN-Zero style):
      x = x + gate_sa  * SelfAttention(AdaLN(x, t_mod))
      x = x + CrossAttention(x, audio)
      x = x + gate_mlp * MLP(AdaLN(x, t_mod))

    Args:
        in_dim:    model (hidden) dimension
        num_heads: number of attention heads
        ffn_dim:   feed-forward expansion dimension
        text_dim:  audio feature dimension
    """

    def __init__(self, in_dim: int, num_heads: int, ffn_dim: int, text_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(in_dim)
        self.norm2 = RMSNorm(in_dim)
        self.norm3 = RMSNorm(in_dim)

        self.self_attn  = SelfAttention(in_dim, num_heads)
        self.cross_attn = CrossAttention(in_dim, num_heads, text_dim)
        self.mlp        = MLP(in_dim, ffn_dim)

        # AdaLN-Zero modulation: predicts (shift_sa, scale_sa, gate_sa,
        #                                   shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, 6 * in_dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        audio_context: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x(Tensor):     Shape [B, L1, C]  — patchified video latents
            t_mod(Tensor): Shape [B, C] or [B*T, C] — timestep modulation signal
            audio_context: Shape [B, L_audio, text_dim]
            freqs_cis:     Shape [L1, head_dim//2] complex

        Returns:
            (B, L1, C)
        """
        B, L, C = x.shape

        # Broadcast t_mod to match batch if needed
        if t_mod.shape[0] != B:
            t_mod = t_mod[:B]

        mods = self.adaLN_modulation(t_mod)               # (B, 6*C)
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

        # Self-attention branch
        x_sa = self.norm1(x) * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(x_sa, freqs_cis)

        # Audio cross-attention (no modulation)
        x = x + self.cross_attn(self.norm2(x), audio_context)

        # MLP branch
        x_mlp = self.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)

        return x


# ---------------------------------------------------------------------------
# Audio projection
# ---------------------------------------------------------------------------

class AudioProjModel(nn.Module):
    """
    Projects audio encoder features to the model's hidden dimension.

    A simple two-layer MLP with LayerNorm.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_audio, in_dim) — audio features from Wav2Vec2

        Returns:
            (B, T_audio, out_dim)
        """
        return self.proj(x)


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------

class Head(nn.Module):
    """
    Final prediction head: AdaLN + linear projection to output channels.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, 2 * in_dim, bias=True),
        )
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, L, in_dim)
            t_mod: (B, in_dim) timestep modulation

        Returns:
            (B, L, out_dim)
        """
        shift, scale = self.adaLN_modulation(t_mod).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ExpressionModel(nn.Module):
    """
    Audio-conditioned Diffusion Transformer for expression avatar generation.

    Takes noisy video latent patches + audio features + timestep and predicts
    the denoised latent (flow-matching velocity field).

    Args:
        in_dim:     input latent patch channels
        out_dim:    output latent patch channels (usually == in_dim)
        freq_dim:   sinusoidal timestep embedding dimension
        ffn_dim:    feed-forward hidden dimension inside each DiT block
        num_heads:  number of attention heads
        num_layers: number of DiTAudioBlocks
        text_dim:   audio feature dimension (from Wav2Vec2 + AudioProjModel)
    """

    def __init__(
        self,
        in_dim:     int,
        out_dim:    int,
        freq_dim:   int,
        ffn_dim:    int,
        num_heads:  int,
        num_layers: int,
        text_dim:   int,
    ):
        super().__init__()
        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.num_heads = num_heads
        self.head_dim  = in_dim // num_heads

        # Patch embedding
        self.patch_embed = nn.Linear(in_dim, in_dim, bias=True)

        # Timestep embedding: sinusoidal → MLP
        self.time_embed = nn.Sequential(
            nn.Linear(freq_dim, in_dim * 4),
            nn.SiLU(),
            nn.Linear(in_dim * 4, in_dim),
        )
        self.freq_dim = freq_dim

        # Audio projection
        self.audio_proj = AudioProjModel(text_dim, in_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTAudioBlock(in_dim, num_heads, ffn_dim, in_dim)
            for _ in range(num_layers)
        ])

        # Output head
        self.head = Head(in_dim, out_dim)

    # ------------------------------------------------------------------
    # Patchify / unpatchify
    # ------------------------------------------------------------------

    def patchify(
        self,
        x: torch.Tensor,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ) -> torch.Tensor:
        """
        Reshape video latent into a flat sequence of patches.

        Args:
            x:          (B, C, T, H, W)
            patch_size: (pt, ph, pw)

        Returns:
            (B, T//pt * H//ph * W//pw, C * pt * ph * pw)
        """
        pt, ph, pw = patch_size
        return rearrange(
            x,
            "b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)",
            pt=pt, ph=ph, pw=pw,
        )

    def unpatchify(
        self,
        x: torch.Tensor,
        t: int,
        h: int,
        w: int,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ) -> torch.Tensor:
        """
        Reshape flat patch sequence back to video latent shape.

        Args:
            x:          (B, T//pt * H//ph * W//pw, C * pt * ph * pw)
            t, h, w:    grid dimensions (in patches)
            patch_size: (pt, ph, pw)

        Returns:
            (B, C, T, H, W)
        """
        pt, ph, pw = patch_size
        return rearrange(
            x,
            "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=t, h=h, w=w, pt=pt, ph=ph, pw=pw,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        audio_features: torch.Tensor,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, C, T, H, W) noisy video latent
            t:              (B,) float timesteps in [0, 1]
            audio_features: (B, L_audio, text_dim) from Wav2Vec2
            patch_size:     spatial/temporal patch stride
            freqs_cis:      precomputed 3-D RoPE frequencies (optional)

        Returns:
            (B, C, T, H, W) predicted velocity (flow-matching target)
        """
        B, C, T, H, W = x.shape
        pt, ph, pw = patch_size
        gt, gh, gw = T // pt, H // ph, W // pw

        # Timestep embedding
        t_emb = sinusoidal_embedding_1d(self.freq_dim, 1000).to(x.device)
        t_idx = (t * 999).long().clamp(0, 999)
        t_mod = self.time_embed(t_emb[t_idx])                 # (B, in_dim)

        # Patchify + embed
        tokens = self.patchify(x, patch_size)                 # (B, L, C*pt*ph*pw)
        tokens = self.patch_embed(tokens)                     # (B, L, in_dim)

        # Project audio
        audio_ctx = self.audio_proj(audio_features)           # (B, L_audio, in_dim)

        # Precompute RoPE if not provided
        if freqs_cis is None:
            freqs_cis = precompute_freqs_cis_3d(
                self.head_dim, gt, gh, gw
            ).to(x.device)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, t_mod, audio_ctx, freqs_cis)

        # Head + unpatchify
        tokens = self.head(tokens, t_mod)
        return self.unpatchify(tokens, gt, gh, gw, patch_size)
