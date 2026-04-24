"""
bithuman_expression/src/modules/expression_model.py

BitHuman Expression Model — reverse-engineered from bithuman_expression_dit_1_3b.safetensors.

Actual architecture (confirmed from weight shapes):
  hidden dim (dim):  1536
  audio cross-attn:  1536-dim keys/values
  num_heads:         12  →  head_dim = 1536/12 = 128
  num_layers:        30
  ffn_dim:           8960
  in_channels:       256   (noisy_latent ∥ ref_latent, each 128-ch)
  out_channels:      128
  freq_dim:          256   (sinusoidal timestep embedding)
  audio_dim:         768   (Wav2Vec2-base hidden size)

Layer names (as stored in safetensors after stripping "bithuman." prefix):
  patch_embedding            Conv3d(256, 1536, 1)
  time_embedding.{0,2}       Linear(256→1536→1536) with SiLU
  time_projection.1          Linear(1536, 9216=6×1536)  [with SiLU at idx 0]
  text_embedding.{0,2}       Linear(4096→1536→1536) — null/unused in audio-only mode
  audio_emb.proj.{0,1,3,4}  LayerNorm(768) → Linear(768,768) → SiLU → Linear(768,1536) → LayerNorm(1536)
  audio_proj.{norm,proj1,proj1_vf,proj2,proj3}
                             Aggregates audio sequence into 32 global conditioning tokens
  blocks.{0..29}             DiTBlock(dim=1536, ffn_dim=8960, num_heads=12)
  head.{head,modulation}     Final linear + AdaLN bias
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding_1d(dim: int, length: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    pos   = torch.arange(length, dtype=torch.float32)
    emb   = pos.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)   # (length, dim)


def precompute_freqs_cis_3d(
    head_dim: int, t_len: int, h_len: int, w_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """3-D RoPE frequencies for (T×H×W) positions, shape (T*H*W, head_dim//2) complex."""
    d = head_dim // 6

    def _1d(n):
        f = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        t = torch.arange(n, dtype=torch.float32)
        return torch.polar(torch.ones_like(torch.outer(t, f)), torch.outer(t, f))

    ft = _1d(t_len).unsqueeze(1).unsqueeze(1).expand(-1, h_len, w_len, -1)
    fh = _1d(h_len).unsqueeze(0).unsqueeze(2).expand(t_len, -1, w_len, -1)
    fw = _1d(w_len).unsqueeze(0).unsqueeze(0).expand(t_len, h_len, -1, -1)
    freqs = torch.cat([ft, fh, fw], dim=-1)              # (T,H,W, head_dim//2)
    return freqs.reshape(t_len * h_len * w_len, -1)


def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to (B, L, H, D); freqs is (L, D//2) complex."""
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out  = torch.view_as_real(x_c * freqs.unsqueeze(0).unsqueeze(2)).flatten(-2)
    return out.to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).to(x.dtype) * self.weight


class SelfAttention(nn.Module):
    """Multi-head self-attention with RMSNorm on Q/K and RoPE."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.q      = nn.Linear(dim, dim, bias=True)
        self.k      = nn.Linear(dim, dim, bias=True)
        self.v      = nn.Linear(dim, dim, bias=True)
        self.o      = nn.Linear(dim, dim, bias=True)
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, freqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        H, D    = self.num_heads, self.head_dim

        q = self.q(x).view(B, L, H, D)
        k = self.k(x).view(B, L, H, D)
        v = self.v(x).view(B, L, H, D)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if freqs is not None:
            q = rope_apply(q, freqs)
            k = rope_apply(k, freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.o(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention (video queries, audio keys/values)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.q      = nn.Linear(dim, dim, bias=True)
        self.k      = nn.Linear(dim, dim, bias=True)
        self.v      = nn.Linear(dim, dim, bias=True)
        self.o      = nn.Linear(dim, dim, bias=True)
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, L1, C = x.shape
        L2       = ctx.shape[1]
        H, D     = self.num_heads, self.head_dim

        q = self.norm_q(self.q(x).view(B, L1, H, D)).transpose(1, 2)
        k = self.norm_k(self.k(ctx).view(B, L2, H, D)).transpose(1, 2)
        v = self.v(ctx).view(B, L2, H, D).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L1, C)
        return self.o(out)


class DiTBlock(nn.Module):
    """
    Single DiT block (matched to actual weight layout):
      self_attn  with AdaLN-Zero modulation (gate/shift/scale × 2 for SA + FFN)
      cross_attn on audio context
      FFN: Linear → GELU → Linear (NOT SwiGLU — confirmed by 2-layer ffn.{0,2})
      norm3: LayerNorm after cross-attn
      modulation: [1, 6, dim] learnable bias added to global timestep mod
    """

    def __init__(self, dim: int, ffn_dim: int, num_heads: int):
        super().__init__()
        self.self_attn  = SelfAttention(dim, num_heads)
        self.cross_attn = CrossAttention(dim, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(ffn_dim, dim, bias=True),
        )

        self.norm3      = nn.LayerNorm(dim, elementwise_affine=True)
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(
        self,
        x:        torch.Tensor,            # (B, L, dim)
        t_mod6:   torch.Tensor,            # (B, 6, dim) from time_projection
        audio_ctx: torch.Tensor,           # (B, L_audio, dim)
        freqs:    Optional[torch.Tensor],  # RoPE frequencies
    ) -> torch.Tensor:
        # Block-specific modulation = global timestep mod + learned offset
        mod = (t_mod6 + self.modulation)   # (B, 6, dim)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = mod.unbind(1)

        # Self-attention with AdaLN
        h    = x * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        x    = x + gate_sa.unsqueeze(1) * self.self_attn(h, freqs)

        # Cross-attention (no modulation, plain residual)
        x = x + self.cross_attn(self.norm3(x), audio_ctx)

        # FFN with AdaLN
        h = x * (1 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
        x = x + gate_ff.unsqueeze(1) * self.ffn(h)

        return x


class AudioEmbedding(nn.Module):
    """
    Wav2Vec2 features (768-dim) → model hidden dim (1536).
    Weight layout: proj.{0,1,3,4} → LayerNorm→Linear→SiLU→Linear→LayerNorm
    """

    def __init__(self, audio_dim: int = 768, dim: int = 1536):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(audio_dim),       # idx 0
            nn.Linear(audio_dim, audio_dim, bias=True),  # idx 1
            nn.SiLU(),                     # idx 2  (no params)
            nn.Linear(audio_dim, dim, bias=True),        # idx 3
            nn.LayerNorm(dim),             # idx 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AudioProjection(nn.Module):
    """
    Aggregates per-frame audio features into a fixed set of global
    conditioning tokens appended to the cross-attention context.

    Two input size variants:
      proj1    — expects audio_len == SHORT_LEN  (30 frames × dim = 46080)
      proj1_vf — expects audio_len == LONG_LEN   (72 frames × dim = 110592)

    Output: (B, 32, dim)  — 32 global conditioning tokens
    """

    SHORT_LEN = 30   # 46080 / 1536
    LONG_LEN  = 72   # 110592 / 1536

    def __init__(self, dim: int = 1536):
        super().__init__()
        self.norm     = nn.LayerNorm(dim)
        self.proj1    = nn.Linear(self.SHORT_LEN * dim, 512, bias=True)
        self.proj1_vf = nn.Linear(self.LONG_LEN  * dim, 512, bias=True)
        self.proj2    = nn.Linear(512, 512, bias=True)
        self.proj3    = nn.Linear(512, 32 * dim, bias=True)
        self.dim      = dim

    def forward(self, audio_emb_out: torch.Tensor) -> torch.Tensor:
        """
        audio_emb_out: (B, L, dim) — output of AudioEmbedding
        Returns: (B, 32, dim) global conditioning tokens
        """
        B, L, D = audio_emb_out.shape
        x = self.norm(audio_emb_out)                # (B, L, dim)

        flat = x.reshape(B, -1)                     # (B, L*dim)

        # Choose pathway by audio length
        if L == self.LONG_LEN:
            h = F.silu(self.proj1_vf(flat))
        else:
            # Resize flat to SHORT_LEN*dim regardless of actual length
            if flat.shape[1] != self.SHORT_LEN * D:
                flat = F.interpolate(
                    flat.unsqueeze(1),
                    size=self.SHORT_LEN * D,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            h = F.silu(self.proj1(flat))

        h = F.silu(self.proj2(h))                   # (B, 512)
        out = self.proj3(h)                          # (B, 32*dim)
        return out.view(B, 32, D)                    # (B, 32, dim)


class OutputHead(nn.Module):
    """
    Final AdaLN + linear prediction head.
    Weight layout: head.head + head.modulation [1, 2, dim]
    """

    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.head       = nn.Linear(dim, out_channels, bias=True)
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

    def forward(self, x: torch.Tensor, t_mod6: torch.Tensor) -> torch.Tensor:
        # Use first 2 of the 6 modulation channels (shift, scale)
        head_mod = t_mod6[:, :2, :] + self.modulation   # (B, 2, dim)
        shift, scale = head_mod.unbind(1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class BitHumanExpressionModel(nn.Module):
    """
    Audio-conditioned video generation DiT.
    Matches the weights in bithuman_expression_dit_1_3b.safetensors.

    Input:
      x:             (B, in_channels=256, T, H, W)
                     Concatenation of [noisy_latent (128-ch) ∥ ref_latent (128-ch)]
      t:             (B,) float in [0, 1]  — flow-matching timestep
      audio_features:(B, L_audio, audio_dim=768)  — Wav2Vec2 frame features
      freqs_cis:     optional precomputed 3-D RoPE (omit to compute on-the-fly)

    Output:
      (B, out_channels=128, T, H, W) — predicted velocity field
    """

    def __init__(
        self,
        dim:          int = 1536,
        in_channels:  int = 256,
        out_channels: int = 128,
        freq_dim:     int = 256,
        ffn_dim:      int = 8960,
        num_heads:    int = 12,
        num_layers:   int = 30,
        audio_dim:    int = 768,
        text_dim:     int = 4096,
    ):
        super().__init__()
        self.dim          = dim
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.freq_dim     = freq_dim
        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads

        # Input: patchify video latent (no spatial patching; kernel=(1,1,1))
        self.patch_embedding = nn.Conv3d(in_channels, dim, kernel_size=1, bias=True)

        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        # Global AdaLN modulation (SiLU + Linear, weights at index 1)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # Text/null conditioning (4096-dim → dim; not used in audio-only inference)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        # Audio embedding: Wav2Vec2 768 → 1536
        self.audio_emb  = AudioEmbedding(audio_dim, dim)

        # Global audio projection: aggregates audio sequence → 32 conditioning tokens
        self.audio_proj = AudioProjection(dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)
        ])

        # Output head
        self.head = OutputHead(dim, out_channels)

    def forward(
        self,
        x:              torch.Tensor,
        t:              torch.Tensor,
        audio_features: torch.Tensor,
        freqs_cis:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # ── Patch embed ───────────────────────────────────────────────────────
        tokens = self.patch_embedding(x)              # (B, dim, T, H, W)
        tokens = tokens.flatten(2).transpose(1, 2)    # (B, T*H*W, dim)

        # ── Timestep modulation ───────────────────────────────────────────────
        t_emb  = sinusoidal_embedding_1d(self.freq_dim, 1000).to(x.device)
        t_idx  = (t * 999).long().clamp(0, 999)
        t_h    = self.time_embedding(t_emb[t_idx])          # (B, dim)
        t_mod  = self.time_projection(t_h).view(B, 6, self.dim)  # (B, 6, dim)

        # ── Audio conditioning ────────────────────────────────────────────────
        audio_emb  = self.audio_emb(audio_features)          # (B, L, dim)
        global_tok = self.audio_proj(audio_emb)               # (B, 32, dim)
        audio_ctx  = torch.cat([audio_emb, global_tok], dim=1)  # (B, L+32, dim)

        # ── RoPE ─────────────────────────────────────────────────────────────
        if freqs_cis is None:
            freqs_cis = precompute_freqs_cis_3d(
                self.head_dim, T, H, W
            ).to(x.device)

        # ── Transformer ───────────────────────────────────────────────────────
        for block in self.blocks:
            tokens = block(tokens, t_mod, audio_ctx, freqs_cis)

        # ── Head + unpatchify ─────────────────────────────────────────────────
        tokens = self.head(tokens, t_mod)                     # (B, T*H*W, out_channels)
        tokens = tokens.transpose(1, 2).view(B, self.out_channels, T, H, W)
        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Legacy alias — kept so old code that imports ExpressionModel still works
# ─────────────────────────────────────────────────────────────────────────────

ExpressionModel = BitHumanExpressionModel
