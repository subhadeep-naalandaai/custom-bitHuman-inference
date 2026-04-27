"""
Naalanda ExpressionModel — audio-conditioned video diffusion transformer.

Architecture (identical to bitHuman's design, reverse-engineered from weight shapes):
  dim=1536, 30 layers, 12 heads, ffn_dim=8960
  Input:  (B, 256, T, H, W) = cat([noisy_latent(128-ch), ref_latent(128-ch)])
  Audio:  (B, L, 768) Wav2Vec2-base features (cross-attention conditioning)
  Output: (B, 128, T, H, W) velocity field (flow-matching)

Standalone — does not import from app/.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_embedding_1d(dim: int, length: int) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    pos   = torch.arange(length, dtype=torch.float32)
    emb   = pos.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


def precompute_freqs_cis_3d(
    head_dim: int, t_len: int, h_len: int, w_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """3-D RoPE frequencies for (T×H×W) positions, shape (T*H*W, head_dim//2) complex."""
    # Split head_dim evenly across T/H/W axes; assign remainder to W
    d_each = (head_dim // 3) & ~1          # round down to even
    d_w    = head_dim - 2 * d_each         # remainder (even by construction)

    def _1d(n, d):
        half   = d // 2
        f      = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
        angles = torch.outer(torch.arange(n, dtype=torch.float32), f)
        return torch.polar(torch.ones_like(angles), angles)  # (n, half) complex

    ft = _1d(t_len, d_each).unsqueeze(1).unsqueeze(1).expand(-1, h_len, w_len, -1)
    fh = _1d(h_len, d_each).unsqueeze(0).unsqueeze(2).expand(t_len, -1, w_len, -1)
    fw = _1d(w_len, d_w).unsqueeze(0).unsqueeze(0).expand(t_len, h_len, -1, -1)
    freqs = torch.cat([ft, fh, fw], dim=-1)   # (T, H, W, head_dim//2)
    return freqs.reshape(t_len * h_len * w_len, -1)


def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
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
        return self.o(out.transpose(1, 2).reshape(B, L, C))


class CrossAttention(nn.Module):
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
        L2 = ctx.shape[1]
        H, D = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x).view(B, L1, H, D)).transpose(1, 2)
        k = self.norm_k(self.k(ctx).view(B, L2, H, D)).transpose(1, 2)
        v = self.v(ctx).view(B, L2, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(B, L1, C))


class DiTBlock(nn.Module):
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
        x:         torch.Tensor,
        t_mod6:    torch.Tensor,
        audio_ctx: torch.Tensor,
        freqs:     Optional[torch.Tensor],
    ) -> torch.Tensor:
        mod = t_mod6 + self.modulation
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = mod.unbind(1)
        h = x * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(h, freqs)
        x = x + self.cross_attn(self.norm3(x), audio_ctx)
        h = x * (1 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
        x = x + gate_ff.unsqueeze(1) * self.ffn(h)
        return x


class AudioEmbedding(nn.Module):
    def __init__(self, audio_dim: int = 768, dim: int = 1536):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, audio_dim, bias=True),
            nn.SiLU(),
            nn.Linear(audio_dim, dim, bias=True),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AudioProjection(nn.Module):
    SHORT_LEN = 30
    LONG_LEN  = 72

    def __init__(self, dim: int = 1536):
        super().__init__()
        self.norm     = nn.LayerNorm(dim)
        self.proj1    = nn.Linear(self.SHORT_LEN * dim, 512, bias=True)
        self.proj1_vf = nn.Linear(self.LONG_LEN  * dim, 512, bias=True)
        self.proj2    = nn.Linear(512, 512, bias=True)
        self.proj3    = nn.Linear(512, 32 * dim, bias=True)
        self.dim      = dim

    def forward(self, audio_emb_out: torch.Tensor) -> torch.Tensor:
        B, L, D = audio_emb_out.shape
        x    = self.norm(audio_emb_out)
        flat = x.reshape(B, -1)
        if L == self.LONG_LEN:
            h = F.silu(self.proj1_vf(flat))
        else:
            if flat.shape[1] != self.SHORT_LEN * D:
                flat = F.interpolate(
                    flat.unsqueeze(1), size=self.SHORT_LEN * D,
                    mode="linear", align_corners=False
                ).squeeze(1)
            h = F.silu(self.proj1(flat))
        h   = F.silu(self.proj2(h))
        out = self.proj3(h)
        return out.view(B, 32, D)


class OutputHead(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.head       = nn.Linear(dim, out_channels, bias=True)
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

    def forward(self, x: torch.Tensor, t_mod6: torch.Tensor) -> torch.Tensor:
        head_mod = t_mod6[:, :2, :] + self.modulation
        shift, scale = head_mod.unbind(1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class ExpressionModel(nn.Module):
    """
    Audio-conditioned video generation DiT.

    Input:
      x:              (B, 256, T, H, W)  — cat([noisy_latent, ref_latent])
      t:              (B,) float in [0,1] — flow-matching timestep
      audio_features: (B, L, 768)        — Wav2Vec2 frame features
    Output:
      (B, 128, T, H, W) — predicted velocity field
    """

    def __init__(
        self,
        dim:                    int  = 1536,
        in_channels:            int  = 256,
        out_channels:           int  = 128,
        freq_dim:               int  = 256,
        ffn_dim:                int  = 8960,
        num_heads:              int  = 12,
        num_layers:             int  = 30,
        audio_dim:              int  = 768,
        text_dim:               int  = 4096,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim                    = dim
        self.in_channels            = in_channels
        self.out_channels           = out_channels
        self.freq_dim               = freq_dim
        self.num_heads              = num_heads
        self.head_dim               = dim // num_heads
        self.gradient_checkpointing = gradient_checkpointing

        self.patch_embedding = nn.Conv3d(in_channels, dim, kernel_size=1, bias=True)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # Text/null conditioning — kept for weight-file compatibility; unused in audio-only mode
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        self.audio_emb  = AudioEmbedding(audio_dim, dim)
        self.audio_proj = AudioProjection(dim)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)
        ])
        self.head = OutputHead(dim, out_channels)

    def forward(
        self,
        x:              torch.Tensor,
        t:              torch.Tensor,
        audio_features: torch.Tensor,
        freqs_cis:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape

        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)  # (B, T*H*W, dim)

        t_emb = sinusoidal_embedding_1d(self.freq_dim, 1000).to(x.device)
        t_idx = (t * 999).long().clamp(0, 999)
        t_h   = self.time_embedding(t_emb[t_idx])
        t_mod = self.time_projection(t_h).view(B, 6, self.dim)

        audio_emb  = self.audio_emb(audio_features)
        global_tok = self.audio_proj(audio_emb)
        audio_ctx  = torch.cat([audio_emb, global_tok], dim=1)

        if freqs_cis is None:
            freqs_cis = precompute_freqs_cis_3d(self.head_dim, T, H, W).to(x.device)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(
                    block, tokens, t_mod, audio_ctx, freqs_cis, use_reentrant=False
                )
            else:
                tokens = block(tokens, t_mod, audio_ctx, freqs_cis)

        tokens = self.head(tokens, t_mod)
        return tokens.transpose(1, 2).view(B, self.out_channels, T, H, W)
