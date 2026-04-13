"""
BitHumanExpressionModel — reverse-engineered from safetensors weight inspection.

Confirmed tensor structure (843 tensors, all under "bithuman.*"):

  patch_embedding:     Conv3d(256, 1536, kernel=(1,1,1))     ← concat(noisy, ref) = 2×128ch
  time_embedding:      Sequential(Lin(256,1536), SiLU, Lin(1536,1536))
  time_projection:     Sequential(SiLU, Lin(1536, 9216))     ← 9216 = 6×1536 mod factors
  text_embedding:      Sequential(Lin(4096,1536), SiLU, Lin(1536,1536))  ← per-token audio ctx

  audio_emb.proj:      Sequential(LN(768), Lin(768,768), GELU, Lin(768,1536), LN(1536))
  audio_proj:
    norm:              LayerNorm(1536)
    proj1:             Linear(30*1536=46080, 512)    ← 30 audio frames flattened
    proj1_vf:          Linear(72*1536=110592, 512)   ← motion-frame video feats (optional)
    proj2:             Linear(512, 512)
    proj3:             Linear(512, 49152)             ← 49152 = 12*4096

  Per block (×30):
    modulation:        Parameter[1, 6, 1536]         ← per-block AdaLN bias
    self_attn:         q,k,v,o: Lin(1536,1536) + norm_q,norm_k: RMSNorm(1536)
    cross_attn:        q,k,v,o: Lin(1536,1536) + norm_q,norm_k: RMSNorm(1536)
    norm3:             LayerNorm(1536)
    ffn:               Sequential(Lin(1536,8960), GELU, Lin(8960,1536))

  head:
    modulation:        Parameter[1, 2, 1536]
    head:              Linear(1536, 128)
"""
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding for scalar timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)   # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]


class RMSNorm(nn.Module):
    """Per-head RMS norm (no bias, scale only)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


def adalnorm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: LayerNorm then modulate with (1 + scale) * x + shift."""
    x = F.layer_norm(x, [x.shape[-1]])
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head attention with QK-norm. Used for both self- and cross-attn."""
    def __init__(self, dim: int, num_heads: int, context_dim: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        ctx = context_dim if context_dim is not None else dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(ctx, dim)
        self.v = nn.Linear(ctx, dim)
        self.o = nn.Linear(dim, dim)
        # QK-norm applied at full dim before head split
        # (checkpoint: norm_q/norm_k.weight shape is [1536], not [head_dim])
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:        [B, N, dim]
        context:  [B, M, ctx_dim] for cross-attn; None for self-attn
        Returns:  [B, N, dim]
        """
        B, N, _ = x.shape
        if context is None:
            context = x

        # Project → normalize at full dim → then split into heads
        q = self.norm_q(self.q(x))          # [B, N, dim]
        k = self.norm_k(self.k(context))    # [B, M, dim]
        v = self.v(context)                 # [B, M, dim]

        # Reshape to heads
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            b, s, d = t.shape
            return t.reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            # → [B, H, S, head_dim]

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Per-head QK-norm
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v)  # [B, H, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, -1)    # [B, N, dim]
        return self.o(out)


# ---------------------------------------------------------------------------
# DiT Block
# ---------------------------------------------------------------------------

class DiTAudioBlock(nn.Module):
    """
    One transformer block with:
      1. Self-attention  (AdaLN modulated: shift1/scale1/gate1)
      2. Cross-attention on audio context (fixed LayerNorm = norm3, no AdaLN gate)
      3. FFN            (AdaLN modulated: shift2/scale2/gate2)

    modulation [1, 6, 1536] is a learned per-block bias added to the shared
    time_projection output before extracting the 6 AdaLN factors.
    """
    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))
        self.self_attn  = Attention(dim, num_heads)
        self.norm3      = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(
        self,
        x:          torch.Tensor,   # [B, N, dim]
        audio_ctx:  torch.Tensor,   # [B, 12, dim]
        time_proj:  torch.Tensor,   # [B, 6, dim]  shared from time_projection
    ) -> torch.Tensor:
        # Per-block AdaLN modulation = shared time proj + learned bias
        mod = time_proj + self.modulation                  # [B, 6, dim]
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.unbind(dim=1)

        # 1. Self-attention with AdaLN
        x = x + gate1.unsqueeze(1) * self.self_attn(adalnorm(x, shift1, scale1))

        # 2. Cross-attention on audio context (no AdaLN, no gate)
        x = x + self.cross_attn(self.norm3(x), audio_ctx)

        # 3. FFN with AdaLN
        x = x + gate2.unsqueeze(1) * self.ffn(adalnorm(x, shift2, scale2))

        return x


# ---------------------------------------------------------------------------
# Audio embedding + projection
# ---------------------------------------------------------------------------

class AudioEmbedding(nn.Module):
    """
    Projects Wav2Vec2 features [B, 30, 768] → [B, 30, 1536].
    Sequential(LN(768), Linear(768,768), GELU, Linear(768,1536), LN(1536))
    Tensor indices 0,1,_,3,4  (index 2 = GELU, no params).
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(768),       # idx 0
            nn.Linear(768, 768),     # idx 1
            nn.GELU(),               # idx 2 (no params — matches missing index)
            nn.Linear(768, 1536),    # idx 3
            nn.LayerNorm(1536),      # idx 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AudioProjection(nn.Module):
    """
    Compresses 30 audio frames → 12 context tokens of dim 4096.

    proj1    : Linear(30*1536=46080,  512)   — audio path
    proj1_vf : Linear(72*1536=110592, 512)   — motion-frame path (optional)
    proj2    : Linear(512, 512)
    proj3    : Linear(512, 49152)  → reshape [B, 12, 4096]
    """
    AUDIO_FLAT   = 30 * 1536    # 46080
    VIDEO_FLAT   = 72 * 1536    # 110592
    CTX_TOKENS   = 12
    CTX_DIM      = 4096

    def __init__(self):
        super().__init__()
        self.norm     = nn.LayerNorm(1536)
        self.proj1    = nn.Linear(self.AUDIO_FLAT,  512)
        self.proj1_vf = nn.Linear(self.VIDEO_FLAT, 512)
        self.proj2    = nn.Linear(512, 512)
        self.proj3    = nn.Linear(512, self.CTX_TOKENS * self.CTX_DIM)

    def forward(
        self,
        audio_emb: torch.Tensor,              # [B, 30, 1536]
        video_ref: torch.Tensor | None = None,  # [B, 72, 1536] optional motion frames
    ) -> torch.Tensor:
        """Returns [B, 12, 4096]."""
        B = audio_emb.shape[0]

        x = self.norm(audio_emb)               # [B, 30, 1536]
        x = x.reshape(B, -1)                   # [B, 46080]
        h = self.proj1(x)                      # [B, 512]

        if video_ref is not None:
            vf = video_ref.reshape(B, -1)      # [B, 110592]
            h = h + self.proj1_vf(vf)

        h = F.silu(h)
        h = self.proj2(h)                      # [B, 512]
        h = F.silu(h)
        h = self.proj3(h)                      # [B, 49152]

        return h.reshape(B, self.CTX_TOKENS, self.CTX_DIM)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class BitHumanExpressionModel(nn.Module):
    """
    1.3B parameter DiT for audio-conditioned talking-head video generation.

    Input:
      noisy_latent : [B, 128, T_lat, H_lat, W_lat]  — noisy video latents
      ref_latent   : [B, 128, T_lat, H_lat, W_lat]  — reference image latents
      audio_feats  : [B, 30, 768]                    — Wav2Vec2 features
      timesteps    : [B]                              — diffusion timesteps in [0, 1000]

    Output:
      [B, 128, T_lat, H_lat, W_lat]  — predicted velocity (flow matching)
    """

    def __init__(self, config: dict):
        super().__init__()
        dim      = config["dim"]        # 1536
        heads    = config["num_heads"]  # 12
        layers   = config["num_layers"] # 30
        ffn_dim  = config["ffn_dim"]    # 8960
        freq_dim = config["freq_dim"]   # 256
        text_dim = AudioProjection.CTX_DIM  # 4096
        out_dim  = config["out_dim"]    # 128

        # Patch embedding: Conv3d(256, 1536, (1,1,1))
        self.patch_embedding = nn.Conv3d(
            config["in_dim"], dim, kernel_size=(1, 1, 1)
        )

        # Timestep embedding: sinusoidal(freq_dim) → MLP → [B, dim]
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Shared time-to-modulation projection: [B, dim] → [B, 6*dim]
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

        # Audio pipeline
        self.audio_emb  = AudioEmbedding()
        self.audio_proj = AudioProjection()

        # text_embedding maps each of 12 audio context tokens [4096] → [1536]
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # 30 DiT blocks
        self.blocks = nn.ModuleList([
            DiTAudioBlock(dim, heads, ffn_dim) for _ in range(layers)
        ])

        # Output head
        self.head = _OutputHead(dim, out_dim)

    # ------------------------------------------------------------------
    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """[B, C, T, H, W] → [B, T*H*W, dim] after Conv3d."""
        x = self.patch_embedding(x)           # [B, dim, T, H, W]
        B, D, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)         # [B, T, H, W, dim]
        x = x.reshape(B, T * H * W, D)
        return x, (T, H, W)

    def _unpatchify(self, x: torch.Tensor, shape: tuple) -> torch.Tensor:
        """[B, T*H*W, out_dim] → [B, out_dim, T, H, W]."""
        T, H, W = shape
        B = x.shape[0]
        out_dim = x.shape[-1]
        x = x.reshape(B, T, H, W, out_dim)
        x = x.permute(0, 4, 1, 2, 3)         # [B, out_dim, T, H, W]
        return x

    # ------------------------------------------------------------------
    def forward(
        self,
        noisy_latent: torch.Tensor,   # [B, 128, T, H, W]
        ref_latent:   torch.Tensor,   # [B, 128, T, H, W]
        audio_feats:  torch.Tensor,   # [B, 30, 768]
        timesteps:    torch.Tensor,   # [B]
    ) -> torch.Tensor:
        B = noisy_latent.shape[0]

        # --- 1. Patch embedding ---
        x_in = torch.cat([noisy_latent, ref_latent], dim=1)  # [B, 256, T, H, W]
        x, spatial_shape = self._patchify(x_in)              # [B, N, 1536]

        # --- 2. Audio context ---
        audio_emb = self.audio_emb(audio_feats)               # [B, 30, 1536]
        audio_ctx_4096 = self.audio_proj(audio_emb)            # [B, 12, 4096]
        audio_ctx = self.text_embedding(audio_ctx_4096)        # [B, 12, 1536]

        # --- 3. Time conditioning ---
        t_emb = sinusoidal_embedding(timesteps, 256)           # [B, 256]
        t_emb = self.time_embedding(t_emb.to(x.dtype))        # [B, 1536]
        time_proj = self.time_projection(t_emb)                # [B, 9216]
        time_proj = time_proj.reshape(B, 6, -1)               # [B, 6, 1536]

        # --- 4. Transformer blocks ---
        for block in self.blocks:
            x = block(x, audio_ctx, time_proj)

        # --- 5. Output head ---
        x = self.head(x)                                       # [B, N, 128]
        return self._unpatchify(x, spatial_shape)              # [B, 128, T, H, W]

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        config_path: str,
        weights_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "BitHumanExpressionModel":
        config = json.loads(Path(config_path).read_text())
        model  = cls(config)

        state = load_file(weights_path, device="cpu")
        # Strip "bithuman." prefix from all keys
        state = {k.removeprefix("bithuman."): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[model] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[model] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        return model.to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------

class _OutputHead(nn.Module):
    """
    Final LayerNorm + linear projection.
    modulation [1, 2, 1536]: learned (shift, scale) bias for the final AdaLN.
    The shift/scale are independent of timestep — this is a fixed global bias.
    """
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))
        self.head        = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation.unbind(dim=1)  # each [1, dim]
        x = F.layer_norm(x, [x.shape[-1]])
        x = x * (1.0 + scale) + shift
        return self.head(x)
