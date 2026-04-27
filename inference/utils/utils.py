"""
bithuman_expression/utils/utils.py
Python replacement for utils.cpython-310-x86_64-linux-gnu.so

Implements 4 PyTorch tensor utilities:
  - rgb_to_lab_torch
  - lab_to_rgb_torch
  - resize_and_centercrop
  - match_and_blend_colors_torch
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# sRGB ↔ CIELab conversion helpers (D65 white point)
# ---------------------------------------------------------------------------

# sRGB → linear RGB (undo gamma)
def _srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)


# linear RGB → sRGB (apply gamma)
def _linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    return torch.where(
        t <= 0.0031308,
        12.92 * t,
        1.055 * t.clamp(min=0.0) ** (1.0 / 2.4) - 0.055,
    )


# sRGB → XYZ (D65, IEC 61966-2-1)
_RGB2XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=torch.float32,
)

# XYZ → sRGB (inverse)
_XYZ2RGB = torch.tensor(
    [
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ],
    dtype=torch.float32,
)

# D65 white point
_XN, _YN, _ZN = 0.95047, 1.00000, 1.08883
_DELTA = 6.0 / 29.0


def _f(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t > _DELTA ** 3, t.clamp(min=1e-10) ** (1.0 / 3.0), t / (3 * _DELTA ** 2) + 4.0 / 29.0)


def _f_inv(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))


def _to_channels_last(t: torch.Tensor):
    """(B,C,…spatial) → (B,…spatial,C)"""
    dims = list(range(t.ndim))
    return t.permute(dims[0], *dims[2:], dims[1])


def _to_channels_first(t: torch.Tensor):
    """(B,…spatial,C) → (B,C,…spatial)"""
    dims = list(range(t.ndim))
    return t.permute(dims[0], dims[-1], *dims[1:-1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rgb_to_lab_torch(img: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB tensor to CIELab.

    Args:
        img: (B,3,H,W) or (B,3,T,H,W), values in [-1, 1]

    Returns:
        lab: same shape, L in [0,100], a/b in [-128, 127]
    """
    device, dtype = img.device, img.dtype

    # Denormalize [-1,1] → [0,1]
    rgb = (img.float().clamp(-1, 1) + 1.0) / 2.0

    # Move channel dim last for matmul
    rgb_cl = _to_channels_last(rgb)          # (..., 3)
    lin = _srgb_to_linear(rgb_cl)

    M = _RGB2XYZ.to(device)
    xyz = lin @ M.T                           # (..., 3)

    # Normalise by white point
    xyz[..., 0] /= _XN
    xyz[..., 1] /= _YN
    xyz[..., 2] /= _ZN

    fx = _f(xyz[..., 0])
    fy = _f(xyz[..., 1])
    fz = _f(xyz[..., 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab = torch.stack([L, a, b], dim=-1)
    return _to_channels_first(lab).to(dtype)


def lab_to_rgb_torch(lab: torch.Tensor) -> torch.Tensor:
    """
    Convert CIELab tensor to sRGB.

    Args:
        lab: (B,3,H,W) or (B,3,T,H,W), L in [0,100], a/b in [-128,127]

    Returns:
        rgb: same shape, values in [-1, 1]
    """
    device, dtype = lab.device, lab.dtype
    lab_f = lab.float()
    lab_cl = _to_channels_last(lab_f)        # (..., 3)

    L = lab_cl[..., 0]
    a = lab_cl[..., 1]
    b = lab_cl[..., 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    X = _XN * _f_inv(fx)
    Y = _YN * _f_inv(fy)
    Z = _ZN * _f_inv(fz)

    xyz = torch.stack([X, Y, Z], dim=-1)
    M = _XYZ2RGB.to(device)
    lin = xyz @ M.T                           # (..., 3)

    srgb = _linear_to_srgb(lin.clamp(0.0, 1.0))
    rgb = _to_channels_first(srgb).to(dtype)

    # Normalize [0,1] → [-1,1]
    return rgb * 2.0 - 1.0


def resize_and_centercrop(
    source_chunk: torch.Tensor,
    reference_image: torch.Tensor,
) -> torch.Tensor:
    """
    Resize source_chunk spatial dims to match reference_image, then center-crop.

    Args:
        source_chunk:    (B, C, T, H, W), values in [-1, 1]
        reference_image: (B, C, 1, H, W), values in [-1, 1]

    Returns:
        Tensor of shape (B, C, T, ref_H, ref_W)
    """
    B, C, T, sH, sW = source_chunk.shape
    _, _, _, rH, rW = reference_image.shape

    if sH == rH and sW == rW:
        return source_chunk

    # Merge batch+time → process as a stack of 2-D frames
    flat = source_chunk.reshape(B * T, C, sH, sW)

    # Scale so both dims are >= reference (preserve aspect, upscale if needed)
    scale = max(rH / sH, rW / sW)
    iH = max(rH, int(round(sH * scale)))
    iW = max(rW, int(round(sW * scale)))

    resized = F.interpolate(flat.float(), size=(iH, iW), mode="bilinear", align_corners=False)

    # Center-crop to (rH, rW)
    y0 = (iH - rH) // 2
    x0 = (iW - rW) // 2
    cropped = resized[:, :, y0 : y0 + rH, x0 : x0 + rW]

    return cropped.to(source_chunk.dtype).reshape(B, C, T, rH, rW)


def match_and_blend_colors_torch(
    source: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Transfer the color statistics of `reference` onto `source` in LAB space
    (Reinhard et al. 2001 color transfer).

    Args:
        source:    (B, 3, …), values in [-1, 1]
        reference: (B, 3, …), values in [-1, 1]  (same spatial shape or broadcastable)

    Returns:
        Tensor with source content but reference color distribution, values in [-1, 1]
    """
    src_lab = rgb_to_lab_torch(source)
    ref_lab = rgb_to_lab_torch(reference)

    # Compute per-channel stats over all spatial/temporal dims
    spatial_dims = list(range(2, src_lab.ndim))

    src_mean = src_lab.mean(dim=spatial_dims, keepdim=True)
    src_std  = src_lab.std(dim=spatial_dims, keepdim=True).clamp(min=1e-6)
    ref_mean = ref_lab.mean(dim=spatial_dims, keepdim=True)
    ref_std  = ref_lab.std(dim=spatial_dims, keepdim=True).clamp(min=1e-6)

    matched = (src_lab - src_mean) / src_std * ref_std + ref_mean

    return lab_to_rgb_torch(matched)
