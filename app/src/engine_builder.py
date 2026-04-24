"""
src/engine_builder.py
Python replacement for engine_builder.cpython-310-x86_64-linux-gnu.so

Handles DiT weight loading and TRT engine building with a three-tier fallback:
  Tier 1: Load pre-built TRT engine  (0 s extra startup)
  Tier 2: Auto-build TRT from ONNX   (one-time ~10-30 min, cached by GPU SM)
  Tier 3: Fall back to PyTorch        (no TRT required)

This is the ONLY code path that loads DiT weights and never touches
safetensors directly from other modules, so branding changes are isolated here.

Exports:
  resolve_engine(engine_path, ckpt_dir, cache_dir, device_id) → (path_or_none, use_pytorch_fallback)
  load_dit_model(model_dir)                         → ExpressionModel (eval, on device)
  build_trt_engine(engine_path, ckpt_dir, ...)      → engine_path
  export_onnx(onnx_path, ckpt_dir)                  → onnx_path
  get_gpu_sm_tag()                                  → e.g. 'sm89'
  try_load_engine(engine_path)                      → (engine, error_msg)
  _find_safetensors(model_dir)                      → Path
  _monkey_patch_rope_for_onnx(model)                → model (patched in-place)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger("expression-avatar.engine_builder")


# ---------------------------------------------------------------------------
# Weight discovery
# ---------------------------------------------------------------------------

def _find_safetensors(model_dir: Union[str, Path]) -> Path:
    """
    Find safetensors file in model_dir, checking branded name first then legacy.

    Handles both formats:
      - Branded: bithuman_expression_dit_1_3b.safetensors  with 'bithuman.' key prefix
      - Legacy:  diffusion_pytorch_model.safetensors        with original keys
    """
    model_dir = Path(model_dir)
    branded = model_dir / "bithuman_expression_dit_1_3b.safetensors"
    legacy  = model_dir / "diffusion_pytorch_model.safetensors"
    if branded.exists():
        return branded
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"No safetensors weight file found in {model_dir}. "
        f"Expected {branded.name} or {legacy.name}."
    )


def load_dit_model(model_dir: Union[str, Path]) -> torch.nn.Module:
    """
    Load DiT model from branded or legacy safetensors.

    This is the ONLY code path that loads DiT weights and never touches
    safetensors, so branding changes are isolated here.

    Args:
        model_dir: directory containing config.json + safetensors weights

    Returns:
        ExpressionModel in eval mode on the appropriate device
    """
    import json
    from safetensors.torch import load_file as load_safetensors

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    from bithuman_expression.src.modules.expression_model import ExpressionModel

    # Config keys are at top level (not nested under "expression_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ExpressionModel(
        dim         = cfg.get("dim",        1536),
        in_channels = cfg.get("in_dim",      256),
        out_channels= cfg.get("out_dim",     128),
        freq_dim    = cfg.get("freq_dim",    256),
        ffn_dim     = cfg.get("ffn_dim",    8960),
        num_heads   = cfg.get("num_heads",    12),
        num_layers  = cfg.get("num_layers",   30),
        text_dim    = cfg.get("text_dim",   4096),
    ).to(device)

    safetensors_path = _find_safetensors(model_dir)
    logger.info(f"[engine_builder] Loading DiT weights from {safetensors_path}")

    state = load_safetensors(str(safetensors_path), device=str(device))

    # Strip branded key prefix if present
    if any(k.startswith("bithuman.") for k in state):
        state = {k.removeprefix("bithuman."): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info("[engine_builder] DiT weights loaded")
    return model


# ---------------------------------------------------------------------------
# GPU identification
# ---------------------------------------------------------------------------

def get_gpu_sm_tag() -> str:
    """
    Return SM version tag like 'sm89' for the current GPU.
    Used to cache TRT engines per GPU architecture.
    """
    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(0)
    major, minor = props.major, props.minor
    return f"sm{major}{minor}"


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _monkey_patch_rope_for_onnx(model: torch.nn.Module) -> torch.nn.Module:
    """
    Monkey-patch rope_apply to use real-valued ops (ONNX-compatible).

    The original uses torch.view_as_complex/torch.polar/torch.view_as_real
    which are not ONNX-exportable.  Replace with equivalent real-valued math:

      complex multiply (a+bi)(c+di) = (ac-bd) + (ad+bc)i

    RoPE using only real-valued ops (ONNX-safe).
    """
    from bithuman_expression.src.modules import expression_model as _em

    def rope_apply_real(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x: (..., D), freqs: (..., D//2, 2)  — cos/sin
        *leading, D = x.shape
        x_    = x.reshape(*leading, D // 2, 2)
        x_r   = x_[..., 0]
        x_i   = x_[..., 1]
        cos_  = freqs[..., 0]
        sin_  = freqs[..., 1]
        out_r = x_r * cos_ - x_i * sin_
        out_i = x_r * sin_ + x_i * cos_
        return torch.stack([out_r, out_i], dim=-1).reshape(*leading, D)

    _em.rope_apply = rope_apply_real
    return model


def export_onnx(
    onnx_path: Union[str, Path],
    ckpt_dir: Union[str, Path],
    opset: int = 17,
) -> Path:
    """
    Export ExpressionModel to ONNX with RoPE monkey-patch.

    Args:
        onnx_path: destination .onnx file
        ckpt_dir:  checkpoint directory (contains config.json + weights)
        opset:     ONNX opset version

    Returns:
        Path to the exported ONNX file
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[engine_builder] Exporting ONNX to {onnx_path}")
    model = load_dit_model(ckpt_dir)
    model = _monkey_patch_rope_for_onnx(model)

    device = next(model.parameters()).device

    # Build dummy inputs matching ExpressionModel.forward signature
    B, T, C = 1, 25, 512
    H = W = 64
    x              = torch.randn(B, C, T, H // 8, W // 8, device=device)
    t              = torch.rand(B, device=device)
    audio_features = torch.randn(B, T, 768, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x, t, audio_features),
            str(onnx_path),
            opset_version=opset,
            input_names=["x", "t", "audio_features"],
            output_names=["output"],
            dynamic_axes={
                "x":              {0: "batch", 2: "time"},
                "t":              {0: "batch"},
                "audio_features": {0: "batch", 1: "time"},
                "output":         {0: "batch", 2: "time"},
            },
        )

    logger.info(f"[engine_builder] ONNX exported to {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# TRT engine building and loading
# ---------------------------------------------------------------------------

def try_load_engine(engine_path: Union[str, Path]) -> Tuple[object, str]:
    """
    Try to deserialize a TRT engine.

    Returns:
        (engine, error_msg) — engine is None when loading fails
    """
    try:
        import tensorrt as trt
    except ImportError:
        return None, "TensorRT not installed"

    engine_path = str(engine_path)
    if not Path(engine_path).exists():
        return None, f"Engine file not found: {engine_path}"

    try:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None, "deserialize_cuda_engine returned None"
        return engine, ""
    except Exception as e:
        return None, str(e)


def build_trt_engine(
    engine_path: Union[str, Path],
    ckpt_dir: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Build a TRT FP16 engine from ONNX using trtexec or the Python API.

    Production uses TRT.  The ONNX file is exported first if it does not exist.

    Args:
        engine_path: destination .trt file
        ckpt_dir:    checkpoint directory (model weights)
        cache_dir:   directory for intermediate ONNX (defaults to engine_path parent)

    Returns:
        Path to the built engine file
    """
    engine_path = Path(engine_path)
    if cache_dir is None:
        cache_dir = engine_path.parent
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = cache_dir / "expression_model.onnx"
    if not onnx_path.exists():
        export_onnx(onnx_path, ckpt_dir)

    logger.info(f"[engine_builder] Building TRT engine from {onnx_path}")

    try:
        import tensorrt as trt

        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

        with open(str(onnx_path), "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parse errors: {errors}")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build returned None")

        with open(str(engine_path), "wb") as f:
            f.write(serialized)

        logger.info(f"[engine_builder] TRT engine saved to {engine_path}")

    except ImportError:
        raise RuntimeError(
            "TensorRT is not installed. Install it to build TRT engines, "
            "or set BITHUMAN_INFERENCE_MODE=pytorch to skip TRT."
        )

    return engine_path


# ---------------------------------------------------------------------------
# Three-tier engine resolution
# ---------------------------------------------------------------------------

def resolve_engine(
    engine_path: Optional[Union[str, Path]],
    ckpt_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    device_id: int = 0,
) -> Tuple[Optional[Path], bool]:
    """
    GPU-universal TRT engine resolution with three-tier fallback.

    Three-tier engine resolution. Returns (engine_path_or_none, use_pytorch_fallback).

    Tier 1: Try pre-built engine at engine_path
    Tier 2: Try cached engine, or build from ONNX and cache
    Tier 3: Fall back to PyTorch

    Returns:
        (resolved_path, use_fallback) where:
          resolved_path — Path to engine file, or None if using PyTorch
          use_fallback  — True when falling back to PyTorch inference
    """
    # Tier 1 ----------------------------------------------------------------
    if engine_path is not None:
        engine_path = Path(engine_path)
        if engine_path.exists():
            engine, err = try_load_engine(engine_path)
            if engine is not None:
                logger.info(f"[engine_builder] Tier 1: Pre-built engine loaded: {engine_path}")
                return engine_path, False
            else:
                logger.warning(f"[engine_builder] Tier 1: Engine load failed ({err})")

    # Tier 2 ----------------------------------------------------------------
    if ckpt_dir is not None:
        sm_tag = get_gpu_sm_tag()
        if cache_dir is None:
            cache_dir = Path(ckpt_dir) / "trt_cache"
        cache_dir = Path(cache_dir)
        cached_engine = cache_dir / f"expression_model_{sm_tag}.trt"

        if cached_engine.exists():
            engine, err = try_load_engine(cached_engine)
            if engine is not None:
                logger.info(f"[engine_builder] Tier 2: Cached engine loaded: {cached_engine}")
                return cached_engine, False
            logger.warning(f"[engine_builder] Cached engine unusable ({err}), rebuilding…")

        try:
            built = build_trt_engine(cached_engine, ckpt_dir, cache_dir=cache_dir)
            logger.info(f"[engine_builder] Tier 2: Engine built and cached: {built}")
            return built, False
        except Exception as e:
            logger.warning(f"[engine_builder] Tier 2: Build failed ({e})")

    # Tier 3 ----------------------------------------------------------------
    logger.info("[engine_builder] Tier 3: Falling back to PyTorch")
    return None, True
