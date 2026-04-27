#!/usr/bin/env python3
"""
Initialise ExpressionModel weights from the open-source LTX-Video transformer.

LTX-Video (Lightricks/LTX-Video, Apache 2.0) has inner_dim=2048, 28 blocks.
Our ExpressionModel has dim=1536, 30 blocks.

Mapping strategy
────────────────
For each ExpressionModel block i (0–29), map from LTX block j = min(i, 27):
  self-attention  Q/K/V/O : truncate (2048,2048) → (1536,1536)
  FFN up          (in  ):   truncate+pad  (8192,2048) → (8960,1536)
  FFN down        (out ):   truncate+pad  (2048,8192) → (1536,8960)
  block modulation        : zero-init (matches ExpressionModel default)

Cross-attention Q/K/V/O, audio embedding, patch embedding, time embed, head:
  random init — these layers process audio (not text) and need fresh training.

Usage
─────
  python train/init_weights.py \
    --ltx-dir ltx-weights/transformer \
    --output  naalanda-weights/ \
    [--verbose]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: truncate / pad a weight matrix to a target shape
# ─────────────────────────────────────────────────────────────────────────────

def _fit(W: torch.Tensor, out_rows: int, out_cols: int) -> torch.Tensor:
    """
    Return a (out_rows, out_cols) version of W by:
      - truncating rows/cols when W is larger
      - zero-padding when W is smaller

    This preserves the top-left block of W (most structurally-similar region).
    """
    r, c = W.shape
    rows = min(r, out_rows)
    cols = min(c, out_cols)
    W_new = torch.zeros(out_rows, out_cols, dtype=W.dtype)
    W_new[:rows, :cols] = W[:rows, :cols]
    return W_new


def _fit_bias(b: torch.Tensor, out_dim: int) -> torch.Tensor:
    n = b.shape[0]
    if n >= out_dim:
        return b[:out_dim].clone()
    result = torch.zeros(out_dim, dtype=b.dtype)
    result[:n] = b
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Discover LTX-Video state-dict key patterns
# ─────────────────────────────────────────────────────────────────────────────

def _load_ltx_state_dict(ltx_dir: str) -> dict:
    """Load LTX-Video transformer weights; auto-detect format."""
    from safetensors.torch import load_file

    p = Path(ltx_dir)
    sf_files = sorted(p.glob("*.safetensors"))
    if not sf_files:
        # Some checkpoints use .bin
        import glob
        bin_files = sorted(Path(ltx_dir).glob("*.bin"))
        if not bin_files:
            raise FileNotFoundError(f"No weight files in {ltx_dir}")
        state = {}
        for bf in bin_files:
            state.update(torch.load(bf, map_location="cpu"))
        return state

    state = {}
    for sf in sf_files:
        state.update(load_file(str(sf), device="cpu"))
    return state


def _find_attn_keys(state: dict, block_idx: int) -> dict:
    """
    Return a dict of role → weight_tensor for self-attention in block i.
    Tries multiple diffusers key conventions.

    Returns {} if the block is not found.
    """
    # Try primary convention (diffusers >=0.28 LTXVideoTransformer3DModel)
    prefix = f"transformer_blocks.{block_idx}.attn1"
    candidates = {
        "q_w": [f"{prefix}.to_q.weight"],
        "q_b": [f"{prefix}.to_q.bias"],
        "k_w": [f"{prefix}.to_k.weight"],
        "k_b": [f"{prefix}.to_k.bias"],
        "v_w": [f"{prefix}.to_v.weight"],
        "v_b": [f"{prefix}.to_v.bias"],
        "o_w": [f"{prefix}.to_out.0.weight"],
        "o_b": [f"{prefix}.to_out.0.bias"],
    }
    result = {}
    for role, keys in candidates.items():
        for k in keys:
            if k in state:
                result[role] = state[k]
                break
    return result


def _find_ffn_keys(state: dict, block_idx: int) -> dict:
    """
    Return FFN weight tensors for block i.

    LTX-Video uses GEGLU which has gate+proj fused or separate.
    Handles both:
      .ff.net.0.proj.weight  (fused gate+proj, shape [2*inner, dim])
      .ff.net.0.weight       (unfused)
      .ff.net.2.weight       (down projection)
    """
    prefix = f"transformer_blocks.{block_idx}.ff"
    result = {}

    for key_up in [
        f"{prefix}.net.0.proj.weight",
        f"{prefix}.net.0.weight",
    ]:
        if key_up in state:
            result["up_w"] = state[key_up]
            break

    for key_up_b in [
        f"{prefix}.net.0.proj.bias",
        f"{prefix}.net.0.bias",
    ]:
        if key_up_b in state:
            result["up_b"] = state[key_up_b]
            break

    for key_down in [
        f"{prefix}.net.2.weight",
        f"{prefix}.net.1.weight",
    ]:
        if key_down in state:
            result["down_w"] = state[key_down]
            break

    for key_down_b in [
        f"{prefix}.net.2.bias",
        f"{prefix}.net.1.bias",
    ]:
        if key_down_b in state:
            result["down_b"] = state[key_down_b]
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Build ExpressionModel initialised from LTX-Video
# ─────────────────────────────────────────────────────────────────────────────

def build_init_state_dict(
    ltx_state:  dict,
    dim:        int = 1536,
    ffn_dim:    int = 8960,
    num_layers: int = 30,
    verbose:    bool = False,
) -> dict:
    """
    Return a state_dict for ExpressionModel pre-populated from LTX-Video weights.

    Unmapped layers keep their default PyTorch init (zero for biases,
    Kaiming uniform for Linear weights).
    """
    # Import ExpressionModel to get the correct default state_dict
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference.model import ExpressionModel

    model = ExpressionModel(dim=dim, ffn_dim=ffn_dim, num_layers=num_layers)
    our_state = model.state_dict()

    mapped = 0
    skipped = 0

    # Discover LTX-Video dimension from first self-attn Q weight
    ltx_dim = None
    for i in range(50):
        k = f"transformer_blocks.{i}.attn1.to_q.weight"
        if k in ltx_state:
            ltx_dim = ltx_state[k].shape[0]
            break
    if ltx_dim is None:
        logger.warning("Could not find self-attention weights in LTX state dict; keys available:")
        sample_keys = list(ltx_state.keys())[:20]
        for k in sample_keys:
            logger.warning(f"  {k}")
        return our_state

    logger.info(f"LTX-Video inner_dim detected: {ltx_dim}  →  compressing to {dim}")

    # Count actual LTX blocks
    ltx_blocks = sum(
        1 for k in ltx_state
        if k.startswith("transformer_blocks.") and ".attn1.to_q.weight" in k
    )
    logger.info(f"LTX-Video has {ltx_blocks} transformer blocks; our model has {num_layers}")

    # ── Transformer blocks ────────────────────────────────────────────────────
    for our_i in range(num_layers):
        # Map our block i to LTX block j (repeat last LTX block for extra layers)
        ltx_i = min(our_i, ltx_blocks - 1)

        # Self-attention
        attn = _find_attn_keys(ltx_state, ltx_i)
        for role, our_key in [
            ("q_w", f"blocks.{our_i}.self_attn.q.weight"),
            ("q_b", f"blocks.{our_i}.self_attn.q.bias"),
            ("k_w", f"blocks.{our_i}.self_attn.k.weight"),
            ("k_b", f"blocks.{our_i}.self_attn.k.bias"),
            ("v_w", f"blocks.{our_i}.self_attn.v.weight"),
            ("v_b", f"blocks.{our_i}.self_attn.v.bias"),
            ("o_w", f"blocks.{our_i}.self_attn.o.weight"),
            ("o_b", f"blocks.{our_i}.self_attn.o.bias"),
        ]:
            if role not in attn:
                skipped += 1
                continue
            src = attn[role]
            dst = our_state[our_key]
            if src.ndim == 2:
                our_state[our_key] = _fit(src, dst.shape[0], dst.shape[1]).to(dst.dtype)
            else:  # bias (1-D)
                our_state[our_key] = _fit_bias(src, dst.shape[0]).to(dst.dtype)
            mapped += 1

        if verbose:
            n_found = sum(1 for r in ["q_w", "k_w", "v_w", "o_w"] if r in attn)
            logger.debug(f"  block {our_i} (ltx {ltx_i}): self-attn {n_found}/4 weights found")

        # FFN
        ffn = _find_ffn_keys(ltx_state, ltx_i)

        if "up_w" in ffn:
            src = ffn["up_w"]
            # GEGLU: weight is (2*intermediate, in_dim); take first half
            if src.shape[0] > ffn_dim * 2:
                src = src[:src.shape[0] // 2, :]  # gate half
            elif src.shape[0] > ffn_dim:
                src = src[:ffn_dim, :]
            dst = our_state[f"blocks.{our_i}.ffn.0.weight"]
            our_state[f"blocks.{our_i}.ffn.0.weight"] = _fit(src, dst.shape[0], dst.shape[1]).to(dst.dtype)
            mapped += 1

        if "up_b" in ffn:
            src = ffn["up_b"]
            if src.shape[0] > ffn_dim:
                src = src[:ffn_dim]
            dst = our_state[f"blocks.{our_i}.ffn.0.bias"]
            our_state[f"blocks.{our_i}.ffn.0.bias"] = _fit_bias(src, dst.shape[0]).to(dst.dtype)
            mapped += 1

        if "down_w" in ffn:
            src = ffn["down_w"]
            dst = our_state[f"blocks.{our_i}.ffn.2.weight"]
            our_state[f"blocks.{our_i}.ffn.2.weight"] = _fit(src, dst.shape[0], dst.shape[1]).to(dst.dtype)
            mapped += 1

        if "down_b" in ffn:
            src = ffn["down_b"]
            dst = our_state[f"blocks.{our_i}.ffn.2.bias"]
            our_state[f"blocks.{our_i}.ffn.2.bias"] = _fit_bias(src, dst.shape[0]).to(dst.dtype)
            mapped += 1

    total = len(our_state)
    logger.info(f"Weight mapping: {mapped}/{total} tensors transferred from LTX-Video")
    logger.info(f"  ({skipped} skipped — keys not found)  ({total - mapped - skipped} random-init)")
    return our_state


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Initialise ExpressionModel from LTX-Video weights"
    )
    parser.add_argument("--ltx-dir",  required=True, help="LTX-Video transformer dir (ltx-weights/transformer)")
    parser.add_argument("--output",   default="naalanda-weights/", help="Output root directory")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Loading LTX-Video weights from {args.ltx_dir} …")
    ltx_state = _load_ltx_state_dict(args.ltx_dir)
    logger.info(f"  {len(ltx_state)} tensors loaded")

    logger.info("Building initialised ExpressionModel state dict …")
    init_state = build_init_state_dict(ltx_state, verbose=args.verbose)

    # Save output
    out_dir = Path(args.output) / "bithuman-expression" / "Model_Lite"
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_path = out_dir / "naalanda_expression_dit.safetensors"
    logger.info(f"Saving → {weight_path}")
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in init_state.items()}, str(weight_path))

    # Write config.json (same format as bh-weights/Model_Lite/config.json)
    config = {
        "_class_name": "NaalandaExpressionModel",
        "dim": 1536,
        "in_dim": 256,
        "out_dim": 128,
        "freq_dim": 256,
        "ffn_dim": 8960,
        "num_heads": 12,
        "num_layers": 30,
        "text_dim": 4096,
        "vae_stride": [8, 32, 32],
        "width": 512,
        "height": 512,
        "fps": 25,
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config  → {config_path}")

    # Remind user to copy/symlink VAE
    vae_src = Path(args.ltx_dir).parent / "vae"
    vae_dst = Path(args.output) / "bithuman-expression" / "VAE_LTX"
    if vae_src.exists() and not vae_dst.exists():
        import shutil
        shutil.copytree(str(vae_src), str(vae_dst))
        logger.info(f"VAE     → {vae_dst}")
    elif not vae_src.exists():
        logger.warning(f"VAE not found at {vae_src}; copy it manually to {vae_dst}")

    logger.info("Done — run the smoke-test:")
    logger.info(f"  PYTHONPATH=app python train/init_weights.py --ltx-dir ... (rerun for re-init)")
    logger.info(f"  Or test: PYTHONPATH=app python -c \"import torch; sys.path.insert(0,'app'); ...")


if __name__ == "__main__":
    main()
