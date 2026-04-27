#!/usr/bin/env python3
"""
Export a trained ExpressionModel checkpoint to the naalanda-weights/ format
so it can be used with the existing inference pipeline (run.py, app/).

The exported layout mirrors bh-weights/ exactly so that --weights-dir can be
pointed at either directory without any inference-code changes.

Usage:
    python train/export.py \
        --checkpoint checkpoints/step_0100000.pt \
        --output     naalanda-weights/ \
        [--ltx-vae   ltx-weights/vae]    # optional: copy VAE from LTX-Video
        [--bh-vae    bh-weights/bithuman-expression/VAE_LTX]  # or copy from bh-weights
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export trained ExpressionModel weights")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint from train.py")
    parser.add_argument("--output",     default="naalanda-weights/", help="Output root directory")
    parser.add_argument("--ltx-vae",    default=None, help="LTX-Video vae/ directory to copy")
    parser.add_argument("--bh-vae",     default="bh-weights/bithuman-expression/VAE_LTX",
                        help="Fallback VAE source")
    args = parser.parse_args()

    # Load checkpoint
    logger.info(f"Loading checkpoint {args.checkpoint} …")
    ckpt   = torch.load(args.checkpoint, map_location="cpu")
    state  = ckpt["model"] if "model" in ckpt else ckpt
    config = ckpt.get("config", {})
    step   = ckpt.get("step", 0)
    logger.info(f"  step={step}, {len(state)} tensors")

    # Verify weights load into ExpressionModel
    from inference.model import ExpressionModel
    mcfg = config.get("model", {})
    model = ExpressionModel(
        dim          = mcfg.get("dim",          1536),
        in_channels  = mcfg.get("in_channels",   256),
        out_channels = mcfg.get("out_channels",  128),
        freq_dim     = mcfg.get("freq_dim",      256),
        ffn_dim      = mcfg.get("ffn_dim",      8960),
        num_heads    = mcfg.get("num_heads",      12),
        num_layers   = mcfg.get("num_layers",     30),
        audio_dim    = mcfg.get("audio_dim",     768),
        text_dim     = mcfg.get("text_dim",     4096),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"  Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
    logger.info(f"  State dict verified — missing={len(missing)}, unexpected={len(unexpected)}")

    # Create output directory layout
    model_lite_dir = Path(args.output) / "bithuman-expression" / "Model_Lite"
    model_lite_dir.mkdir(parents=True, exist_ok=True)

    # Save weights as safetensors
    from safetensors.torch import save_file
    weight_path = model_lite_dir / "naalanda_expression_dit.safetensors"
    logger.info(f"Saving weights → {weight_path}")
    save_file({k: v.contiguous() for k, v in state.items()}, str(weight_path))

    # Write config.json (compatible with bh-weights/Model_Lite/config.json)
    out_config = {
        "_class_name":          "NaalandaExpressionModel",
        "_diffusers_version":   "0.36.0",
        "dim":                  mcfg.get("dim",         1536),
        "in_dim":               mcfg.get("in_channels",  256),
        "out_dim":              mcfg.get("out_channels", 128),
        "freq_dim":             mcfg.get("freq_dim",     256),
        "ffn_dim":              mcfg.get("ffn_dim",     8960),
        "num_heads":            mcfg.get("num_heads",     12),
        "num_layers":           mcfg.get("num_layers",    30),
        "text_dim":             mcfg.get("text_dim",    4096),
        "vae_stride":           [8, 32, 32],
        "width":                512,
        "height":               512,
        "fps":                  25,
    }
    config_path = model_lite_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(out_config, f, indent=2)
    logger.info(f"Config        → {config_path}")

    # Copy VAE
    vae_dst = Path(args.output) / "bithuman-expression" / "VAE_LTX"
    vae_src = None
    if args.ltx_vae and Path(args.ltx_vae).exists():
        vae_src = Path(args.ltx_vae)
    elif Path(args.bh_vae).exists():
        vae_src = Path(args.bh_vae)

    if vae_src and not vae_dst.exists():
        shutil.copytree(str(vae_src), str(vae_dst))
        logger.info(f"VAE           → {vae_dst}  (copied from {vae_src})")
    elif vae_dst.exists():
        logger.info(f"VAE           → {vae_dst}  (already present)")
    else:
        logger.warning(f"VAE not found — copy manually to {vae_dst}")

    logger.info("Export complete.")
    logger.info(f"Test with:")
    logger.info(f"  python run.py --image face.jpg --audio speech.wav --weights-dir {args.output}")


if __name__ == "__main__":
    main()
