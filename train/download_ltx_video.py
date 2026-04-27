#!/usr/bin/env python3
"""
Download open-source LTX-Video weights from HuggingFace (Lightricks/LTX-Video).
Apache 2.0 licence — no IP restrictions.

Usage:
    python train/download_ltx_video.py [--output ltx-weights/]
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ltx-weights/", help="Download destination directory")
    parser.add_argument("--token", default=None, help="HuggingFace access token (optional)")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub first:  pip install huggingface-hub")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Lightricks/LTX-Video → {output} …")
    print("This is ~12 GB; grab a coffee.")

    path = snapshot_download(
        repo_id="Lightricks/LTX-Video",
        local_dir=str(output),
        # Skip large quantised variants to save space
        ignore_patterns=["*.gguf", "ltx-video-2b-*", "*distilled*"],
        token=args.token,
    )
    print(f"\nDone. Weights at: {path}")

    # Quick sanity check
    transformer_dir = Path(path) / "transformer"
    vae_dir         = Path(path) / "vae"
    if transformer_dir.exists():
        print(f"  transformer/ ... OK ({sum(1 for _ in transformer_dir.iterdir())} files)")
    else:
        print("  WARNING: transformer/ not found — check repo structure")
    if vae_dir.exists():
        print(f"  vae/           ... OK ({sum(1 for _ in vae_dir.iterdir())} files)")
    else:
        print("  WARNING: vae/ not found — check repo structure")


if __name__ == "__main__":
    main()
