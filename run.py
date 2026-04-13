#!/usr/bin/env python3
"""
Standalone inference CLI for bitHuman Expression Avatar.

Usage:
    python run.py --image face.jpg --audio speech.wav
    python run.py --image face.jpg --audio speech.wav --output out.mp4 --steps 20
    python run.py --image face.jpg --audio speech.wav --device cpu   # CPU fallback
"""
import argparse
import sys
import os

# Make sure we can find the inference package from wherever this script is run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video from a face image + audio"
    )
    parser.add_argument("--image",       required=True,  help="Path to face image (JPG/PNG)")
    parser.add_argument("--audio",       required=True,  help="Path to audio file (WAV/MP3/etc.)")
    parser.add_argument("--output",      default="output.mp4", help="Output MP4 path")
    parser.add_argument("--steps",       default=20, type=int,   help="Denoising steps (20=balanced, 50=high quality)")
    parser.add_argument("--seed",        default=42, type=int,   help="Random seed")
    parser.add_argument("--device",      default="cuda",         help="cuda or cpu")
    parser.add_argument("--weights-dir", default="bh-weights",   help="Path to bh-weights directory")
    parser.add_argument("--wav2vec-dir", default="app/bundled/wav2vec2-base-960h",
                                                                  help="Path to wav2vec2 model")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[run] CUDA not available, falling back to CPU")
        args.device = "cpu"

    dtype = torch.float16 if args.device == "cuda" else torch.float32

    from inference.pipeline import ExpressionPipeline

    pipe = ExpressionPipeline(
        weights_dir=args.weights_dir,
        wav2vec_dir=args.wav2vec_dir,
        device=args.device,
        dtype=dtype,
    )

    pipe.generate(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        num_steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
