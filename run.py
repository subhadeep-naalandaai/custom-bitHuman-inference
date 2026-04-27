#!/usr/bin/env python3
"""
Naalanda talking-head inference — image + audio → video.

Uses inference/ for the standalone pipeline and naalanda-weights/ (or bh-weights/)
for model weights.

Examples:
  # With trained naalanda weights:
  python run.py --image face.jpg --audio speech.wav --weights-dir naalanda-weights/

  # With bh-weights (original bitHuman weights):
  python run.py --image face.jpg --audio speech.wav --weights-dir bh-weights/

  # CPU fallback:
  python run.py --image face.jpg --audio speech.wav --device cpu
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Naalanda talking-head generation from image + audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image",        required=True,           help="Reference face image (JPG/PNG)")
    parser.add_argument("--audio",        required=True,           help="Audio file (WAV/MP3/…)")
    parser.add_argument("--output",       default="output.mp4",    help="Output video path (default: output.mp4)")
    parser.add_argument("--weights-dir",  default="naalanda-weights", help="Weights root directory")
    parser.add_argument("--wav2vec-dir",  default=None,            help="Wav2Vec2 directory (auto-detected)")
    parser.add_argument("--steps",        default=20,  type=int,   help="Denoising steps (default: 20)")
    parser.add_argument("--guidance",     default=3.5, type=float, help="CFG guidance scale (default: 3.5)")
    parser.add_argument("--fps",          default=25,  type=int,   help="Output FPS (default: 25)")
    parser.add_argument("--seed",         default=42,  type=int,   help="Random seed (default: 42)")
    parser.add_argument("--device",       default=None,            help="cuda or cpu (auto-detected)")
    args = parser.parse_args()

    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[run] CUDA not available, falling back to CPU")
        args.device = "cpu"

    weights_dir = Path(args.weights_dir)
    ckpt_dir    = weights_dir / "bithuman-expression" / "Model_Lite"

    if not ckpt_dir.exists():
        sys.exit(
            f"[run] Weights not found at {ckpt_dir}.\n"
            f"  For naalanda weights: python train/export.py --checkpoint <ckpt.pt> --output {weights_dir}\n"
            f"  For LTX-Video init:  python train/init_weights.py --ltx-dir ltx-weights/transformer --output {weights_dir}"
        )

    # Auto-detect bundled wav2vec2
    wav2vec_dir = args.wav2vec_dir
    if wav2vec_dir is None:
        bundled = Path(__file__).parent / "app" / "bundled" / "wav2vec2-base-960h"
        if bundled.exists():
            wav2vec_dir = str(bundled)

    # Load pipeline from inference/
    from inference.pipeline import (
        get_pipeline,
        get_audio_embedding,
        get_base_data,
        get_infer_params,
        run_pipeline,
    )

    print(f"[run] Loading pipeline from {ckpt_dir} …")
    pipeline = get_pipeline(
        ckpt_dir    = str(ckpt_dir),
        models_dir  = str(weights_dir),
        wav2vec_dir = wav2vec_dir,
        device      = args.device,
    )

    torch.manual_seed(args.seed)

    # Load audio
    import torchaudio
    waveform, sr = torchaudio.load(args.audio)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.squeeze(0)

    duration   = waveform.shape[0] / 16000
    num_frames = max(1, int(duration * args.fps))
    print(f"[run] Audio: {duration:.2f}s  →  {num_frames} frames at {args.fps} fps")

    # Encode audio
    audio_emb, _ = get_audio_embedding(pipeline, waveform, sample_rate=16000, num_frames=num_frames)

    # Load reference image
    base_data = get_base_data(pipeline, args.image)

    # Build inference params
    params = get_infer_params(
        pipeline,
        cached_audio_duration = duration,
        tgt_fps               = args.fps,
        num_inference_steps   = args.steps,
        guidance_scale        = args.guidance,
        seed                  = args.seed,
    )

    # Run denoising + decode
    print(f"[run] Running {args.steps}-step denoising …")
    frames, timings = run_pipeline(pipeline, audio_emb, base_data, params)

    print(
        f"[run] Generated {len(frames)} frames  "
        f"(denoise {timings['denoise_ms']:.0f}ms  "
        f"decode {timings['decode_ms']:.0f}ms)"
    )

    _save_video(frames, args.output, args.fps)
    print(f"[run] Saved → {args.output}")


def _save_video(frames, output_path: str, fps: int):
    """Write list of (H, W, 3) uint8 arrays to MP4."""
    try:
        import imageio
        imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    except ImportError:
        import torch, numpy as np
        import torchvision.io as tvio
        tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
        tvio.write_video(output_path, tensor, fps=fps, video_codec="libx264", options={"crf": "18"})


if __name__ == "__main__":
    main()
