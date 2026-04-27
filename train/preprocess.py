#!/usr/bin/env python3
"""
Offline preprocessing: encode a directory of talking-head videos into
VAE latents + Wav2Vec2 audio features and save as NumPy arrays.

Each output clip directory contains:
    latents.npy    float32   (128, T_lat, 16, 16)  — VAE-encoded video
    audio.npy      float32   (T_frames, 768)        — Wav2Vec2 features
    meta.json                {"fps", "duration", "source", ...}

This step is done ONCE before training.  Training reads from the cached .npy
files so the VAE and Wav2Vec2 are never loaded during the training loop.

Usage:
    python train/preprocess.py \
        --source  /data/voxceleb2/mp4/ \
        --output  data/preprocessed/voxceleb2/ \
        [--resolution 512] \
        [--fps 25] \
        [--max-clips 10000] \
        [--workers 4]

Requirements:
    pip install opencv-python torchaudio decord
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make inference/ importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Video loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_video_frames(path: str, target_fps: int = 25, resolution: int = 512, max_duration: float = 30.0):
    """
    Load video file and return (T, 3, H, W) float32 tensor in [-1, 1].
    Uses decord for fast decode; falls back to OpenCV.
    max_duration caps frames before loading to avoid RAM exhaustion on long clips.
    """
    max_out = max(1, int(max_duration * target_fps))

    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(path, width=resolution, height=resolution)
        src_fps = vr.get_avg_fps()
        if src_fps <= 0:
            src_fps = target_fps
        n_frames = len(vr)
        duration = n_frames / src_fps
        n_out    = min(max_out, max(1, int(duration * target_fps)))
        indices  = (np.arange(n_out) * src_fps / target_fps).astype(int).clip(0, n_frames - 1)
        frames   = vr.get_batch(indices)     # (T, H, W, 3) uint8
        frames   = frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # (T, 3, H, W) in [-1,1]
        actual_duration = min(duration, max_duration)
        return frames, actual_duration
    except Exception:
        pass

    # OpenCV fallback
    import cv2
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames_list = []
    max_src = int(max_duration * src_fps)
    while len(frames_list) < max_src:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
        frames_list.append(frame)
    cap.release()
    if not frames_list:
        return None, 0.0
    frames = torch.from_numpy(np.stack(frames_list))  # (T_src, H, W, 3) uint8
    duration = len(frames_list) / src_fps
    n_out    = min(max_out, max(1, int(duration * target_fps)))
    indices  = (np.arange(n_out) * src_fps / target_fps).astype(int).clip(0, len(frames_list) - 1)
    frames   = frames[indices].permute(0, 3, 1, 2).float() / 127.5 - 1.0  # (T, 3, H, W) [-1,1]
    return frames, duration


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio(path: str, target_sr: int = 16000):
    """Return mono waveform at target_sr."""
    import torchaudio
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)  # (T_audio,)


# ─────────────────────────────────────────────────────────────────────────────
# Face crop
# ─────────────────────────────────────────────────────────────────────────────

def _face_crop_frames(frames: torch.Tensor, resolution: int = 512) -> torch.Tensor:
    """
    Apply face detection on the first frame and crop all frames to the face region.
    Falls back to full-frame if detection fails.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
        from bithuman_expression.utils.facecrop import FaceCropper
        cropper = FaceCropper(target_size=resolution)
        # Convert first frame from tensor to PIL/numpy for detection
        first = ((frames[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        bbox  = cropper.detect(first)
        if bbox is not None:
            frames_np = ((frames.permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            cropped   = cropper.crop_and_resize(frames_np, bbox, resolution)
            return torch.from_numpy(cropped).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    except Exception:
        pass
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Main encode function
# ─────────────────────────────────────────────────────────────────────────────

def encode_clip(
    video_path: str,
    audio_path: Optional[str],
    output_dir: Path,
    vae,
    wav2vec,
    device: str = "cuda",
    fps: int = 25,
    resolution: int = 512,
    min_duration: float = 1.5,
    max_duration: float = 30.0,
) -> bool:
    """
    Encode one video clip → latents.npy + audio.npy + meta.json.
    Returns True on success, False on skip.
    """
    if output_dir.exists() and (output_dir / "latents.npy").exists():
        return True  # already done

    # Load video (max_duration cap applied inside to avoid RAM spike on long clips)
    frames, duration = _load_video_frames(video_path, fps, resolution, max_duration)
    if frames is None or duration < min_duration:
        return False

    # Apply face crop
    frames = _face_crop_frames(frames, resolution)

    # Load audio (same file or separate)
    audio_file = audio_path or video_path  # many datasets embed audio in video
    try:
        waveform = _load_audio(audio_file)
    except Exception:
        waveform = torch.zeros(int(duration * 16000))

    n_frames = frames.shape[0]

    # VAE encode: expects (B, 3, T, H, W) in [-1, 1], returns (128, T_lat, H_lat, W_lat)
    video_tensor = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device, dtype=torch.bfloat16)
    # (1, 3, T, H, W)
    with torch.no_grad():
        latents = vae.encode(video_tensor)  # (128, T_lat, H_lat, W_lat) — no batch dim
    latents = latents.cpu().float().numpy()

    # Wav2Vec2 encode: (B, T_audio) at 16 kHz → (B, n_frames, 768)
    from inference.audio import encode_audio
    audio_emb = encode_audio(wav2vec, waveform, n_frames, device)  # (1, n_frames, 768)
    audio_np  = audio_emb[0].cpu().float().numpy()                  # (n_frames, 768)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "latents.npy", latents.astype(np.float32))
    np.save(output_dir / "audio.npy",   audio_np.astype(np.float32))
    with open(output_dir / "meta.json", "w") as f:
        json.dump({
            "fps":         fps,
            "duration":    duration,
            "n_frames":    n_frames,
            "t_lat":       latents.shape[1],
            "source":      video_path,
        }, f)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess talking-head videos for training")
    parser.add_argument("--source",     required=True, help="Input video directory")
    parser.add_argument("--output",     required=True, help="Output preprocessed directory")
    parser.add_argument("--vae-dir",    default="bh-weights/bithuman-expression/VAE_LTX",
                        help="VAE weights directory")
    parser.add_argument("--wav2vec-dir",default="app/bundled/wav2vec2-base-960h",
                        help="Wav2Vec2 checkpoint directory")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--fps",        default=25, type=int)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--max-clips",    default=0,   type=int,   help="Limit number of clips (0=all)")
    parser.add_argument("--max-duration", default=8.0, type=float, help="Max clip duration in seconds (default: 8.0)")
    parser.add_argument("--extensions",   default="mp4,avi,mkv,mov,webm")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VAE
    logger.info("Loading VAE …")
    from inference.vae import LtxVAE
    vae = LtxVAE(args.vae_dir, device=args.device, dtype=torch.bfloat16)

    # Load Wav2Vec2
    logger.info("Loading Wav2Vec2 …")
    from inference.audio import load_wav2vec
    wav2vec = load_wav2vec(args.wav2vec_dir, args.device)

    # Collect video files
    exts = {f".{e.strip()}" for e in args.extensions.split(",")}
    video_files = sorted(
        p for p in Path(args.source).rglob("*")
        if p.suffix.lower() in exts
    )
    if args.max_clips > 0:
        video_files = video_files[:args.max_clips]
    logger.info(f"Found {len(video_files)} video files in {args.source}")

    output_root = Path(args.output)
    ok = skipped = failed = 0

    for i, vf in enumerate(video_files):
        clip_id  = vf.stem
        out_dir  = output_root / clip_id
        try:
            result = encode_clip(
                video_path=str(vf),
                audio_path=None,
                output_dir=out_dir,
                vae=vae,
                wav2vec=wav2vec,
                device=args.device,
                fps=args.fps,
                resolution=args.resolution,
                max_duration=args.max_duration,
            )
            if result:
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            logger.warning(f"[{i+1}/{len(video_files)}] Failed {vf.name}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{len(video_files)} — ok={ok} skip={skipped} fail={failed}")

    logger.info(f"Done: {ok} encoded, {skipped} skipped, {failed} failed → {output_root}")


if __name__ == "__main__":
    main()
