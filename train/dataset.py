"""
TalkingHeadDataset — loads pre-processed talking-head clips from disk.

Each clip directory (produced by preprocess.py) contains:
    latents.npy      float32  (128, T_lat, H_lat, W_lat)  — VAE-encoded video
    audio.npy        float32  (T_frames, 768)              — Wav2Vec2 features
    meta.json                 {"fps": 25, "duration": ..., "source": ...}

The dataset returns per-clip:
    ref_latent:     (128, 1, H_lat, W_lat)   — first latent frame (reference)
    video_latents:  (128, T_lat, H_lat, W_lat) — full clip latents (ground truth)
    audio_emb:      (T_frames, 768)            — Wav2Vec2 features aligned with frames
    t_lat:          int                        — number of latent time steps

Usage:
    ds = TalkingHeadDataset(["data/preprocessed/voxceleb2", ...], clip_frames=(2,8), fps=25)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
"""
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TalkingHeadDataset(Dataset):
    """
    Dataset of pre-processed (VAE-encoded + Wav2Vec2-encoded) talking-head clips.

    Args:
        preprocessed_dirs: list of directories produced by preprocess.py
        clip_frames:        (min_frames, max_frames) — uniformly sampled per clip
        fps:                video frame rate (used to compute latent frame count)
        val:                if True, use the last val_fraction of each dir
        val_fraction:       fraction of data reserved for validation
    """

    def __init__(
        self,
        preprocessed_dirs: List[str],
        clip_frames: Tuple[int, int] = (50, 200),  # 2–8 sec at 25 fps
        fps: int = 25,
        val: bool = False,
        val_fraction: float = 0.005,
    ):
        self.fps         = fps
        self.clip_frames = clip_frames
        self.val         = val

        # Collect all clip directories
        all_clips: List[Path] = []
        for d in preprocessed_dirs:
            root = Path(d)
            if not root.exists():
                continue
            clips = sorted(p for p in root.iterdir() if (p / "latents.npy").exists())
            all_clips.extend(clips)

        if not all_clips:
            raise RuntimeError(
                f"No preprocessed clips found in: {preprocessed_dirs}\n"
                f"Run: python train/preprocess.py --source <video_dir> --output <preprocessed_dir>"
            )

        # Split train / val deterministically
        n_val   = max(1, int(len(all_clips) * val_fraction))
        if val:
            self.clips = all_clips[-n_val:]
        else:
            self.clips = all_clips[:-n_val]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip_dir = self.clips[idx]

        latents = torch.from_numpy(np.load(clip_dir / "latents.npy"))  # (128, T_lat, H, W)
        audio   = torch.from_numpy(np.load(clip_dir / "audio.npy"))    # (T_frames, 768)

        # Read clip metadata
        meta_path = clip_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        C, T_lat, H_lat, W_lat = latents.shape
        T_frames = audio.shape[0]

        # Sample a random sub-clip of target length
        min_frames, max_frames = self.clip_frames
        target_frames = random.randint(min_frames, min(max_frames, T_frames))

        # Compute corresponding latent frame count: T_lat = (T_frames - 1) // 8 + 1
        target_lat = max(1, (target_frames - 1) // 8 + 1)
        target_lat = min(target_lat, T_lat)

        # Random start within clip
        max_start_lat = max(0, T_lat - target_lat)
        start_lat     = random.randint(0, max_start_lat)
        end_lat       = start_lat + target_lat

        # Corresponding frame window
        start_frame = start_lat * 8
        end_frame   = min(start_frame + target_frames, T_frames)

        video_latents = latents[:, start_lat:end_lat, :, :].float()   # (128, t, H, W)
        audio_emb     = audio[start_frame:end_frame, :].float()        # (t_frames, 768)

        # Pad audio to target_frames if shorter
        if audio_emb.shape[0] < target_frames:
            pad = torch.zeros(target_frames - audio_emb.shape[0], 768)
            audio_emb = torch.cat([audio_emb, pad], dim=0)

        # Reference latent: use first frame of the full clip (not the sub-clip)
        ref_latent = latents[:, 0:1, :, :].float()  # (128, 1, H, W)

        return {
            "ref_latent":    ref_latent,     # (128, 1, H_lat, W_lat)
            "video_latents": video_latents,  # (128, T_lat, H_lat, W_lat)
            "audio_emb":     audio_emb,      # (T_frames, 768)
            "t_lat":         video_latents.shape[1],
            "t_frames":      audio_emb.shape[0],
            "clip_id":       str(clip_dir),
        }


def collate_fn(batch: List[dict]) -> Optional[dict]:
    """
    Pad batch items to the same (T_lat, T_frames) within the batch.
    Items with very different lengths are padded with zeros.
    """
    if not batch:
        return None

    max_t_lat    = max(b["t_lat"]    for b in batch)
    max_t_frames = max(b["t_frames"] for b in batch)

    ref_latents    = []
    video_latents  = []
    audio_embs     = []
    t_lats         = []
    t_frames_list  = []

    for b in batch:
        rl = b["ref_latent"]          # (128, 1, H, W)
        vl = b["video_latents"]       # (128, t, H, W)
        ae = b["audio_emb"]           # (t_frames, 768)

        # Pad video latents along time dim
        if vl.shape[1] < max_t_lat:
            pad = torch.zeros(vl.shape[0], max_t_lat - vl.shape[1], *vl.shape[2:])
            vl  = torch.cat([vl, pad], dim=1)

        # Pad audio along time dim
        if ae.shape[0] < max_t_frames:
            pad = torch.zeros(max_t_frames - ae.shape[0], ae.shape[1])
            ae  = torch.cat([ae, pad], dim=0)

        ref_latents.append(rl)
        video_latents.append(vl)
        audio_embs.append(ae)
        t_lats.append(b["t_lat"])
        t_frames_list.append(b["t_frames"])

    return {
        "ref_latents":   torch.stack(ref_latents),    # (B, 128, 1,     H, W)
        "video_latents": torch.stack(video_latents),  # (B, 128, T_lat, H, W)
        "audio_embs":    torch.stack(audio_embs),     # (B, T_frames, 768)
        "t_lats":        torch.tensor(t_lats),
        "t_frames":      torch.tensor(t_frames_list),
    }
