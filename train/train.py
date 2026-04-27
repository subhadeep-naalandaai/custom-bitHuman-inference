#!/usr/bin/env python3
"""
Flow-matching training loop for the Naalanda ExpressionModel.

Trains the ExpressionModel (ExpressionModel DiT, dim=1536, 30 layers, 12 heads) to generate
talking-head video latents conditioned on audio (Wav2Vec2 features) and a reference image latent.

Loss: flow-matching MSE on the velocity field
  x_t = (1-t)*x_0 + t*noise       (noisy latent at time t)
  v*  = noise - x_0                (target velocity)
  v   = model(cat([x_t, ref], dim=1), t, audio_emb)
  loss = MSE(v, v*)

Supports single-GPU and multi-GPU (torchrun) training.

Usage (single GPU):
    PYTHONPATH=app python train/train.py --config train/config.yaml

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --nproc_per_node=4 train/train.py --config train/config.yaml
"""
import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Make inference/ importable (project root)
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


# ─────────────────────────────────────────────────────────────────────────────
# Cosine LR with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_lr(step: int, warmup: int, total: int, lr: float, min_lr: float = 1e-7) -> float:
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def _train_step(model, batch: dict, cfg: dict, device: str) -> torch.Tensor:
    """
    Single flow-matching training step.

    batch keys (from collate_fn):
        ref_latents   : (B, 128, 1, H, W)
        video_latents : (B, 128, T, H, W)
        audio_embs    : (B, T_frames, 768)

    Returns scalar loss.
    """
    ref     = batch["ref_latents"].to(device)     # (B, 128, 1, H, W)
    x_clean = batch["video_latents"].to(device)   # (B, 128, T, H, W)
    audio   = batch["audio_embs"].to(device)      # (B, T_frames, 768)

    B, C, T, H, W = x_clean.shape

    # Expand reference to match video length
    ref_exp = ref.expand(B, C, T, H, W)  # (B, 128, T, H, W)

    # Sample timestep t ~ U(0, 1)
    t = torch.rand(B, device=device, dtype=x_clean.dtype)

    # Flow-matching interpolation: x_t = (1-t)*x_0 + t*noise
    noise   = torch.randn_like(x_clean)
    t_view  = t.view(B, 1, 1, 1, 1)
    x_noisy = (1.0 - t_view) * x_clean + t_view * noise

    # Target velocity: v* = noise - x_0
    v_target = noise - x_clean

    # Classifier-free guidance dropout: zero out audio with cfg_dropout_prob
    cfg_p = float(cfg.get("training", {}).get("cfg_dropout_prob", 0.1))
    if cfg_p > 0:
        drop_mask = (torch.rand(B, device=device) < cfg_p).view(B, 1, 1)
        audio = audio.masked_fill(drop_mask, 0.0)

    # Model input: concatenate noisy latent + reference latent
    x_input = torch.cat([x_noisy, ref_exp], dim=1)  # (B, 256, T, H, W)

    # Forward pass
    v_pred = model(x_input, t, audio)  # (B, 128, T, H, W)

    return F.mse_loss(v_pred, v_target)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    parser.add_argument("--resume", default=None, help="Override resume_from in config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Distributed setup ─────────────────────────────────────────────────────
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(_local_rank())

    device = f"cuda:{_local_rank()}" if torch.cuda.is_available() else "cpu"
    seed   = cfg["training"].get("seed", 42) + _local_rank()
    torch.manual_seed(seed)

    if _is_main():
        logger.info(f"World size: {_world_size()}")
        logger.info(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    from inference.model import ExpressionModel
    mcfg = cfg["model"]
    model = ExpressionModel(
        dim                    = mcfg["dim"],
        in_channels            = mcfg["in_channels"],
        out_channels           = mcfg["out_channels"],
        freq_dim               = mcfg["freq_dim"],
        ffn_dim                = mcfg["ffn_dim"],
        num_heads              = mcfg["num_heads"],
        num_layers             = mcfg["num_layers"],
        audio_dim              = mcfg["audio_dim"],
        text_dim               = mcfg["text_dim"],
        gradient_checkpointing = True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    if _is_main():
        logger.info(f"ExpressionModel: {n_params:.2f}B parameters")

    if dist.is_initialized():
        model = DDP(model, device_ids=[_local_rank()])

    _model = model.module if isinstance(model, DDP) else model

    # ── Dataset ───────────────────────────────────────────────────────────────
    sys.path.insert(0, str(_ROOT))
    from train.dataset import TalkingHeadDataset, collate_fn
    dcfg = cfg["data"]
    train_ds = TalkingHeadDataset(
        preprocessed_dirs=dcfg["preprocessed_dirs"],
        clip_frames=(
            int(dcfg["clip_duration_min"] * dcfg["fps"]),
            int(dcfg["clip_duration_max"] * dcfg["fps"]),
        ),
        fps=dcfg["fps"],
        val=False,
        val_fraction=dcfg.get("val_fraction", 0.005),
    )

    sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    loader  = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    if _is_main():
        logger.info(f"Train clips: {len(train_ds)}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    tcfg   = cfg["training"]
    opt    = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        betas=tuple(tcfg["betas"]),
        weight_decay=tcfg["weight_decay"],
        eps=tcfg["eps"],
    )

    # ── Mixed precision scaler ─────────────────────────────────────────────────
    mp     = tcfg.get("mixed_precision", "bf16")
    dtype  = torch.bfloat16 if mp == "bf16" else torch.float16 if mp == "fp16" else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(mp == "fp16"))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step  = 0
    resume_path = args.resume or tcfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        if str(resume_path).endswith(".safetensors"):
            # Pre-trained bitHuman weights — load as raw state dict (fine-tune)
            from safetensors.torch import load_file as load_sf
            state = load_sf(str(resume_path), device=device)
            if any(k.startswith("bithuman.") for k in state):
                state = {k.removeprefix("bithuman."): v for k, v in state.items()}
            missing, unexpected = _model.load_state_dict(state, strict=False)
            if _is_main():
                logger.info(f"Fine-tuning from {resume_path}  "
                            f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            # Training checkpoint — restore model + optimizer + step
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
            _model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            if _is_main():
                logger.info(f"Resumed from {resume_path} at step {start_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    out_dir    = Path(tcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    total_steps = tcfg["total_steps"]
    grad_accum  = tcfg.get("grad_accumulation", 1)
    log_every   = tcfg.get("log_every", 100)
    ckpt_every  = tcfg.get("checkpoint_every", 5000)
    grad_clip   = tcfg.get("grad_clip", 1.0)

    step       = start_step
    accum_loss = 0.0
    t_epoch    = time.perf_counter()

    loader_iter = iter(loader)
    opt.zero_grad()

    while step < total_steps:
        # Refresh epoch if needed
        try:
            batch = next(loader_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step)
            loader_iter = iter(loader)
            batch = next(loader_iter)

        if batch is None:
            continue

        # Update learning rate
        lr_now = _cosine_lr(step, tcfg["warmup_steps"], total_steps, tcfg["lr"])
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        # Forward + backward
        with torch.cuda.amp.autocast(enabled=(mp != "no"), dtype=dtype):
            loss = _train_step(model, batch, cfg, device)
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        # Gradient step every grad_accum micro-steps
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        step += 1

        # Logging
        if _is_main() and step % log_every == 0:
            elapsed = time.perf_counter() - t_epoch
            logger.info(
                f"step={step:7d}/{total_steps}  "
                f"loss={accum_loss / log_every:.4f}  "
                f"lr={lr_now:.2e}  "
                f"elapsed={elapsed:.0f}s"
            )
            accum_loss = 0.0
            t_epoch    = time.perf_counter()

        # Checkpoint
        if _is_main() and step % ckpt_every == 0:
            ckpt_path = out_dir / f"step_{step:07d}.pt"
            torch.save({
                "step":      step,
                "model":     _model.state_dict(),
                "optimizer": opt.state_dict(),
                "config":    cfg,
            }, str(ckpt_path))
            logger.info(f"Saved checkpoint → {ckpt_path}")

    if _is_main():
        logger.info("Training complete.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
