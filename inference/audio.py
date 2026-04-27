"""
Audio encoder: Wav2Vec2-base-960h wrapper.

Takes raw 16 kHz waveform, returns per-frame feature vectors (768-dim).
Standalone — does not import from app/.
"""
import torch
import torch.nn.functional as F


def load_wav2vec(model_dir: str, device: str = "cuda") -> "Wav2Vec2Model":
    """Load Wav2Vec2-base from a local directory or HuggingFace."""
    from transformers import Wav2Vec2Model as _HF, Wav2Vec2Config

    try:
        model = _HF.from_pretrained(model_dir)
    except Exception:
        model = _HF(Wav2Vec2Config())
    return model.to(device).eval()


def encode_audio(
    wav2vec,
    waveform: torch.Tensor,
    num_frames: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode raw audio waveform to Wav2Vec2 feature vectors.

    Args:
        wav2vec:    loaded Wav2Vec2Model (from load_wav2vec)
        waveform:   (T_audio,) or (B, T_audio) float32 at 16 kHz
        num_frames: target output length (number of video frames)
        device:     inference device

    Returns:
        (B, num_frames, 768) float tensor
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device, dtype=torch.float32)

    with torch.no_grad():
        # Extract CNN features, shape: (B, C, T_feat)
        feats = wav2vec.feature_extractor(waveform)
        # (B, T_feat, C)
        feats = feats.transpose(1, 2)
        # Interpolate to target num_frames
        if feats.shape[1] != num_frames:
            feats = F.interpolate(
                feats.permute(0, 2, 1).float(),
                size=num_frames,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
        # Feature projection: (B, T, hidden_size)
        feats, _ = wav2vec.feature_projection(feats)
        # Transformer encoder
        out = wav2vec.encoder(feats).last_hidden_state

    return out  # (B, num_frames, 768)
