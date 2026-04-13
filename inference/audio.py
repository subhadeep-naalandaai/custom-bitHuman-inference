"""
Audio encoder: WAV file → fixed-length Wav2Vec2 features.

The DiT expects exactly 30 audio frames (after audio_emb.proj) per video chunk.
We adaptive-pool Wav2Vec2 output to AUDIO_FRAMES=30 regardless of audio length.
"""
import torch
import torch.nn.functional as F

AUDIO_FRAMES = 30   # matches audio_proj.proj1.weight input dim / 1536
SAMPLE_RATE = 16000


class AudioEncoder:
    def __init__(self, model_path: str = "app/bundled/wav2vec2-base-960h", device: str = "cuda"):
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2Model.from_pretrained(model_path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, audio_path: str) -> torch.Tensor:
        """
        Returns [1, AUDIO_FRAMES, 768] Wav2Vec2 features.
        """
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(0)  # → mono [samples]

        inputs = self.feature_extractor(
            waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden = self.model(**inputs).last_hidden_state  # [1, T_audio, 768]

        # Pool temporal dim to fixed AUDIO_FRAMES
        hidden = F.adaptive_avg_pool1d(
            hidden.transpose(1, 2), AUDIO_FRAMES
        ).transpose(1, 2)   # [1, AUDIO_FRAMES, 768]

        return hidden

    @torch.no_grad()
    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [samples] float32 at SAMPLE_RATE Hz.
        Returns [1, AUDIO_FRAMES, 768].
        """
        inputs = self.feature_extractor(
            waveform.cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden = self.model(**inputs).last_hidden_state  # [1, T_audio, 768]
        hidden = F.adaptive_avg_pool1d(
            hidden.transpose(1, 2), AUDIO_FRAMES
        ).transpose(1, 2)
        return hidden
