"""
bithuman_expression/audio_analysis/wav2vec2.py
Python replacement for wav2vec2.cpython-310-x86_64-linux-gnu.so

Exposes:
  Wav2Vec2Model(config: Wav2Vec2Config)
      .feature_extract(input_values, seq_len) -> torch.Tensor
      .encode(extract_features, attention_mask, ...) -> BaseModelOutput
      .forward(input_values, seq_len, attention_mask, ...) -> BaseModelOutput

Also re-exports:
  Wav2Vec2Config  (from transformers)
  BaseModelOutput (from transformers)
"""

from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model as _HFWav2Vec2Model,
)

from inference.audio_analysis.torch_utils import linear_interpolation


class Wav2Vec2Model(_HFWav2Vec2Model):
    """
    bitHuman extension of HuggingFace Wav2Vec2Model.

    Splits the standard forward pass into two callable stages:
      1. feature_extract() — CNN feature extractor + linear interpolation to seq_len
      2. encode()          — Transformer encoder over extracted features

    This allows the caller to cache or manipulate features between the two stages.
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

    def feature_extract(
        self,
        input_values: "torch.Tensor",
        seq_len: int,
    ) -> "torch.Tensor":
        """
        Run the CNN feature extractor on raw audio and interpolate to seq_len.

        Args:
            input_values: (B, T_audio) float tensor — raw waveform samples
            seq_len:      target sequence length (number of video frames)

        Returns:
            (B, seq_len, hidden_size) float tensor — projected feature vectors
        """
        # 1. CNN feature extraction → (B, C, T_feat)
        extract_features = self.feature_extractor(input_values)
        # 2. (B, C, T_feat) → (B, T_feat, C)
        extract_features = extract_features.transpose(1, 2)
        # 3. Interpolate to target sequence length
        extract_features = linear_interpolation(extract_features, seq_len)
        # 4. Feature projection (linear + layer norm) → (B, seq_len, hidden_size)
        hidden_states, _ = self.feature_projection(extract_features)
        return hidden_states

    def encode(
        self,
        extract_features: "torch.Tensor",
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> BaseModelOutput:
        """
        Run the Transformer encoder over pre-extracted feature vectors.

        Args:
            extract_features: (B, T, hidden_size) — output of feature_extract()
            attention_mask:   optional (B, T) bool tensor
            mask_time_indices: optional masking indices for spec-augment
            output_attentions: bool
            output_hidden_states: bool
            return_dict: bool

        Returns:
            BaseModelOutput with .last_hidden_state (B, T, hidden_size)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = extract_features

        # Apply spec-augment masking during training if requested
        if self.training and mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )

    def forward(
        self,
        input_values: "torch.Tensor",
        seq_len: int,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> BaseModelOutput:
        """
        Full forward pass: feature_extract() followed by encode().

        Args:
            input_values: (B, T_audio) raw waveform
            seq_len:      target number of output frames
            attention_mask, mask_time_indices, output_attentions,
            output_hidden_states, return_dict: standard HF arguments

        Returns:
            BaseModelOutput with .last_hidden_state (B, seq_len, hidden_size)
        """
        extract_features = self.feature_extract(input_values, seq_len)
        return self.encode(
            extract_features,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
