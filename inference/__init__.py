"""
Naalanda inference package — audio-conditioned talking-head video generation.

Public API:
    get_pipeline(ckpt_dir, models_dir, wav2vec_dir)
    get_infer_params(pipeline, ...)
    get_audio_embedding(pipeline, audio, sample_rate, num_frames)
    get_base_data(pipeline, image_path)
    run_pipeline(pipeline, audio_emb, base_data, params)
"""
from inference.pipeline import (
    NaalandaPipeline,
    get_pipeline,
    get_infer_params,
    get_audio_embedding,
    get_base_data,
    run_pipeline,
)
from inference.model import ExpressionModel

__all__ = [
    "NaalandaPipeline",
    "ExpressionModel",
    "get_pipeline",
    "get_infer_params",
    "get_audio_embedding",
    "get_base_data",
    "run_pipeline",
]
