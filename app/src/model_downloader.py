import os
import logging
from pathlib import Path

logger = logging.getLogger("expression-avatar.downloader")

HF_REPO = "token-wizard-naalanda/bithuman-weights"
_DEFAULT_WEIGHTS = "/workspace/bh-weights"


def ensure_weights(models_dir=None, api_secret=None, *args, **kwargs):
    if models_dir is None:
        models_dir = Path(os.environ.get("BITHUMAN_WEIGHTS_PATH", _DEFAULT_WEIGHTS))
    else:
        models_dir = Path(models_dir)

    marker = models_dir / "bithuman-expression" / "Model_Lite" / "config.json"
    if marker.exists():
        logger.info(f"Weights already present at {models_dir}")
        return str(models_dir)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            f"Weights not found at {models_dir} and HF_TOKEN env var is not set.\n"
            "Set HF_TOKEN to your HuggingFace read token in the RunPod pod environment."
        )

    logger.info(f"[downloader] Downloading weights from {HF_REPO} to {models_dir} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO, local_dir=str(models_dir), token=token)
    logger.info("[downloader] Download complete.")
    return str(models_dir)
