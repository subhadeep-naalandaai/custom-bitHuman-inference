import os
import logging

logger = logging.getLogger("expression-avatar.downloader")

_DEFAULT_WEIGHTS = "/bh-weights"


def ensure_weights(models_dir=None, api_secret=None, *args, **kwargs):
    weights_path = os.environ.get("BITHUMAN_WEIGHTS_PATH", _DEFAULT_WEIGHTS)
    logger.info(f"Using pre-existing local weights: {weights_path}")
    return weights_path
