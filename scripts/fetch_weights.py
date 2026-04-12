import os
import sys
from pathlib import Path

sys.path.insert(0, '/app')

api_secret = os.environ.get('BITHUMAN_API_SECRET', '').strip()
if not api_secret:
    print("ERROR: BITHUMAN_API_SECRET not set", file=sys.stderr)
    sys.exit(1)

from src.model_downloader import ensure_weights

# Compiled ensure_weights expects (models_dir: Path, api_secret: str)
path = ensure_weights(Path('/data/models'), api_secret)
print(f"Decrypted weights at: {path}")
