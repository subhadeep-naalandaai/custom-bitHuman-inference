import os
import sys

sys.path.insert(0, '/app')

api_secret = os.environ.get('BITHUMAN_API_SECRET', '').strip()
if not api_secret:
    print("ERROR: BITHUMAN_API_SECRET not set", file=sys.stderr)
    sys.exit(1)

from src.model_downloader import ensure_weights

path = ensure_weights(api_secret=api_secret)
print(f"Decrypted weights at: {path}")
