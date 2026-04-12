FROM sgubithuman/expression-avatar:latest

# Use the original model_downloader to fetch + decrypt weights via BITHUMAN_API_SECRET.
# The compiled module calls api.bithuman.ai to exchange the secret for an HF token,
# downloads from the private HF repo, encrypts to /data/models, decrypts to /tmp/bh-weights.
# We then copy the decrypted weights to /opt/bh-weights and clean up the encrypted blobs.
RUN --mount=type=secret,id=bithuman_secret \
    cd /app && \
    python3 -c "
import os, sys
sys.path.insert(0, '/app')
os.environ['BITHUMAN_API_SECRET'] = open('/run/secrets/bithuman_secret').read().strip()
from src.model_downloader import ensure_weights
path = ensure_weights(api_secret=os.environ['BITHUMAN_API_SECRET'])
print('Decrypted weights at:', path)
" && \
    cp -r /tmp/bh-weights /opt/bh-weights && \
    rm -rf /data/models

# Set weights path so our model_downloader stub finds them
ENV BITHUMAN_WEIGHTS_PATH=/opt/bh-weights

# Override compiled billing heartbeat with a no-op stub
COPY app/billing/heartbeat.py /app/billing/heartbeat.py

# Override compiled model downloader with a local-weights stub
COPY app/src/model_downloader.py /app/src/model_downloader.py
