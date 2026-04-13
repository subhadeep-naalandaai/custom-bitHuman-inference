FROM sgubithuman/expression-avatar:latest

# Use the original model_downloader to fetch + decrypt weights via BITHUMAN_API_SECRET.
# The compiled module calls api.bithuman.ai to exchange the secret for an HF token,
# downloads from the private HF repo, encrypts to /data/models, decrypts to /tmp/bh-weights.
# We then copy the decrypted weights to /opt/bh-weights and clean up the encrypted blobs.
COPY scripts/fetch_weights.py /tmp/fetch_weights.py
RUN --mount=type=secret,id=bithuman_secret \
    cd /app && \
    BITHUMAN_API_SECRET=$(cat /run/secrets/bithuman_secret) \
    python3 /tmp/fetch_weights.py && \
    cp -r /tmp/bh-weights /opt/bh-weights && \
    rm -rf /data/models /tmp/fetch_weights.py

# Set weights path so our model_downloader stub finds them
ENV BITHUMAN_WEIGHTS_PATH=/opt/bh-weights

# Override compiled billing heartbeat with a no-op stub
COPY app/billing/heartbeat.py /app/billing/heartbeat.py

# Override compiled model downloader with a local-weights stub
COPY app/src/model_downloader.py /app/src/model_downloader.py

# Add standalone inference CLI (no compiled deps, no billing, no network)
COPY inference/ /app/inference/
COPY run.py /app/run.py
RUN pip install --no-cache-dir torchaudio "imageio[ffmpeg]"
