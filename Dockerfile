FROM sgubithuman/expression-avatar:latest

# Download weights from HuggingFace into the image (requires HF_TOKEN build secret)
RUN --mount=type=secret,id=hf_token \
    pip install -q huggingface_hub && \
    huggingface-cli download stevegu1984/expression-avatar-weights \
      --local-dir /opt/bh-weights \
      --token $(cat /run/secrets/hf_token)

# Set weights path so our model_downloader stub finds them
ENV BITHUMAN_WEIGHTS_PATH=/opt/bh-weights

# Override compiled billing heartbeat with a no-op stub
COPY app/billing/heartbeat.py /app/billing/heartbeat.py

# Override compiled model downloader with a local-weights stub
COPY app/src/model_downloader.py /app/src/model_downloader.py
