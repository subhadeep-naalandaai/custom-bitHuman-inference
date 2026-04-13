FROM sgubithuman/expression-avatar:latest

# Download pre-decrypted weights from user's own private HuggingFace repo.
# No bitHuman API needed — weights were uploaded once from local bh-weights/.
RUN --mount=type=secret,id=hf_token \
    pip install -q huggingface_hub && \
    huggingface-cli download token-wizard-naalanda/bithuman-weights \
      --local-dir /opt/bh-weights \
      --token $(cat /run/secrets/hf_token)

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
