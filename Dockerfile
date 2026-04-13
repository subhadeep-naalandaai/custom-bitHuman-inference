FROM sgubithuman/expression-avatar:latest

# Bypass billing heartbeats
COPY app/billing/heartbeat.py /app/billing/heartbeat.py

# Redirect weight downloads to user's own HF repo
COPY app/src/model_downloader.py /app/src/model_downloader.py

RUN pip install --no-cache-dir huggingface_hub
