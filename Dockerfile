FROM sgubithuman/expression-avatar:latest

# Override compiled billing heartbeat with a no-op stub
COPY app/billing/heartbeat.py /app/billing/heartbeat.py

# Override compiled model downloader with a local-weights stub
COPY app/src/model_downloader.py /app/src/model_downloader.py
