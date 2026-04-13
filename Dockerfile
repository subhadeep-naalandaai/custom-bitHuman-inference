FROM sgubithuman/expression-avatar:latest

# Delete compiled .so modules so our .py stubs take precedence
RUN rm -f /app/src/model_downloader.cpython-310-x86_64-linux-gnu.so \
          /app/billing/heartbeat.cpython-310-x86_64-linux-gnu.so

# Bypass billing heartbeats
COPY app/billing/heartbeat.py /app/billing/heartbeat.py

# Redirect weight downloads to user's own HF repo
COPY app/src/model_downloader.py /app/src/model_downloader.py

RUN pip install --no-cache-dir huggingface_hub
