FROM sgubithuman/expression-avatar:latest

# Delete compiled .so modules so our .py stubs take precedence
RUN rm -f /app/src/model_downloader.cpython-310-x86_64-linux-gnu.so \
          /app/billing/heartbeat.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/utils/utils.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/utils/facecrop.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/utils/cpu_face_handler.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/audio_analysis/torch_utils.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/audio_analysis/wav2vec2.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/src/distributed/usp_device.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/src/modules/expression_model.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/src/pipeline/expression_pipeline.cpython-310-x86_64-linux-gnu.so \
          /app/bithuman_expression/inference.cpython-310-x86_64-linux-gnu.so \
          /app/src/engine_builder.cpython-310-x86_64-linux-gnu.so \
          /app/src/session_pool.cpython-310-x86_64-linux-gnu.so \
          /app/src/streaming_engine.cpython-310-x86_64-linux-gnu.so \
          /app/src/video_generator.cpython-310-x86_64-linux-gnu.so \
          /app/src/trt_pipeline.cpython-310-x86_64-linux-gnu.so \
          /app/src/trt_runner.cpython-310-x86_64-linux-gnu.so \
          /app/src/idle_audio.cpython-310-x86_64-linux-gnu.so

# Bypass billing heartbeats
COPY app/billing/heartbeat.py /app/billing/heartbeat.py

# Redirect weight downloads to user's own HF repo
COPY app/src/model_downloader.py /app/src/model_downloader.py

# Readable Python utils (replaces Cython .so)
COPY app/bithuman_expression/utils/utils.py /app/bithuman_expression/utils/utils.py
COPY app/bithuman_expression/utils/facecrop.py /app/bithuman_expression/utils/facecrop.py
COPY app/bithuman_expression/utils/cpu_face_handler.py /app/bithuman_expression/utils/cpu_face_handler.py

# Readable Python audio analysis (replaces Cython .so)
COPY app/bithuman_expression/audio_analysis/torch_utils.py /app/bithuman_expression/audio_analysis/torch_utils.py
COPY app/bithuman_expression/audio_analysis/wav2vec2.py /app/bithuman_expression/audio_analysis/wav2vec2.py

# Readable Python src modules (replaces Cython .so)
COPY app/bithuman_expression/src/distributed/usp_device.py /app/bithuman_expression/src/distributed/usp_device.py
COPY app/bithuman_expression/src/modules/expression_model.py /app/bithuman_expression/src/modules/expression_model.py
COPY app/bithuman_expression/src/pipeline/expression_pipeline.py /app/bithuman_expression/src/pipeline/expression_pipeline.py

# Top-level inference API (replaces Cython .so)
COPY app/bithuman_expression/inference.py /app/bithuman_expression/inference.py

# App src modules (replace Cython .so)
COPY app/src/idle_audio.py /app/src/idle_audio.py
COPY app/src/trt_runner.py /app/src/trt_runner.py
COPY app/src/engine_builder.py /app/src/engine_builder.py
COPY app/src/trt_pipeline.py /app/src/trt_pipeline.py
COPY app/src/session_pool.py /app/src/session_pool.py
COPY app/src/streaming_engine.py /app/src/streaming_engine.py
COPY app/src/video_generator.py /app/src/video_generator.py

RUN pip install --no-cache-dir huggingface_hub safetensors
