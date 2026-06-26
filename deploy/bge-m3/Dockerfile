# Reference multi-output BGE-M3 sidecar (dense + learned-sparse + ColBERT).
# CPU image by default; for GPU build FROM an nvidia/cuda + torch base and run
# with the NVIDIA Container Toolkit (BGE-M3 is comfortable on CPU for small
# corpora, but the GPU is worth it for ingest throughput).
FROM python:3.12-slim

WORKDIR /srv/sidecar

# torch (a FlagEmbedding dep) is large; install deps first so the layer caches
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 8090
# single worker: the model is held in-process and is not fork-safe to duplicate
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
