# Reference reranker sidecar (Qwen3-Reranker via sentence-transformers CrossEncoder).
# CPU image by default; for GPU build FROM an nvidia/cuda + torch base and run
# with the NVIDIA Container Toolkit (the 0.6B reranker is fine on CPU for small
# shortlists; the 4B/8B want a GPU).
FROM python:3.12-slim

WORKDIR /srv/reranker

# torch (a sentence-transformers dep) is large; install deps first so it caches
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 8091
# single worker: the model is held in-process and not fork-safe to duplicate
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
