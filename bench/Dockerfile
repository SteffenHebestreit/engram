# Benchmark runner: engram's app + a local embedding model + cross-encoder.
# Build context is the repo root (see bench/docker-compose.yml).
FROM python:3.12-slim

WORKDIR /srv/engram

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# CPU-only torch keeps the image lean (the default wheel pulls CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY bench/requirements.txt ./bench-requirements.txt
RUN pip install --no-cache-dir -r bench-requirements.txt

COPY app ./app
COPY bench ./bench

CMD ["python", "-m", "bench.run_benchmark"]
