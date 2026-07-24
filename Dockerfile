FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache

# REQUIRED at `docker run` / RunPod template time: -e HF_TOKEN=hf_xxx
# google/gemma-2-9b-it is a GATED checkpoint -- the account behind this
# token must have accepted its license on huggingface.co, or model load
# fails with a 401/gated-repo error. Never bake a real token into the
# image itself; this empty default only documents that the var exists.
ENV HF_TOKEN=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev git curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --upgrade pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install deps first so this layer is cached across code-only changes.
#
# FIX (was: `RUN pip install -r requirements.txt`):
#   1. That plain install pulled in bitsandbytes + nvidia-nvjitlink-cu13
#      unconditionally. Both are only needed for GEMMA_USE_4BIT=1 (off by
#      default) -- if either wheel fails to install on this base image, the
#      ENTIRE docker build failed, even for the bf16-only default path.
#      Mirrors the notebooks' own `grep -v -E "^(bitsandbytes|nvidia-nvjitlink-cu13)"`
#      + best-effort install pattern instead.
#   2. Now explicitly invokes python3.11's pip (via the get-pip.py install
#      above), instead of relying on whatever "pip" resolved to.
COPY requirements.txt .
RUN grep -v -E "^(bitsandbytes|nvidia-nvjitlink-cu13)" requirements.txt > /tmp/requirements_core.txt \
    && python3.11 -m pip install --no-cache-dir -r /tmp/requirements_core.txt \
    && (python3.11 -m pip install --no-cache-dir "bitsandbytes>=0.43.0" nvidia-nvjitlink-cu13 \
        || echo "4-bit extras failed to install -- fine if staying on bf16 (GEMMA_USE_4BIT=0).")

# App code + the pre-built ChromaDB vector store (already in the repo).
# NOTE: retrieval_caps.py is now imported directly by src/app.py, so unlike
# the notebook's server-launch cell, no sitecustomize.py / PYTHONPATH dance
# is needed here -- app.py is this container's own entrypoint.
COPY src/ ./src/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

# Retrieval caps (from 02_Gemma_Inference_and_Serving.ipynb, cell "Runtime
# caps on retrieval breadth") -- kept as env vars, not notebook magic.
ENV RAG_MAX_CANDIDATES=20 \
    RAG_MAX_CONTEXT=8 \
    GEMMA_USE_4BIT=0 \
    GEMMA_MAX_NEW_TOKENS=512 \
    ALLOWED_ORIGINS="*"

EXPOSE 8000

# Cold start on a fresh volume can take 10-20+ min (BGE-M3 ~2.3GB +
# reranker ~2.3GB + Gemma-2-9B bf16 ~18GB from HF_HOME). start-period gives
# that room before a failing check counts against the container.
HEALTHCHECK --interval=30s --timeout=10s --start-period=25m --retries=3 \
    CMD curl -f http://127.0.0.1:8000/status || exit 1

# HF weights (BGE-M3, BGE-reranker, Gemma-2-9B-it) download on first
# container start into HF_HOME. Mount a persistent volume at /models on
# your GPU host so redeploys don't re-download ~20GB every time.
CMD ["python3.11", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]