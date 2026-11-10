# API image: FastAPI + the local cross-encoder reranker on CPU-only PyTorch.
#   docker build -t doc-processing-api:local apps/api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch first, in its own layer: it is most of the image and rarely changes
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install ".[rerank]"

# Bake the reranker into the image, so a new pod doesn't download it before it can serve
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RERANKER_MODEL}')"
# Same for the BM25 term weights Qdrant uses when VECTOR_STORE=qdrant
ENV FASTEMBED_CACHE_PATH=/opt/fastembed
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"

RUN useradd --uid 10001 --no-create-home app \
    && mkdir -p /data/uploads && chown app /data/uploads
USER 10001

ENV HF_HUB_OFFLINE=1 \
    UPLOAD_DIR=/data/uploads

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
