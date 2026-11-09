# Doc Processing AI

Full-stack project for LLM/RAG engineering: upload documents, embed with
Google Gemini, retrieve with hybrid search (pgvector + Postgres full-text, RRF, cross-encoder rerank), answer with LangChain/LangGraph, serve via
FastAPI, and chat from a React TypeScript UI.

## Stack

- Frontend: React + TypeScript + Vite + MUI + Framer Motion (`apps/web`)
- Backend: Python + FastAPI (`apps/api`)
- LLM / embeddings: Google Gemini (AI Studio free API key)
- Orchestration: LangChain + LangGraph
- Observability: Langfuse tracing per retrieval stage and LLM call (optional, self-hostable), LangSmith (optional)
- Database: Postgres 16 + pgvector (SQLAlchemy async + asyncpg), via Docker
- Vector store, either one: pgvector, or [Qdrant](https://qdrant.tech) with dense + BM25 sparse vectors fused on the server (`VECTOR_STORE=qdrant`)
- Retrieval: vector + full-text search, Reciprocal Rank Fusion, local cross-encoder reranker (`sentence-transformers`)
- Evals: golden-set retrieval metrics and RAGAS, enforced by a GitHub Actions quality gate

## Prerequisites

- Node.js 20+
- Python 3.11+
- Docker (for Postgres + pgvector)
- A free Gemini API key from https://aistudio.google.com/apikey
- Optional: Langfuse keys (cloud, or local via `docker-compose.langfuse.yml`) or a LangSmith API key, for tracing

## Quick start

1. Copy env and set key:

   ```bash
   cp .env.example .env
   # set GEMINI_API_KEY=...
   # optional: set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY, or LANGSMITH_API_KEY
   ```

2. Database:

   ```bash
   docker compose up -d postgres qdrant   # qdrant is only needed for VECTOR_STORE=qdrant and the evals
   ```

3. Backend:

   ```bash
   cd apps/api
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   pip install -e ".[rerank]"   # drop [rerank] and set RETRIEVAL_MODE=hybrid to skip torch
   uvicorn app.main:app --reload --port 8000
   ```

4. Frontend (new terminal):

   ```bash
   cd apps/web
   npm install
   npm run dev
   ```

5. Open http://localhost:5173 — upload a PDF/TXT, wait until status is `ready`, chat.

6. (Optional) Run evals. `--seed` indexes `samples/` first; the run compares vector, hybrid, and hybrid + rerank, on pgvector and on Qdrant (rebuilt from the chunks and embeddings in Postgres, so both stores see the same data; `--stores pgvector` skips it). RAGAS and the gate need `pip install -e ".[rerank,evals]"`:

   ```bash
   cd apps/api
   .\.venv\Scripts\activate
   python -m evals.run_evals --seed --retrieval-only   # hit@1, hit@4, MRR per mode and store
   python -m evals.ragas.run_ragas                     # RAGAS on the 7-case CI subset
   python -m evals.gate                                # compare with evals/baseline.json
   ```

7. (Optional) Local Langfuse: `docker compose -f docker-compose.langfuse.yml up -d`, then set the three `LANGFUSE_*` values shown at the top of that file and open http://localhost:3100.

## Project structure

```
doc-processing-ai/
  README.md
  .env.example
  samples/        # sample docs + eval questions
  apps/
    web/          # React + MUI frontend
    api/          # FastAPI backend (+ evals/)
```

Uploaded files live under `apps/api/data/uploads/` (gitignored); everything else is in Postgres.

## Features

- RAG pipeline: load → chunk → embed → retrieve → grounded generate
- LangGraph tool-calling agent over the same index
- Streaming tokens (SSE) to a React client
- Langfuse traces with a span per retrieval stage, plus LLM generations with token usage; LangSmith also supported
- Golden-set evals (chunk-level hit@k and MRR, answer contains, faithfulness) and RAGAS (faithfulness, answer relevancy, context precision, context recall)
- CI quality gate: pytest against real pgvector, then evals compared with a stored baseline
- Google Gemini for chat and embeddings
- Postgres + pgvector for metadata, chunks, embeddings, and the full-text index
- Hybrid retrieval with RRF fusion and a local reranker
- Qdrant as a drop-in second store: each chunk is one point with a dense vector and a BM25 sparse vector, and hybrid search is a single Query API call (two prefetches, fused with RRF on the server). Postgres stays the source of truth, and `python -m app.services.retrieval.qdrant_store` rebuilds Qdrant from it without re-embedding

## Results

Hybrid search plus the reranker raised hit@1 from 73% (vector only) to 91% on the 22-case golden set. On the 7-case CI subset, RAGAS scores faithfulness 1.00, answer relevancy 0.90, context precision 0.91, and context recall 1.00. Details, including where the reranker made things worse, are in [docs/production-rag.md](docs/production-rag.md).

pgvector and Qdrant on the same chunks and embeddings (22 cases, k=4):

| Mode | pgvector hit@1 | pgvector MRR | Qdrant hit@1 | Qdrant MRR |
|---|---|---|---|---|
| vector | 73% | 0.864 | 73% | 0.864 |
| hybrid | 86% | 0.924 | 91% | 0.947 |
| hybrid + rerank | 91% | 0.936 | 91% | 0.936 |

Vector search ranks every case the same in both, as it should with identical embeddings and exact search at this size. The hybrid gap is the keyword half: Qdrant's BM25 put `runbook-err-4021` first, where Postgres `ts_rank_cd` put it second. Once the reranker reorders the candidates, the two stores tie. The gate checks all six rows, so a regression in either store fails CI.

## Out of scope

- Auth, LlamaIndex comparison rebuild
