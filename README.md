# Doc Processing AI

Full-stack project for LLM/RAG engineering: upload documents, embed with
Google Gemini, retrieve with hybrid search (pgvector + Postgres full-text, RRF, cross-encoder rerank), answer with LangChain/LangGraph, serve via
FastAPI, and chat from a React TypeScript UI.

## Stack

- Frontend: React + TypeScript + Vite + MUI + Framer Motion (`apps/web`)
- Backend: Python + FastAPI (`apps/api`)
- LLM / embeddings: Google Gemini (AI Studio free API key)
- Orchestration: LangChain + LangGraph
- Observability: LangSmith tracing (optional)
- Database: Postgres 16 + pgvector (SQLAlchemy async + asyncpg), via Docker
- Retrieval: vector + full-text search, Reciprocal Rank Fusion, local cross-encoder reranker (`sentence-transformers`)

## Prerequisites

- Node.js 20+
- Python 3.11+
- Docker (for Postgres + pgvector)
- A free Gemini API key from https://aistudio.google.com/apikey
- Optional: LangSmith API key from https://smith.langchain.com for tracing/evals

## Quick start

1. Copy env and set key:

   ```bash
   cp .env.example .env
   # set GEMINI_API_KEY=...
   # optional: set LANGSMITH_API_KEY=...
   ```

2. Database:

   ```bash
   docker compose up -d db
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

6. (Optional) Run evals. `--seed` indexes `samples/` first; the run compares vector, hybrid, and hybrid + rerank:

   ```bash
   cd apps/api
   .\.venv\Scripts\activate
   python -m evals.run_evals --seed
   ```

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
- LangSmith tracing for LLM/agent runs
- Golden-set retrieval evals (hit@k, answer contains, faithfulness)
- Google Gemini for chat and embeddings
- Postgres + pgvector for metadata, chunks, embeddings, and the full-text index
- Hybrid retrieval with RRF fusion and a local reranker

## In progress: production RAG upgrade

Work on the `production-rag` branch adds hybrid search (Postgres full-text + pgvector), a local cross-encoder reranker, RAGAS evals as a CI quality gate, and Langfuse tracing with per-query cost and latency. See the plan in [docs/production-rag.md](docs/production-rag.md).

## Out of scope

- Auth, LlamaIndex comparison rebuild
