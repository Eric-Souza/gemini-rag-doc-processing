# Doc Processing AI

Full-stack learning project for LLM/RAG engineering: upload documents, embed with
Google Gemini, retrieve with Chroma, answer with LangChain/LangGraph, serve via
FastAPI, and chat from a React TypeScript UI.

## Stack

- Frontend: React + TypeScript + Vite + MUI + Framer Motion (`apps/web`)
- Backend: Python + FastAPI (`apps/api`)
- LLM / embeddings: Google Gemini (AI Studio free API key)
- Orchestration: LangChain + LangGraph
- Metadata DB: SQLite (SQLAlchemy async)
- Vectors: Chroma (local disk)

## Prerequisites

- Node.js 20+
- Python 3.11+
- A free Gemini API key from https://aistudio.google.com/apikey

## Quick start

1. Copy env and set key:

   ```bash
   cp .env.example .env
   # set GEMINI_API_KEY=...
   ```

2. Backend:

   ```bash
   cd apps/api
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   pip install -e .
   uvicorn app.main:app --reload --port 8000
   ```

3. Frontend (new terminal):

   ```bash
   cd apps/web
   npm install
   npm run dev
   ```

4. Open http://localhost:5173 — upload a PDF/TXT, wait until status is `ready`, chat.

## Project structure

```
doc-processing-ai/
  README.md
  .env.example
  apps/
    web/          # React + MUI frontend
    api/          # FastAPI backend
```

Runtime data (`app.db`, uploads, Chroma) lives under `apps/api/data/` and is gitignored.

## Learning goals (resume talking points)

- RAG pipeline: load → chunk → embed → retrieve → grounded generate
- LangGraph tool-calling agent over the same index
- Streaming tokens (SSE) to a React client
- Provider-locked to Gemini for free-tier development
- SQLite for document metadata; Chroma for vectors

## Next steps (not in v1)

- LangSmith tracing, retrieval evals, auth, pgvector migration, LlamaIndex comparison rebuild
