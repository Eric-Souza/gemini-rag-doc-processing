# API (`apps/api`)

FastAPI backend for ingest, RAG, and LangGraph agent chat.

## Gemini setup (required)

1. Open https://aistudio.google.com/apikey
2. Create an API key (free tier)
3. Put it in repo-root `.env` or `apps/api/.env`:

   ```
   GEMINI_API_KEY=your_key_here
   ```

4. Restart uvicorn. `GET /health` should show `gemini_configured: true`.

Free-tier limits apply (RPM/RPD). Prefer `gemini-3.6-flash` for chat
and `gemini-embedding-001` for embeddings to stay within quota.

## LangSmith tracing (optional)

1. Create an API key at https://smith.langchain.com
2. Add to `.env`:

   ```
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=doc-processing-ai
   ```

   (`LANGCHAIN_API_KEY` is accepted as an alias for `LANGSMITH_API_KEY`.)

3. Restart uvicorn. `GET /health` should show `langsmith_configured: true`.
4. Ask a question in Chat (RAG or Agent) — open the LangSmith project to inspect traces
   (prompts, retrieved context, tool calls, latency).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

OpenAPI: http://localhost:8000/docs

## Retrieval evals

After uploading the repo `samples/` docs and waiting until status is `ready`:

```bash
python -m evals.run_evals
```

Scores per golden case:

- **hit@k** — expected filename appears in top-k retrieval
- **contains** — answer includes at least one expected phrase
- **faithfulness** — Gemini judge: answer supported by retrieved context only

When LangSmith is configured, each case also logs feedback scores to the project.

## Data on disk

- `data/uploads/` — original files (gitignored, except `.gitkeep`)

Document metadata, chunks, embeddings, and the full-text index all live in Postgres
(`docker compose up -d db` from the repo root).

## Main endpoints

- `GET /health` — liveness + Gemini/LangSmith flags + model names
- `POST /documents` — upload (PDF, TXT, MD)
- `GET /documents` — list
- `GET /documents/{id}` — detail
- `DELETE /documents/{id}` — remove doc (chunks cascade)
- `POST /chat` — SSE stream (`mode`: `rag` | `agent`)

### SSE event payload

Each `data:` line is JSON with an `event` field:

- `token` — `{ "event": "token", "text": "..." }`
- `status` — `{ "event": "status", "message": "Retrieving documents…" }`
- `citation` — `{ "event": "citation", "document_id", "filename", "excerpt", "score" }`
- `done` — `{ "event": "done" }`
- `error` — `{ "event": "error", "message": "..." }`

## Architecture notes

- Ingest runs in `BackgroundTasks` after upload
- Retrieval (`app/services/retrieval/`): pgvector cosine search + Postgres full-text search, fused with RRF, reranked by a local cross-encoder. `RETRIEVAL_MODE` picks `vector`, `hybrid`, or `hybrid_rerank`
- Both searches filter by `document_id` when the client scopes chat to selected docs
- Agent mode uses LangGraph with a `retrieve_documents` tool
- RAG/Agent LLM calls pass LangSmith tags/metadata (`mode:rag|agent`)
