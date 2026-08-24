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

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

OpenAPI: http://localhost:8000/docs

## Data on disk

- `data/app.db` — SQLite document metadata
- `data/uploads/` — original files
- `data/chroma/` — vector store

All under `data/` are gitignored (except `.gitkeep`).

## Main endpoints

- `GET /health` — liveness + whether Gemini key is configured
- `POST /documents` — upload (PDF, TXT, MD)
- `GET /documents` — list
- `GET /documents/{id}` — detail
- `DELETE /documents/{id}` — remove doc + vectors
- `POST /chat` — SSE stream (`mode`: `rag` | `agent`)

### SSE event payload

Each `data:` line is JSON with an `event` field:

- `token` — `{ "event": "token", "text": "..." }`
- `citation` — `{ "event": "citation", "document_id", "filename", "excerpt", "score" }`
- `done` — `{ "event": "done" }`
- `error` — `{ "event": "error", "message": "..." }`

## Architecture notes

- Ingest runs in `BackgroundTasks` after upload
- Chroma filters by `document_id` metadata when the client scopes chat to selected docs
- Agent mode uses LangGraph with a `retrieve_documents` tool
