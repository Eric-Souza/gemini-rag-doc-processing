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

## Langfuse tracing (optional)

Langfuse records a trace per chat request, with a span for each retrieval stage (embed, vector,
keyword, RRF, rerank) and a generation for each LLM call, including token usage.

1. Use [Langfuse Cloud](https://cloud.langfuse.com), or run it locally from the repo root:

   ```bash
   docker compose -f docker-compose.langfuse.yml up -d   # http://localhost:3100, dev@example.com / devpassword
   ```

2. Add to `.env` (these are the local stack's preconfigured keys):

   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-local-dev
   LANGFUSE_SECRET_KEY=sk-lf-local-dev
   LANGFUSE_BASE_URL=http://localhost:3100
   ```

3. Restart uvicorn. `GET /health` should show `langfuse_configured: true`.

Langfuse's price table doesn't include Gemini 3.x models, so add a model definition
(Settings → Models) if you want costs next to token counts. LangSmith and Langfuse can be on
at the same time.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

OpenAPI: http://localhost:8000/docs

## Evals

`--seed` indexes the repo `samples/` docs first (or upload them and wait until status is `ready`):

```bash
python -m evals.run_evals --seed                 # retrieval + answer evals, every retrieval mode
python -m evals.run_evals --retrieval-only       # embeddings only, no chat calls
pip install -e ".[rerank,evals]"
python -m evals.ragas.run_ragas                  # RAGAS on the 7-case CI subset (--all for 22)
python -m evals.gate                             # fail if a metric dropped below evals/baseline.json
```

Scores per golden case:

- **hit@k / MRR** — rank of the first retrieved chunk that contains the case's `evidence` string
- **contains** — answer includes at least one expected phrase
- **faithfulness** — Gemini judge: answer supported by retrieved context only
- **RAGAS** — faithfulness, answer relevancy, context precision, context recall (see `evals/ragas/README.md`)

When LangSmith is configured, each case also logs feedback scores to the project. The same
evals run in GitHub Actions (`.github/workflows/evals.yml`) when the repo has a
`GEMINI_API_KEY` secret.

## Data on disk

- `data/uploads/` — original files (gitignored, except `.gitkeep`)

Document metadata, chunks, embeddings, and the full-text index all live in Postgres
(`docker compose up -d postgres` from the repo root).

## Main endpoints

- `GET /health` — liveness + Gemini/LangSmith/Langfuse flags + model names
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
- RAG/Agent LLM calls pass LangSmith tags/metadata (`mode:rag|agent`); Langfuse traces get the same tags
