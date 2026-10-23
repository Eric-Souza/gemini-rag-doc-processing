# Production RAG upgrade

Plan for moving this project from a working RAG demo to a production-style pipeline: better retrieval, measurable quality, and per-query cost and latency.

## Goals

- **Hybrid retrieval:** combine keyword and semantic search so exact terms (IDs, names, policy numbers) and paraphrased questions both find the right chunks.
- **Reranking:** reorder candidates with a cross-encoder before they reach the LLM.
- **Quality gates:** run RAGAS evals in GitHub Actions and fail the build when scores drop below a baseline.
- **Observability:** trace every query in Langfuse with retrieval steps, token usage, cost, and latency.

Everything runs on free tiers or locally.

## Architecture

```
question
  │
  ├─► Postgres full-text search (tsvector, BM25-style ranking) ─┐
  │                                                              ├─► Reciprocal Rank Fusion ─► cross-encoder rerank ─► top-k ─► Gemini answer
  └─► pgvector similarity search (Gemini embeddings) ───────────┘
                                                                                     │
                                                                     Langfuse trace: spans, tokens, cost, latency
```

## Stack changes

| Area | Today | After upgrade |
|------|-------|---------------|
| Vector store | Chroma (local disk) | Postgres + pgvector (Docker locally, Neon free tier hosted) |
| Keyword search | None | Postgres full-text search (`tsvector` + GIN index) |
| Fusion | None | Reciprocal Rank Fusion (RRF) |
| Reranker | None | `BAAI/bge-reranker-base` cross-encoder via `sentence-transformers`, running locally on CPU |
| Metadata DB | SQLite | Postgres (same database as vectors) |
| Evals | Custom golden set (hit@k, contains, faithfulness) | Golden set kept, plus RAGAS: faithfulness, answer relevancy, context precision, context recall |
| CI | None | GitHub Actions: evals on every PR, fail below baseline |
| Tracing | LangSmith (optional) | Langfuse (Hobby free tier), with LangSmith kept as an option |

## Milestones

- [ ] **1. Postgres + pgvector:** `docker-compose.yml`, SQLAlchemy models for documents and chunks, pgvector column, migrate ingestion off Chroma
- [ ] **2. Hybrid search:** full-text index, vector search, RRF fusion in `services/retrieval/`
- [ ] **3. Reranker:** cross-encoder rerank step with configurable candidate count and top-k
- [ ] **4. Retrieval comparison:** eval runs for vector-only vs hybrid vs hybrid + rerank, results table in this doc
- [ ] **5. RAGAS evals:** `evals/ragas/` suite with a stored baseline (`baseline.json`)
- [ ] **6. CI quality gate:** `.github/workflows/evals.yml` runs evals and fails on regression
- [ ] **7. Langfuse tracing:** spans for each retrieval stage and the LLM call, token usage, cost, latency
- [ ] **8. Write-up:** README update, screenshots, Medium article

## Planned structure

```
apps/api/
  app/services/retrieval/
    keyword.py      # Postgres full-text search
    vector.py       # pgvector similarity search
    fusion.py       # Reciprocal Rank Fusion
    rerank.py       # cross-encoder reranker
    pipeline.py     # hybrid retrieve → rerank → top-k
  evals/ragas/
    dataset.json    # questions, reference answers, reference contexts
    run_ragas.py    # computes metrics, compares with baseline
    baseline.json   # scores the CI gate compares against
docker-compose.yml  # Postgres 16 + pgvector
.github/workflows/
  evals.yml         # RAGAS quality gate
```

## Free-tier notes

- **Gemini API (AI Studio):** free tier for chat and embeddings. Also used as the RAGAS judge model in CI, so keep the eval set small (about 20–30 questions) to stay inside rate limits.
- **Neon:** free Postgres with pgvector for a hosted demo.
- **Langfuse Hobby:** 50K units/month, 30-day retention.
- **Reranker:** runs locally, no API cost.
