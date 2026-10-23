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
| Reranker | None | `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder (configurable via `RERANKER_MODEL`) via `sentence-transformers`, running locally on CPU |
| Metadata DB | SQLite | Postgres (same database as vectors) |
| Evals | Custom golden set (hit@k, contains, faithfulness) | Golden set kept, plus RAGAS: faithfulness, answer relevancy, context precision, context recall |
| CI | None | GitHub Actions: evals on every PR, fail below baseline |
| Tracing | LangSmith (optional) | Langfuse (Hobby free tier), with LangSmith kept as an option |

## Milestones

- [x] **1. Postgres + pgvector:** `docker-compose.yml`, SQLAlchemy models for documents and chunks, pgvector column, migrate ingestion off Chroma
- [x] **2. Hybrid search:** full-text index, vector search, RRF fusion in `services/retrieval/`
- [x] **3. Reranker:** cross-encoder rerank step with configurable candidate count and top-k
- [x] **4. Retrieval comparison:** eval runs for vector-only vs hybrid vs hybrid + rerank, results table in this doc
- [ ] **5. RAGAS evals:** `evals/ragas/` suite with a stored baseline (`baseline.json`)
- [ ] **6. CI quality gate:** `.github/workflows/evals.yml` runs evals and fails on regression
- [ ] **7. Langfuse tracing:** spans for each retrieval stage and the LLM call, token usage, cost, latency
- [ ] **8. Write-up:** README update, screenshots, Medium article

## Retrieval results

`python -m evals.run_evals --seed --retrieval-only` over the 22-case golden set (8 sample docs, 17 chunks, k=4). A case counts as a hit when a retrieved chunk contains the case's `evidence` string, so this measures chunk-level ranking, not just the right file.

| Mode | hit@1 | hit@4 | MRR |
|------|-------|-------|-----|
| Vector only (pgvector cosine) | 73% | 100% | 0.864 |
| Hybrid (vector + full-text, RRF) | 86% | 100% | 0.924 |
| Hybrid + rerank (MiniLM cross-encoder) | **91%** | 100% | **0.936** |

Where it helped:

- **Exact codes:** `runbook-err-4021` was #2 for vector search (ERR-4012 looks almost the same as an embedding). Full-text search matches the exact token, and the reranker puts it at #1.
- **Near-duplicate documents:** in `travel-2025-hotel-cap` and `travel-2026-expense-deadline`, the 2025 and 2026 policies differ only in their numbers. Hybrid search and the reranker pick the right year.
- **Specific nouns:** `catalog-sp30-room` and `security-lost-device` move from #2 to #1 once keywords count.

Where it hurt:

- `damage-report` dropped from #1 to #3 and `travel-2026-per-diem` from #2 to #4 under the reranker. MiniLM is a small, general-purpose model trained on web search queries. A larger reranker (for example `BAAI/bge-reranker-base`, set through `RERANKER_MODEL`) is the next thing to try.

Caveat: with only 17 chunks, every mode finds the evidence somewhere in the top 4, so hit@4 can't tell the modes apart. hit@1 and MRR are the numbers to watch until the corpus grows.

### Answers

`python -m evals.run_evals --mode hybrid_rerank` scores every answer with an expected-phrase check and a faithfulness judge (the judge checks the answer only uses the retrieved context). With hybrid + rerank retrieval: **contains 100%, faithful 100%** (22/22).

The answer evals run on `GEMINI_EVAL_MODEL` (`gemini-3.5-flash-lite`), not the app's chat model. The Gemini free tier allows only 20 requests per day for `gemini-3.6-flash`, and one run needs 44 (answer + judge per case). The runner keeps requests under the per-minute limit (`--rpm`, default 5), waits and retries on per-minute 429s, and stops with a clear message once the daily quota is used up.

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
