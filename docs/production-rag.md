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
- [x] **5. RAGAS evals:** `evals/ragas/` suite with a stored baseline (`evals/baseline.json`)
- [x] **6. CI quality gate:** `.github/workflows/evals.yml` runs evals and fails on regression
- [x] **7. Langfuse tracing:** spans for each retrieval stage and the LLM call, token usage, cost, latency
- [x] **8. Write-up:** README update, screenshots, Medium article (draft)

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

## RAGAS results

`python -m evals.ragas.run_ragas` scores the 7 golden-set cases marked `ragas_ci` (hybrid + rerank, judge `gemini-3.5-flash-lite`). Reference answers live in `golden_set.json` next to each question.

| Case | faithfulness | answer relevancy | context precision | context recall |
|------|---|---|---|---|
| benefits-deductible | 1.00 | 0.95 | 1.00 | 1.00 |
| nova-handle-time | 1.00 | 0.89 | 1.00 | 1.00 |
| damage-report | 1.00 | 0.99 | 0.33 | 1.00 |
| runbook-err-4021 | 1.00 | 0.78 | 1.00 | 1.00 |
| travel-2025-hotel-cap | 1.00 | 0.97 | 1.00 | 1.00 |
| catalog-sp30-room | 1.00 | 0.85 | 1.00 | 1.00 |
| security-lost-device | 1.00 | 0.85 | 1.00 | 1.00 |
| **Mean** | **1.000** | **0.898** | **0.905** | **1.000** |

The run took 85 chat requests and no rate-limit retries. `damage-report`'s context precision of 0.33 is the same weakness the retrieval evals found: the reranker puts its evidence chunk at #3. Answer relevancy is the noisiest metric, because it compares questions an LLM generates back from the answer.

## Quality gate

`.github/workflows/evals.yml` has two jobs:

- **tests:** pytest against a `pgvector/pgvector:pg16` service container, with CPU-only PyTorch for the reranker tests.
- **eval-gate:** seeds `samples/`, runs the retrieval evals and the RAGAS subset, then `python -m evals.gate`. It needs a `GEMINI_API_KEY` repository secret and reports a skip without one (for example on forks).

`evals/gate.py` compares the results with `evals/baseline.json` and exits 1 when a metric falls below its baseline by more than the tolerance:

- **Retrieval (hit@1, MRR for every mode), tolerance 0.01:** retrieval is deterministic, so a single case slipping from #1 to #2 (1/22 of hit@1) fails the build.
- **RAGAS (four means), tolerance 0.1:** the judge is an LLM, so small run-to-run changes don't count as regressions.

Every retrieval mode is gated, not just `hybrid_rerank`, because of a test I ran: reverting keyword search to `websearch_to_tsquery` (the bug from milestone 2) left `hybrid_rerank` at 91% hit@1. The reranker gets 20 candidates and the corpus has 17 chunks, so it reads everything and repairs the ranking on its own. Only `hybrid` showed the damage (hit@1 0.864 → 0.818), and gating it is what fails the build.

The comparison table is written to the job summary. After an intended change, `python -m evals.gate --update-baseline` accepts the new scores.

## Tracing

With `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` set, every chat request is one Langfuse trace (`app/services/tracing.py`). Without them, the tracing helpers do nothing.

```
doc-qa-rag              span        root, tagged mode:rag (eval runs add eval:<case>)
├─ retrieve             retriever   question, k, mode → final chunks with every stage's scores
│  ├─ embed_query       embedding   gemini-embedding-001
│  ├─ vector_search     span        top 20 by cosine
│  ├─ keyword_search    span        top 20 by ts_rank_cd
│  ├─ rrf               span        fused ranking
│  └─ rerank            span        cross-encoder top-k
└─ doc-qa-rag           generation  model, prompt, answer, token usage (LangChain callback)
```

Agent mode traces under `doc-qa-agent`: the LangChain callback records the LangGraph nodes, each LLM call, and each tool call, and every tool call adds a `retrieve` subtree under the trace root. Langfuse's built-in price table doesn't include Gemini 3.x models, so cost stays empty until you add a model definition with your price in Langfuse (Settings → Models). On the Gemini free tier, the real cost is $0.

`docker-compose.langfuse.yml` runs Langfuse locally with a preconfigured project and API keys (see the comment at the top of the file). LangSmith tracing still works as before, and both can be on at once.

## Structure

```
apps/api/
  app/services/retrieval/
    keyword.py        # Postgres full-text search
    vector.py         # pgvector similarity search
    fusion.py         # Reciprocal Rank Fusion
    rerank.py         # cross-encoder reranker
    pipeline.py       # hybrid retrieve → rerank → top-k, one span per stage
  app/services/tracing.py   # optional Langfuse client, spans, LangChain callback
  evals/
    golden_set.json   # questions, evidence strings, reference answers, ragas_ci flags
    run_evals.py      # retrieval (hit@k, MRR) and answer evals
    ragas/run_ragas.py  # RAGAS metrics on the CI subset or all cases
    pacing.py         # free-tier request pacing and 429 handling
    gate.py           # compares results with baseline.json
    baseline.json     # scores the CI gate compares against
docker-compose.yml           # Postgres 16 + pgvector
docker-compose.langfuse.yml  # optional local Langfuse
.github/workflows/
  evals.yml          # tests + eval quality gate
```

The original plan had a separate `evals/ragas/dataset.json`. Reference answers went into `golden_set.json` instead, so both eval suites share one dataset, and the baseline sits next to the gate in `evals/`.

## Free-tier notes

- **Gemini API (AI Studio):** free tier for chat and embeddings. Also the RAGAS judge in CI, through a lite model with a larger daily quota. One RAGAS case costs 11 to 13 chat requests, so CI scores a 7-case subset (about 85 requests).
- **Neon:** free Postgres with pgvector for a hosted demo.
- **Langfuse Hobby:** 50K units/month, 30-day retention.
- **Reranker:** runs locally, no API cost.
