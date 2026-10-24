# RAGAS evals

`run_ragas.py` answers golden-set questions with the RAG pipeline and scores each answer with four [RAGAS](https://docs.ragas.io) metrics:

| Metric | Question it answers | Needs |
|---|---|---|
| faithfulness | Is every claim in the answer supported by the retrieved chunks? | answer, chunks |
| answer_relevancy | Does the answer address the question? | answer, embeddings |
| context_precision | Do the chunks that matter for the reference answer rank high? | chunks, reference |
| context_recall | Do the retrieved chunks contain every statement in the reference answer? | chunks, reference |

Questions and reference answers come from `../golden_set.json` (each case has a `reference`), so the retrieval evals and RAGAS share one dataset.

## Running

From `apps/api`, with Postgres running and `pip install -e ".[rerank,evals]"`:

```bash
python -m evals.run_evals --seed --retrieval-only   # index samples/, retrieval metrics
python -m evals.ragas.run_ragas                     # the 7 "ragas_ci" cases, ~85 chat requests
python -m evals.ragas.run_ragas --all               # all 22 cases, ~270 chat requests
python -m evals.gate                                # compare with ../baseline.json
```

`--limit N` scores only the first N cases and `--rpm` caps chat requests per minute (default 10).

## Free-tier budget

The judge is `GEMINI_EVAL_MODEL` (`gemini-3.5-flash-lite`), called through Gemini's OpenAI-compatible endpoint. One case costs 11 to 13 chat requests at k=4: one answer, two for faithfulness (extract claims, then verify them), three for answer relevancy (`strictness=3`), one per chunk for context precision, and one for context recall.

That's why CI scores 7 cases instead of 22. They're the cases marked `"ragas_ci": true` in the golden set, chosen to cover each hard document: exact error codes, near-duplicate policies, similar product codes, and the original samples.

Every chat request, including the ones RAGAS makes internally, goes through an httpx hook that paces it. Per-minute 429s are retried after the delay Gemini suggests, and running out of the daily quota stops the run with a clear message (`../pacing.py`).

## Baseline

`hybrid_rerank`, 7 cases:

| faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|
| 1.000 | 0.898 | 0.905 | 1.000 |

`damage-report` is the low point for context precision (0.33): the reranker ranks its evidence chunk #3, which the retrieval evals also show. The gate allows RAGAS scores to drop by 0.1 before failing, because the judge is an LLM and repeated runs vary a little.

To accept new scores after an intended change, run the evals and then `python -m evals.gate --update-baseline`.
