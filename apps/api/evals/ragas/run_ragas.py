"""RAGAS evals: answer each golden-set case with the RAG pipeline, then score it.

  faithfulness       every claim in the answer is supported by the retrieved chunks
  answer_relevancy   the answer addresses the question (questions generated back from the
                     answer are compared with the original by embedding similarity)
  context_precision  the retrieved chunks that matter for the reference answer rank high
  context_recall     the retrieved chunks contain every statement in the reference answer

The judge is GEMINI_EVAL_MODEL, called through Gemini's OpenAI-compatible endpoint (the
route ragas recommends for Gemini). One case costs about 11 chat requests at k=4, so the
default run uses the cases marked "ragas_ci" in golden_set.json; --all runs every case.

Usage (from apps/api with the venv active and Postgres running):

    python -m evals.ragas.run_ragas --seed     # 7 CI cases, about 80 chat requests
    python -m evals.ragas.run_ragas --all      # all 22 cases, about 240 chat requests
    python -m evals.gate                       # compare with evals/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

API_ROOT = Path(__file__).resolve().parent.parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.config import apply_langsmith_env, get_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.services.rag import answer_rag  # noqa: E402
from app.services.retrieval import RetrievalMode  # noqa: E402
from app.services.tracing import flush as flush_traces  # noqa: E402
from evals.pacing import Pacer  # noqa: E402
from evals.run_evals import GOLDEN_PATH, RESULTS_DIR, _ensure_index_ready, seed_samples  # noqa: E402

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
RESULTS_PATH = RESULTS_DIR / "ragas_latest.json"


def build_metrics(pacer: Pacer) -> dict[str, Any]:
    # ragas warns about its own deprecations on import; they aren't actionable here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

    settings = get_settings()

    async def pace(request: httpx.Request) -> None:
        # Metrics make several chat calls each; pace every one, not just every metric
        if request.url.path.endswith("/chat/completions"):
            await pacer.wait()

    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url=GEMINI_OPENAI_URL,
        max_retries=0,  # the Pacer handles 429s using the delay Gemini suggests
        http_client=httpx.AsyncClient(event_hooks={"request": [pace]}, timeout=120),
    )
    llm = llm_factory(settings.gemini_eval_model, provider="openai", client=client, temperature=0)
    embeddings = OpenAIEmbeddings(client=client, model=settings.gemini_embed_model)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_case(case: dict[str, Any], mode: RetrievalMode, metrics: dict[str, Any], pacer: Pacer) -> dict[str, Any]:
    result = await pacer.call(
        answer_rag,
        case["question"],
        k=int(case.get("k", 4)),
        case_id=case["id"],
        mode=mode,
        model=get_settings().gemini_eval_model,
    )
    answer = result["answer"]
    contexts = [hit.content for hit in result["hits"]]
    inputs = {
        "faithfulness": {"user_input": case["question"], "response": answer, "retrieved_contexts": contexts},
        "answer_relevancy": {"user_input": case["question"], "response": answer},
        "context_precision": {"user_input": case["question"], "reference": case["reference"], "retrieved_contexts": contexts},
        "context_recall": {"user_input": case["question"], "retrieved_contexts": contexts, "reference": case["reference"]},
    }
    scores: dict[str, float] = {}
    for name, metric in metrics.items():
        scored = await pacer.call(metric.ascore, paced=False, **inputs[name])
        scores[name] = float(scored.value)
    return {"id": case["id"], "answer": answer, **scores}


def _print_table(rows: list[dict[str, Any]], means: dict[str, float]) -> None:
    labels = {"faithfulness": "faithful", "answer_relevancy": "relevancy", "context_precision": "ctx_prec", "context_recall": "ctx_recall"}
    print(f"{'case':<30}" + "".join(f"{labels[m]:>12}" for m in METRICS))
    print("-" * (30 + 12 * len(METRICS)))
    for row in rows:
        print(f"{row['id']:<30}" + "".join(f"{row[m]:>12.2f}" for m in METRICS))
    print("-" * (30 + 12 * len(METRICS)))
    print(f"{'MEAN':<30}" + "".join(f"{means[m]:>12.3f}" for m in METRICS))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", action="store_true", help="index samples/ before running")
    parser.add_argument("--all", action="store_true", help="score every case, not just the ragas_ci subset")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N selected cases")
    parser.add_argument("--mode", default=None, help="retrieval mode (default: RETRIEVAL_MODE setting)")
    parser.add_argument(
        "--rpm",
        type=float,
        default=10,
        help="max chat requests per minute across answers and judge calls (0 disables pacing)",
    )
    args = parser.parse_args()

    apply_langsmith_env()
    settings = get_settings()
    if not settings.gemini_configured:
        raise SystemExit("GEMINI_API_KEY is not set. Add it to .env first.")

    await init_db()
    if args.seed:
        print("Seeding samples/ ...")
        await seed_samples()

    cases: list[dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    selected = cases if args.all else [c for c in cases if c.get("ragas_ci")]
    if args.limit > 0:
        selected = selected[: args.limit]
    await _ensure_index_ready({name for case in selected for name in case["expected_filenames"]})

    mode = RetrievalMode(args.mode or settings.retrieval_mode)
    pacer = Pacer(args.rpm)
    metrics = build_metrics(pacer)
    print(
        f"\nRAGAS: {len(selected)} cases, mode={mode.value}, judge={settings.gemini_eval_model}, "
        f"rpm={args.rpm:g}\n"
    )

    rows = []
    for case in selected:
        print(f"  -> {case['id']}")
        rows.append(await score_case(case, mode, metrics, pacer))

    n = len(rows) or 1
    means = {m: sum(r[m] for r in rows) / n for m in METRICS}
    print()
    _print_table(rows, means)
    print(f"\n{pacer.requests} chat requests, {pacer.rate_limited} rate-limited retries")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subset": "all" if args.all else "ragas_ci",
        "cases_scored": len(rows),
        "mode": mode.value,
        "eval_model": settings.gemini_eval_model,
        "embed_model": settings.gemini_embed_model,
        "metrics": means,
        "cases": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {RESULTS_PATH.relative_to(API_ROOT)}")
    flush_traces()


if __name__ == "__main__":
    asyncio.run(main())
