"""TruLens RAG triad: answer each golden-set case with the RAG pipeline, then score the three links.

  context_relevance  each retrieved chunk is relevant to the question (mean over the chunks)
  groundedness       each sentence of the answer is supported by the retrieved chunks
  answer_relevance   the answer addresses the question

Unlike RAGAS, none of the three needs a reference answer, so the triad can score live traffic
too. The judge is GEMINI_EVAL_MODEL through TruLens's OpenAI provider, pointed at Gemini's
OpenAI-compatible endpoint. One case costs about 8 chat requests at k=4 (1 answer, 4 context
relevance, 2 or more for groundedness, 1 answer relevance), so the default run uses the "ragas_ci" cases.

Usage (from apps/api with the venv active and Postgres running):

    python -m evals.trulens.run_trulens --seed     # 7 CI cases, about 60 chat requests
    python -m evals.trulens.run_trulens --all      # all 22 cases, about 180 chat requests
    python -m evals.gate                           # compare with evals/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import openai

API_ROOT = Path(__file__).resolve().parent.parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.config import apply_langsmith_env, get_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.services.rag import answer_rag  # noqa: E402
from app.services.retrieval import RetrievalMode  # noqa: E402
from app.services.tracing import flush as flush_traces  # noqa: E402
from evals.pacing import Pacer  # noqa: E402
from evals.ragas.run_ragas import GEMINI_OPENAI_URL  # noqa: E402
from evals.run_evals import GOLDEN_PATH, RESULTS_DIR, _ensure_index_ready, seed_samples  # noqa: E402

METRICS = ("context_relevance", "groundedness", "answer_relevance")
RESULTS_PATH = RESULTS_DIR / "trulens_latest.json"


def build_judge(pacer: Pacer, loop: asyncio.AbstractEventLoop):
    from trulens.providers.openai import OpenAI

    class GeminiJudge(OpenAI):
        def _create_chat_completion(self, prompt=None, messages=None, response_format=None, **kwargs):
            if response_format is None:
                # TruLens always sends a seed, which Gemini's endpoint rejects; NOT_GIVEN drops it from the request
                kwargs.setdefault("seed", openai.NOT_GIVEN)
                return super()._create_chat_completion(prompt=prompt, messages=messages, **kwargs)
            # TruLens only asks for structured output through the Responses API, which Gemini doesn't have, and
            # otherwise spends a second call turning the free-text verdict into JSON. Chat completions can take
            # the schema directly, so each score is one request
            completion = self.endpoint.client.chat.completions.parse(
                model=self.model_engine,
                messages=messages or [{"role": "system", "content": prompt}],
                response_format=response_format,
                temperature=kwargs.get("temperature", 0.0),
            )
            return completion.choices[0].message.parsed

    def pace(request: httpx.Request) -> None:
        # Feedback functions are sync and run in a worker thread; each request waits on the shared async Pacer
        if request.url.path.endswith("/chat/completions"):
            asyncio.run_coroutine_threadsafe(pacer.wait(), loop).result()

    settings = get_settings()
    judge = GeminiJudge(
        model_engine=settings.gemini_eval_model,
        api_key=settings.gemini_api_key,
        base_url=GEMINI_OPENAI_URL,
        max_retries=0,
        http_client=httpx.Client(event_hooks={"request": [pace]}, timeout=120),
    )
    # Gemini has no Responses API, so skip TruLens's probe for it and use chat completions from the start
    judge._set_capabilities({"responses_api": False})
    # A 429 goes straight up to the Pacer, which waits the delay Gemini suggests; TruLens would retry within seconds
    judge.endpoint.retries = 0
    return judge


async def _feedback(pacer: Pacer, fn, *args, retries: int = 3) -> tuple[float, Any]:
    for attempt in range(retries + 1):
        try:
            return await pacer.call(asyncio.to_thread, fn, *args, paced=False)
        except RuntimeError as exc:
            # Gemini answers 503 UNAVAILABLE when the model is busy; it passes within a minute or so
            if "UNAVAILABLE" not in str(exc) or attempt == retries:
                raise
            print("     model busy (503), retrying in 30s")
            await asyncio.sleep(30)


async def score_case(case: dict[str, Any], mode: RetrievalMode, judge, pacer: Pacer) -> dict[str, Any]:
    question = case["question"]
    result = await pacer.call(
        answer_rag,
        question,
        k=int(case.get("k", 4)),
        case_id=case["id"],
        mode=mode,
        model=get_settings().gemini_eval_model,
    )
    answer = result["answer"]
    contexts = [hit.content for hit in result["hits"]]

    per_chunk = [(await _feedback(pacer, judge.context_relevance_with_cot_reasons, question, c))[0] for c in contexts]
    grounded, grounded_reasons = await _feedback(
        pacer, judge.groundedness_measure_with_cot_reasons, "\n\n".join(contexts), answer
    )
    relevant, _ = await _feedback(pacer, judge.relevance_with_cot_reasons, question, answer)
    return {
        "id": case["id"],
        "answer": answer,
        "context_relevance": statistics.mean(per_chunk) if per_chunk else 0.0,
        "groundedness": grounded,
        "answer_relevance": relevant,
        "context_relevance_per_chunk": per_chunk,
        # The sentences the judge could not find support for, which is what to read when groundedness drops
        "ungrounded": [r["criteria"] for r in grounded_reasons.get("reasons", []) if r.get("score", 1) < 0.5],
    }


def _print_table(rows: list[dict[str, Any]], means: dict[str, float]) -> None:
    labels = {"context_relevance": "ctx_rel", "groundedness": "grounded", "answer_relevance": "ans_rel"}
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
    judge = build_judge(pacer, asyncio.get_running_loop())
    print(
        f"\nTruLens RAG triad: {len(selected)} cases, mode={mode.value}, judge={settings.gemini_eval_model}, "
        f"rpm={args.rpm:g}\n"
    )

    rows = []
    for case in selected:
        print(f"  -> {case['id']}")
        rows.append(await score_case(case, mode, judge, pacer))

    n = len(rows) or 1
    means = {m: sum(r[m] for r in rows) / n for m in METRICS}
    print()
    _print_table(rows, means)
    for row in rows:
        for sentence in row["ungrounded"]:
            print(f"  ungrounded in {row['id']}: {sentence}")
    print(f"\n{pacer.requests} chat requests, {pacer.rate_limited} rate-limited retries")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subset": "all" if args.all else "ragas_ci",
        "cases_scored": len(rows),
        "mode": mode.value,
        "eval_model": settings.gemini_eval_model,
        "metrics": means,
        "cases": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {RESULTS_PATH.relative_to(API_ROOT)}")
    flush_traces()


if __name__ == "__main__":
    asyncio.run(main())
