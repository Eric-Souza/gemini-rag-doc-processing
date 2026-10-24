"""Golden-set evals: retrieval quality per mode, plus answer quality.

Retrieval metrics are chunk-level: a case counts as a hit when a retrieved chunk
contains the case's `evidence` text.

  hit@1   the top chunk contains the evidence
  hit@k   any of the top-k chunks contains it
  MRR     mean reciprocal rank of the first chunk containing it

Answer metrics (one LLM call per case, plus one judge call):

  contains   the answer includes one of the expected phrases
  faithful   an LLM judge says the answer is supported by the retrieved context

Usage (from apps/api with the venv active and Postgres running):

    python -m evals.run_evals --seed                       # index samples/, then run
    python -m evals.run_evals --modes vector,hybrid,hybrid_rerank --retrieval-only
    python -m evals.run_evals --mode hybrid_rerank          # retrieval + answers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai.chat_models import GoogleRateLimitError
from sqlalchemy import select

# Ensure apps/api is on path when run as a script
API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.config import REPO_ROOT, apply_langsmith_env, get_settings  # noqa: E402
from app.db import AsyncSessionLocal, init_db  # noqa: E402
from app.models import Document, DocumentStatus  # noqa: E402
from app.services.ingest import ingest_document  # noqa: E402
from app.services.llm import get_chat_model  # noqa: E402
from app.services.rag import answer_rag  # noqa: E402
from app.services.retrieval import Candidate, RetrievalMode, retrieve  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SAMPLES_DIR = REPO_ROOT / "samples"
SAMPLE_SUFFIXES = {".md", ".txt", ".pdf"}


def _normalize(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_contains(answer: str, must_contain: list[str]) -> bool:
    norm = _normalize(answer)
    return any(_normalize(phrase) in norm for phrase in must_contain)


def evidence_rank(hits: list[Candidate], evidence: str) -> int | None:
    """1-based rank of the first chunk containing the evidence, or None."""
    needle = _normalize(evidence)
    for rank, hit in enumerate(hits, start=1):
        if needle in _normalize(hit.content):
            return rank
    return None


async def score_faithfulness(context: str, answer: str) -> bool:
    llm = get_chat_model(get_settings().gemini_eval_model)
    messages = [
        SystemMessage(
            content=(
                "You are a strict faithfulness judge. Reply with ONLY yes or no. "
                "Say yes if the answer is fully supported by the context and invents "
                "no facts. Say no otherwise."
            )
        ),
        HumanMessage(content=f"Context:\n{context}\n\nAnswer:\n{answer}\n\nSupported?"),
    ]
    result = await llm.ainvoke(
        messages,
        config={"tags": ["eval:faithfulness"], "run_name": "eval-faithfulness"},
    )
    content = result.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    text = str(content or "").strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return "yes" in text.split()[:3]


async def _indexed_filenames() -> set[str]:
    async with AsyncSessionLocal() as db:
        rows = await db.execute(
            select(Document.filename).where(Document.status == DocumentStatus.ready)
        )
        return set(rows.scalars().all())


async def seed_samples() -> None:
    """Index every file in samples/ that isn't indexed yet (no API server needed)."""
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    indexed = await _indexed_filenames()
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in SAMPLE_SUFFIXES or path.name == "README.md":
            continue
        if path.name in indexed:
            continue
        async with AsyncSessionLocal() as db:
            doc = Document(
                filename=path.name,
                content_type="text/plain",
                size_bytes=path.stat().st_size,
                status=DocumentStatus.pending,
            )
            db.add(doc)
            await db.commit()
            doc_id = doc.id
        shutil.copyfile(path, Path(settings.upload_dir) / f"{doc_id}{path.suffix.lower()}")
        await ingest_document(doc_id)
        async with AsyncSessionLocal() as db:
            doc = await db.get(Document, doc_id)
            if doc.status != DocumentStatus.ready:
                raise SystemExit(f"Indexing {path.name} failed: {doc.error_message}")
            print(f"  indexed {path.name} ({doc.chunk_count} chunks)")


async def _ensure_index_ready(expected_names: set[str]) -> None:
    missing = expected_names - await _indexed_filenames()
    if missing:
        raise SystemExit(
            "Indexed corpus is missing: "
            + ", ".join(sorted(missing))
            + ". Run with --seed, or upload the samples/ files first."
        )


def _log_langsmith_feedback(
    case_id: str,
    question: str,
    answer: str,
    hit: bool,
    contains: bool,
    faithful: bool,
) -> None:
    settings = get_settings()
    if not settings.langsmith_configured:
        return
    try:
        from langsmith import Client

        client = Client()
        run = client.create_run(
            name=f"eval-{case_id}",
            run_type="chain",
            inputs={"question": question},
            outputs={"answer": answer},
            project_name=settings.langsmith_project,
            tags=["eval", f"case:{case_id}"],
        )
        run_id = getattr(run, "id", None) or (run.get("id") if isinstance(run, dict) else None)
        if not run_id:
            return
        for key, value in (
            ("hit_at_k", hit),
            ("answer_contains", contains),
            ("faithfulness", faithful),
        ):
            client.create_feedback(
                run_id=run_id,
                key=key,
                score=1.0 if value else 0.0,
            )
        client.update_run(run_id, outputs={"answer": answer}, end_time=None)
    except Exception as exc:  # noqa: BLE001
        print(f"  (LangSmith feedback skipped for {case_id}: {exc})")


async def run_retrieval(cases: list[dict[str, Any]], mode: RetrievalMode) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        k = int(case.get("k", 4))
        hits = await retrieve(case["question"], k=k, mode=mode)
        rank = evidence_rank(hits, case["evidence"])
        rows.append(
            {
                "id": case["id"],
                "rank": rank,
                "top_file": hits[0].filename if hits else None,
            }
        )
    n = len(rows) or 1
    return {
        "mode": mode.value,
        "hit_at_1": sum(1 for r in rows if r["rank"] == 1) / n,
        "hit_at_k": sum(1 for r in rows if r["rank"] is not None) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "cases": rows,
    }


class Pacer:
    """Spaces out chat calls to stay under the Gemini free tier's requests-per-minute limit,
    and waits out a 429 using the retry delay the API suggests."""

    def __init__(self, rpm: float, retries: int = 3) -> None:
        self.interval = 60 / rpm if rpm > 0 else 0.0
        self.retries = retries
        self._next = 0.0

    async def call(self, fn, *args, **kwargs):
        for attempt in range(self.retries + 1):
            wait = self._next - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._next = time.monotonic() + self.interval
            try:
                return await fn(*args, **kwargs)
            except GoogleRateLimitError as exc:
                if "PerDay" in str(exc):
                    raise SystemExit(
                        "Gemini free-tier daily request quota is used up for this model. "
                        "Retry tomorrow or set GEMINI_CHAT_MODEL to a model with quota left."
                    ) from exc
                if attempt == self.retries:
                    raise
                match = re.search(r"retry in ([\d.]+)s", str(exc))
                delay = float(match.group(1)) + 1 if match else 60.0
                print(f"     rate limited, retrying in {delay:.0f}s")
                await asyncio.sleep(delay)


async def run_answers(cases: list[dict[str, Any]], mode: RetrievalMode, pacer: Pacer) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        print(f"  -> {case['id']}")
        k = int(case.get("k", 4))
        result = await pacer.call(
            answer_rag,
            case["question"],
            k=k,
            case_id=case["id"],
            mode=mode,
            model=get_settings().gemini_eval_model,
        )
        answer = result["answer"]
        hit = evidence_rank(result["hits"], case["evidence"]) is not None
        contains = score_contains(answer, case["must_contain"])
        faithful = await pacer.call(score_faithfulness, result["context"], answer)
        _log_langsmith_feedback(case["id"], case["question"], answer, hit, contains, faithful)
        rows.append(
            {"id": case["id"], "hit": hit, "contains": contains, "faithful": faithful, "answer": answer}
        )
    n = len(rows) or 1
    return {
        "mode": mode.value,
        "contains": sum(1 for r in rows if r["contains"]) / n,
        "faithful": sum(1 for r in rows if r["faithful"]) / n,
        "cases": rows,
    }


def _print_retrieval(results: list[dict[str, Any]]) -> None:
    modes = [r["mode"] for r in results]
    print(f"{'case':<30}" + "".join(f"{m:>15}" for m in modes))
    print("-" * (30 + 15 * len(modes)))
    for i, case in enumerate(results[0]["cases"]):
        cells = []
        for r in results:
            rank = r["cases"][i]["rank"]
            cells.append(f"{(f'#{rank}' if rank else 'miss'):>15}")
        print(f"{case['id']:<30}" + "".join(cells))
    print("-" * (30 + 15 * len(modes)))
    for metric, label in (("hit_at_1", "hit@1"), ("hit_at_k", "hit@k"), ("mrr", "MRR")):
        values = "".join(
            f"{(f'{r[metric]:.0%}' if metric != 'mrr' else f'{r[metric]:.3f}'):>15}"
            for r in results
        )
        print(f"{label:<30}" + values)


def _print_answers(result: dict[str, Any]) -> None:
    print(f"{'case':<30} {'hit@k':<8} {'contains':<10} {'faithful':<10}")
    print("-" * 60)
    for row in result["cases"]:
        print(
            f"{row['id']:<30} "
            f"{('Y' if row['hit'] else 'N'):<8} "
            f"{('Y' if row['contains'] else 'N'):<10} "
            f"{('Y' if row['faithful'] else 'N'):<10}"
        )
    print("-" * 60)
    print(f"{'AGGREGATE':<30} {'':<8} {result['contains']:<10.0%} {result['faithful']:<10.0%}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", action="store_true", help="index samples/ before running")
    parser.add_argument(
        "--modes",
        default="vector,hybrid,hybrid_rerank",
        help="comma-separated retrieval modes to compare",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="retrieval mode for the answer evals (default: RETRIEVAL_MODE setting)",
    )
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer evals (no chat calls)")
    parser.add_argument(
        "--rpm",
        type=float,
        default=5,
        help="max chat requests per minute (Gemini free tier: 5; 0 disables pacing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="answer-eval only the first N cases (2 chat calls each) to save free-tier quota",
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
    await _ensure_index_ready({name for case in cases for name in case["expected_filenames"]})

    modes = [RetrievalMode(m.strip()) for m in args.modes.split(",") if m.strip()]
    print(f"\nRetrieval: {len(cases)} cases x {len(modes)} modes\n")
    retrieval_results = [await run_retrieval(cases, mode) for mode in modes]
    _print_retrieval(retrieval_results)

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chat_model": settings.gemini_chat_model,
        "eval_model": settings.gemini_eval_model,
        "embed_model": settings.gemini_embed_model,
        "reranker_model": settings.reranker_model,
        "retrieval": retrieval_results,
    }

    if not args.retrieval_only:
        mode = RetrievalMode(args.mode or settings.retrieval_mode)
        answer_cases = cases[: args.limit] if args.limit > 0 else cases
        print(
            f"\nAnswers: {len(answer_cases)} cases, mode={mode.value}, model={settings.gemini_eval_model} "
            f"(LangSmith={'on' if settings.langsmith_configured else 'off'})"
        )
        answers = await run_answers(answer_cases, mode, Pacer(args.rpm))
        print()
        _print_answers(answers)
        report["answers"] = answers

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(API_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
