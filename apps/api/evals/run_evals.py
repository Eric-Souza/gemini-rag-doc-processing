"""Run golden-set retrieval + answer evals against the local Chroma index.

Prerequisite: sample docs from repo `samples/` must already be uploaded/indexed
(status ready in the API).

Usage (from apps/api with venv active):

    python -m evals.run_evals
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

# Ensure apps/api is on path when run as a script
API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.config import apply_langsmith_env, get_settings  # noqa: E402
from app.services.llm import get_chat_model  # noqa: E402
from app.services.rag import answer_rag, retrieve  # noqa: E402
from app.services.vectorstore import get_vectorstore  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_contains(answer: str, must_contain: list[str]) -> bool:
    norm = _normalize(answer)
    for phrase in must_contain:
        if _normalize(phrase) in norm:
            return True
    return False


def score_hit_at_k(hits: list[tuple[Any, float]], expected_filenames: list[str]) -> bool:
    found = {doc.metadata.get("filename", "") for doc, _ in hits}
    return any(name in found for name in expected_filenames)


async def score_faithfulness(context: str, answer: str) -> bool:
    llm = get_chat_model()
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


def _ensure_index_ready(expected_names: set[str]) -> None:
    store = get_vectorstore()
    try:
        data = store.get(include=["metadatas"])
        metas = data.get("metadatas") or []
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Could not read Chroma index: {exc}") from exc

    if not metas:
        raise SystemExit(
            "Chroma index is empty. Upload samples/ docs via the API/UI and wait until status is ready."
        )

    present = {m.get("filename") for m in metas if m}
    missing = expected_names - present
    if missing:
        raise SystemExit(
            "Indexed corpus is missing expected filenames: "
            + ", ".join(sorted(missing))
            + ". Upload the samples/ files first."
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


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    question = case["question"]
    k = int(case.get("k", 4))
    expected = case["expected_filenames"]
    must_contain = case["must_contain"]

    hits = retrieve(question, k=k)
    hit = score_hit_at_k(hits, expected)

    result = await answer_rag(question, k=k, case_id=case["id"])
    answer = result["answer"]
    contains = score_contains(answer, must_contain)
    faithful = await score_faithfulness(result["context"], answer)

    _log_langsmith_feedback(case["id"], question, answer, hit, contains, faithful)

    return {
        "id": case["id"],
        "hit": hit,
        "contains": contains,
        "faithful": faithful,
        "answer": answer,
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ("id", "hit@k", "contains", "faithful")
    print(f"{headers[0]:<24} {headers[1]:<8} {headers[2]:<10} {headers[3]:<10}")
    print("-" * 56)
    for row in rows:
        print(
            f"{row['id']:<24} "
            f"{('Y' if row['hit'] else 'N'):<8} "
            f"{('Y' if row['contains'] else 'N'):<10} "
            f"{('Y' if row['faithful'] else 'N'):<10}"
        )
    n = len(rows) or 1
    hit_rate = sum(1 for r in rows if r["hit"]) / n
    contains_rate = sum(1 for r in rows if r["contains"]) / n
    faithful_rate = sum(1 for r in rows if r["faithful"]) / n
    print("-" * 56)
    print(
        f"{'AGGREGATE':<24} {hit_rate:>6.0%}   {contains_rate:>6.0%}     {faithful_rate:>6.0%}"
    )


async def main() -> None:
    apply_langsmith_env()
    settings = get_settings()
    if not settings.gemini_configured:
        raise SystemExit("GEMINI_API_KEY is not set. Add it to .env first.")

    cases: list[dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    expected_names: set[str] = set()
    for case in cases:
        expected_names.update(case["expected_filenames"])
    _ensure_index_ready(expected_names)

    print(f"Running {len(cases)} eval cases (LangSmith={'on' if settings.langsmith_configured else 'off'})...")
    rows: list[dict[str, Any]] = []
    for case in cases:
        print(f"  -> {case['id']}")
        rows.append(await run_case(case))
    print()
    _print_table(rows)


if __name__ == "__main__":
    asyncio.run(main())
