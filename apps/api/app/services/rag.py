from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.services.llm import get_chat_model
from app.services.retrieval import Candidate, retrieve
from app.services.tracing import langchain_callbacks, trace

SYSTEM_PROMPT = """You are a helpful document assistant.
Answer ONLY using the provided context excerpts.
If the context is insufficient, say you don't have enough information in the uploaded documents.
Be concise and cite relevant details from the context when possible.
Do not invent facts."""


def _format_context(hits: list[Candidate]) -> str:
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        parts.append(f"[{i}] ({hit.filename}, score={hit.score:.3f})\n{hit.content}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "")


def _run_config(
    mode: str,
    document_ids: list[str] | None = None,
    case_id: str | None = None,
) -> RunnableConfig:
    tags = [f"mode:{mode}"]
    metadata: dict[str, Any] = {"mode": mode}
    if document_ids:
        metadata["document_ids"] = document_ids
    if case_id:
        tags.append(f"eval:{case_id}")
        metadata["eval_case_id"] = case_id
    return {
        "tags": tags,
        "metadata": metadata,
        "run_name": f"doc-qa-{mode}",
        "callbacks": langchain_callbacks(),
    }


def _citations_from_hits(hits: list[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": hit.document_id,
            "filename": hit.filename,
            "excerpt": hit.content[:400],
            "score": hit.score,
        }
        for hit in hits
    ]


async def answer_rag(
    message: str,
    document_ids: list[str] | None = None,
    k: int = 4,
    case_id: str | None = None,
    mode: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Non-streaming RAG used by evals (and any sync callers)."""
    config = _run_config("rag", document_ids=document_ids, case_id=case_id)
    with trace("doc-qa-rag", tags=config["tags"], metadata=config["metadata"], input=message) as root:
        hits = await retrieve(message, k=k, document_ids=document_ids, mode=mode)
        citations = _citations_from_hits(hits)
        context = _format_context(hits)
        llm = get_chat_model(model)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {message}\n\nAnswer:"),
        ]
        result = await llm.ainvoke(messages, config=config)
        answer = _message_text(result.content)
        root.update(output=answer)
    return {
        "answer": answer,
        "citations": citations,
        "context": context,
        "hits": hits,
    }


async def stream_rag(
    message: str,
    document_ids: list[str] | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    config = _run_config("rag", document_ids=document_ids)
    with trace("doc-qa-rag", tags=config["tags"], metadata=config["metadata"], input=message) as root:
        yield ("status", {"message": "Retrieving documents…"})
        hits = await retrieve(message, k=4, document_ids=document_ids)
        for citation in _citations_from_hits(hits):
            yield ("citation", citation)

        yield ("status", {"message": "Sending to LLM…"})
        context = _format_context(hits)
        llm = get_chat_model()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {message}\n\nAnswer:"
            ),
        ]

        first_token = True
        parts: list[str] = []
        async for chunk in llm.astream(messages, config=config):
            text = _message_text(chunk.content)
            if text:
                if first_token:
                    yield ("status", {"message": "Streaming answer…"})
                    first_token = False
                parts.append(text)
                yield ("token", {"text": text})
        root.update(output="".join(parts))

    yield ("done", {})
