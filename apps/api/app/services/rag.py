from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.services.llm import get_chat_model
from app.services.vectorstore import get_vectorstore

SYSTEM_PROMPT = """You are a helpful document assistant.
Answer ONLY using the provided context excerpts.
If the context is insufficient, say you don't have enough information in the uploaded documents.
Be concise and cite relevant details from the context when possible.
Do not invent facts."""


def _format_context(docs_with_scores: list[tuple[Any, float]]) -> str:
    parts: list[str] = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        filename = doc.metadata.get("filename", "unknown")
        parts.append(f"[{i}] ({filename}, score={score:.3f})\n{doc.page_content}")
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
    return {"tags": tags, "metadata": metadata, "run_name": f"doc-qa-{mode}"}


def retrieve(
    query: str,
    document_ids: list[str] | None = None,
    k: int = 4,
) -> list[tuple[Any, float]]:
    store = get_vectorstore()
    filt = {"document_id": {"$in": document_ids}} if document_ids else None
    try:
        if filt:
            return store.similarity_search_with_score(query, k=k, filter=filt)
        return store.similarity_search_with_score(query, k=k)
    except Exception:
        # Some Chroma filter shapes differ; fall back to unfiltered then filter in Python
        results = store.similarity_search_with_score(query, k=max(k * 3, 12))
        if not document_ids:
            return results[:k]
        filtered = [(d, s) for d, s in results if d.metadata.get("document_id") in document_ids]
        return filtered[:k]


def _citations_from_hits(hits: list[tuple[Any, float]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for doc, score in hits:
        citations.append(
            {
                "document_id": doc.metadata.get("document_id", ""),
                "filename": doc.metadata.get("filename", "unknown"),
                "excerpt": doc.page_content[:400],
                "score": float(score) if score is not None else None,
            }
        )
    return citations


async def answer_rag(
    message: str,
    document_ids: list[str] | None = None,
    k: int = 4,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Non-streaming RAG used by evals (and any sync callers)."""
    hits = retrieve(message, document_ids=document_ids, k=k)
    citations = _citations_from_hits(hits)
    context = _format_context(hits)
    llm = get_chat_model()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {message}\n\nAnswer:"),
    ]
    config = _run_config("rag", document_ids=document_ids, case_id=case_id)
    result = await llm.ainvoke(messages, config=config)
    return {
        "answer": _message_text(result.content),
        "citations": citations,
        "context": context,
        "hits": hits,
    }


async def stream_rag(
    message: str,
    document_ids: list[str] | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    yield ("status", {"message": "Retrieving documents…"})
    hits = retrieve(message, document_ids=document_ids, k=4)
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
    config = _run_config("rag", document_ids=document_ids)

    first_token = True
    async for chunk in llm.astream(messages, config=config):
        text = _message_text(chunk.content)
        if text:
            if first_token:
                yield ("status", {"message": "Streaming answer…"})
                first_token = False
            yield ("token", {"text": text})

    yield ("done", {})
