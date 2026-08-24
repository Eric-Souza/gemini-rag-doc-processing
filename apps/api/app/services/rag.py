from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

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


async def stream_rag(
    message: str,
    document_ids: list[str] | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    hits = retrieve(message, document_ids=document_ids, k=4)
    for doc, score in hits:
        yield (
            "citation",
            {
                "document_id": doc.metadata.get("document_id", ""),
                "filename": doc.metadata.get("filename", "unknown"),
                "excerpt": doc.page_content[:400],
                "score": float(score) if score is not None else None,
            },
        )

    context = _format_context(hits)
    llm = get_chat_model()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {message}\n\nAnswer:"
        ),
    ]

    async for chunk in llm.astream(messages):
        text = chunk.content
        if isinstance(text, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in text
            )
        if text:
            yield ("token", {"text": str(text)})

    yield ("done", {})
