import asyncio
from enum import Enum

from app.config import get_settings
from app.db import AsyncSessionLocal
from app.services.llm import embed_query
from app.services.retrieval import qdrant_store
from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.keyword import keyword_search
from app.services.retrieval.rerank import rerank
from app.services.retrieval.types import Candidate
from app.services.retrieval.vector import vector_search
from app.services.tracing import observe


class RetrievalMode(str, Enum):
    vector = "vector"
    hybrid = "hybrid"
    hybrid_rerank = "hybrid_rerank"


async def retrieve(
    question: str,
    k: int = 4,
    document_ids: list[str] | None = None,
    mode: RetrievalMode | str | None = None,
    store: str | None = None,
) -> list[Candidate]:
    """Retrieve the top-k chunks for a question.

    vector:        vector similarity only
    hybrid:        keyword + vector, fused with RRF
    hybrid_rerank: hybrid candidates reordered by a cross-encoder

    store picks where the searches run: "pgvector" (pgvector + Postgres full-text, fused here) or "qdrant" (dense +
    BM25 sparse vectors, fused by Qdrant). The reranker is the same for both.
    """
    settings = get_settings()
    mode = RetrievalMode(mode or settings.retrieval_mode)
    store = store or settings.vector_store
    if store not in ("pgvector", "qdrant"):
        raise ValueError(f"Unknown vector store {store!r}")
    candidates_n = max(settings.retrieval_candidates, k)

    with observe(
        "retrieve",
        as_type="retriever",
        input={"question": question, "k": k, "document_ids": document_ids},
        metadata={"mode": mode.value, "store": store, "candidates": candidates_n},
    ) as span:
        retriever = _retrieve_qdrant if store == "qdrant" else _retrieve
        hits = await retriever(question, k, document_ids, mode, candidates_n)
        span.update(output=_summarize(hits))
        return hits


def _summarize(hits: list[Candidate]) -> list[dict]:
    """Span output: which chunks each stage returned and their scores, plus a short preview."""
    return [
        {
            "filename": h.filename,
            "chunk_index": h.chunk_index,
            "preview": h.content[:120],
            "scores": h.scores,
        }
        for h in hits
    ]


async def _retrieve(
    question: str,
    k: int,
    document_ids: list[str] | None,
    mode: RetrievalMode,
    candidates_n: int,
) -> list[Candidate]:
    settings = get_settings()
    with observe("embed_query", as_type="embedding", model=settings.gemini_embed_model):
        query_embedding = await embed_query(question)

    if mode is RetrievalMode.vector:
        with observe("vector_search") as span:
            async with AsyncSessionLocal() as db:
                hits = await vector_search(db, query_embedding, k, document_ids)
            span.update(output=_summarize(hits))
            return hits

    # Separate sessions so both searches run concurrently.
    async def _vector() -> list[Candidate]:
        with observe("vector_search") as span:
            async with AsyncSessionLocal() as db:
                hits = await vector_search(db, query_embedding, candidates_n, document_ids)
            span.update(output=_summarize(hits))
            return hits

    async def _keyword() -> list[Candidate]:
        with observe("keyword_search") as span:
            async with AsyncSessionLocal() as db:
                hits = await keyword_search(db, question, candidates_n, document_ids)
            span.update(output=_summarize(hits))
            return hits

    vector_hits, keyword_hits = await asyncio.gather(_vector(), _keyword())
    with observe("rrf", metadata={"rrf_k": settings.rrf_k}) as span:
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=settings.rrf_k)
        span.update(output=_summarize(fused[: max(k, 10)]))

    if mode is RetrievalMode.hybrid:
        return fused[:k]
    with observe("rerank", metadata={"model": settings.reranker_model}) as span:
        reranked = await rerank(question, fused, top_k=k)
        span.update(output=_summarize(reranked))
        return reranked


async def _retrieve_qdrant(
    question: str,
    k: int,
    document_ids: list[str] | None,
    mode: RetrievalMode,
    candidates_n: int,
) -> list[Candidate]:
    settings = get_settings()
    with observe("embed_query", as_type="embedding", model=settings.gemini_embed_model):
        query_embedding = await embed_query(question)

    if mode is RetrievalMode.vector:
        with observe("qdrant_dense_search") as span:
            hits = await qdrant_store.dense_search(query_embedding, k, document_ids)
            span.update(output=_summarize(hits))
            return hits

    # The pgvector path hands the reranker every fused candidate, up to two lists' worth; so does this one
    limit = k if mode is RetrievalMode.hybrid else 2 * candidates_n
    with observe("qdrant_hybrid_search", metadata={"rrf_k": settings.rrf_k}) as span:
        fused = await qdrant_store.hybrid_search(question, query_embedding, limit, candidates_n, document_ids)
        span.update(output=_summarize(fused[: max(k, 10)]))

    if mode is RetrievalMode.hybrid:
        return fused
    with observe("rerank", metadata={"model": settings.reranker_model}) as span:
        reranked = await rerank(question, fused, top_k=k)
        span.update(output=_summarize(reranked))
        return reranked
