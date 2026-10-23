import asyncio
from enum import Enum

from app.config import get_settings
from app.db import AsyncSessionLocal
from app.services.llm import embed_query
from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.keyword import keyword_search
from app.services.retrieval.rerank import rerank
from app.services.retrieval.types import Candidate
from app.services.retrieval.vector import vector_search


class RetrievalMode(str, Enum):
    vector = "vector"
    hybrid = "hybrid"
    hybrid_rerank = "hybrid_rerank"


async def retrieve(
    question: str,
    k: int = 4,
    document_ids: list[str] | None = None,
    mode: RetrievalMode | str | None = None,
) -> list[Candidate]:
    """Retrieve the top-k chunks for a question.

    vector:        pgvector similarity only
    hybrid:        full-text + vector, fused with RRF
    hybrid_rerank: hybrid candidates reordered by a cross-encoder
    """
    settings = get_settings()
    mode = RetrievalMode(mode or settings.retrieval_mode)
    candidates_n = max(settings.retrieval_candidates, k)

    query_embedding = await embed_query(question)

    if mode is RetrievalMode.vector:
        async with AsyncSessionLocal() as db:
            return await vector_search(db, query_embedding, k, document_ids)

    # Separate sessions so both searches run concurrently.
    async def _vector() -> list[Candidate]:
        async with AsyncSessionLocal() as db:
            return await vector_search(db, query_embedding, candidates_n, document_ids)

    async def _keyword() -> list[Candidate]:
        async with AsyncSessionLocal() as db:
            return await keyword_search(db, question, candidates_n, document_ids)

    vector_hits, keyword_hits = await asyncio.gather(_vector(), _keyword())
    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=settings.rrf_k)

    if mode is RetrievalMode.hybrid:
        return fused[:k]
    return await rerank(question, fused, top_k=k)
