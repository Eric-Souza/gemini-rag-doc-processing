"""Qdrant as the vector store (VECTOR_STORE=qdrant).

Each chunk is one Qdrant point with two named vectors: its Gemini embedding ("dense") and a BM25 sparse vector
("bm25", term weights from fastembed, IDF computed by Qdrant). So hybrid search is a single Query API call: both
searches run as prefetches on the server, and Qdrant fuses them with RRF. Postgres stays the source of truth for
documents and chunks, which is also where `rebuild` reads from to refill Qdrant without re-embedding anything.

    python -m app.services.retrieval.qdrant_store   # rebuild the collection from the chunks in Postgres
"""

import asyncio
import uuid
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings
from app.services.retrieval.types import Candidate

DENSE = "dense"
SPARSE = "bm25"


@dataclass(frozen=True)
class ChunkPoint:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    embedding: list[float]


@lru_cache
def get_client() -> AsyncQdrantClient:
    url = get_settings().qdrant_url
    # ":memory:" runs Qdrant in-process, for tests
    return AsyncQdrantClient(location=":memory:") if url == ":memory:" else AsyncQdrantClient(url=url)


@lru_cache
def _bm25():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(get_settings().qdrant_sparse_model)


def _sparse(texts: list[str]) -> list[models.SparseVector]:
    return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in _bm25().embed(texts)]


def _sparse_query(text: str) -> models.SparseVector:
    # Queries weigh each term once; the document side carries the BM25 term-frequency weights
    e = next(iter(_bm25().query_embed(text)))
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be UUIDs or integers; this one is stable for a chunk id"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


_ready: set[str] = set()


async def ensure_collection() -> None:
    settings = get_settings()
    name = settings.qdrant_collection
    if name in _ready:
        return
    client = get_client()
    if not await client.collection_exists(name):
        await client.create_collection(
            name,
            vectors_config={DENSE: models.VectorParams(size=settings.embed_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        await client.create_payload_index(name, "document_id", models.PayloadSchemaType.KEYWORD)
    _ready.add(name)


async def upsert_chunks(chunks: list[ChunkPoint]) -> None:
    if not chunks:
        return
    await ensure_collection()
    sparse = await asyncio.to_thread(_sparse, [c.content for c in chunks])
    points = [
        models.PointStruct(
            id=point_id(c.chunk_id),
            vector={DENSE: c.embedding, SPARSE: s},
            payload={
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "filename": c.filename,
                "chunk_index": c.chunk_index,
                "content": c.content,
            },
        )
        for c, s in zip(chunks, sparse, strict=True)
    ]
    await get_client().upsert(get_settings().qdrant_collection, points, wait=True)


async def delete_document(document_id: str) -> None:
    await ensure_collection()
    await get_client().delete(
        get_settings().qdrant_collection,
        points_selector=models.FilterSelector(filter=_filter([document_id])),
        wait=True,
    )


def _filter(document_ids: list[str] | None) -> models.Filter | None:
    if not document_ids:
        return None
    return models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))])


def _candidates(points: list[models.ScoredPoint], score_key: str) -> list[Candidate]:
    return [
        Candidate(
            chunk_id=p.payload["chunk_id"],
            document_id=p.payload["document_id"],
            filename=p.payload["filename"],
            chunk_index=p.payload["chunk_index"],
            content=p.payload["content"],
            scores={score_key: float(p.score)},
        )
        for p in points
    ]


async def dense_search(query_embedding: list[float], limit: int, document_ids: list[str] | None = None):
    await ensure_collection()
    result = await get_client().query_points(
        get_settings().qdrant_collection,
        query=query_embedding,
        using=DENSE,
        query_filter=_filter(document_ids),
        limit=limit,
        with_payload=True,
    )
    return _candidates(result.points, "vector")


async def sparse_search(question: str, limit: int, document_ids: list[str] | None = None) -> list[Candidate]:
    await ensure_collection()
    query = await asyncio.to_thread(_sparse_query, question)
    if not query.indices:  # only stopwords
        return []
    result = await get_client().query_points(
        get_settings().qdrant_collection,
        query=query,
        using=SPARSE,
        query_filter=_filter(document_ids),
        limit=limit,
        with_payload=True,
    )
    return _candidates(result.points, "keyword")


async def hybrid_search(
    question: str,
    query_embedding: list[float],
    limit: int,
    candidates: int,
    document_ids: list[str] | None = None,
) -> list[Candidate]:
    """Dense and BM25 searches as prefetches, fused by Qdrant with RRF, in one request"""
    await ensure_collection()
    settings = get_settings()
    where = _filter(document_ids)
    sparse = await asyncio.to_thread(_sparse_query, question)
    prefetch = [models.Prefetch(query=query_embedding, using=DENSE, filter=where, limit=candidates)]
    if sparse.indices:
        prefetch.append(models.Prefetch(query=sparse, using=SPARSE, filter=where, limit=candidates))
    result = await get_client().query_points(
        settings.qdrant_collection,
        prefetch=prefetch,
        # The same k as the Postgres path's RRF, so the two stores fuse the same way
        query=models.RrfQuery(rrf=models.Rrf(k=settings.rrf_k)),
        limit=limit,
        with_payload=True,
    )
    return _candidates(result.points, "rrf")


async def rebuild() -> int:
    """Recreate the collection from every chunk of every ready document in Postgres, embeddings included"""
    from sqlalchemy import select

    from app.db import AsyncSessionLocal
    from app.models import Chunk, Document, DocumentStatus

    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(Chunk, Document.filename)
                .join(Document, Document.id == Chunk.document_id)
                .where(Document.status == DocumentStatus.ready)
            )
        ).all()
    name = get_settings().qdrant_collection
    await get_client().delete_collection(name)
    _ready.discard(name)
    chunks = [
        ChunkPoint(c.id, c.document_id, filename, c.chunk_index, c.content, [float(x) for x in c.embedding])
        for c, filename in rows
    ]
    for start in range(0, len(chunks), 256):
        await upsert_chunks(chunks[start : start + 256])
    return len(chunks)


if __name__ == "__main__":
    print(f"Rebuilt Qdrant from {asyncio.run(rebuild())} chunks")
