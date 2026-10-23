from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.services.retrieval.types import Candidate


async def vector_search(
    db: AsyncSession,
    query_embedding: list[float],
    limit: int,
    document_ids: list[str] | None = None,
) -> list[Candidate]:
    """Nearest chunks by cosine distance (HNSW index)."""
    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(limit)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    rows = (await db.execute(stmt)).all()
    return [
        Candidate(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            scores={"vector": 1.0 - float(dist)},
        )
        for chunk, filename, dist in rows
    ]
