from sqlalchemy import column, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.services.retrieval.types import Candidate


def or_tsquery(question: str):
    """Build an OR tsquery from the question, parsed exactly like the chunks were.

    websearch_to_tsquery ANDs every term, so a full question rarely matches any chunk.
    Instead, run the question through to_tsvector('english', ...) (stemming, stopwords,
    and Postgres' tokenizer, which stores "ERR-4021" as 'err' + '-4021') and OR the
    resulting lexemes. ts_rank_cd then ranks chunks that share more of them higher,
    which behaves like BM25-style keyword search. The 'simple' config keeps the
    already-stemmed lexemes as they are.
    """
    lexemes = func.unnest(func.to_tsvector("english", question)).table_valued(
        "lexeme", "positions", "weights"
    )
    return (
        select(
            func.to_tsquery(
                "simple",
                func.coalesce(func.string_agg(func.quote_literal(column("lexeme")), " | "), ""),
            )
        )
        .select_from(lexemes)
        .scalar_subquery()
    )


async def keyword_search(
    db: AsyncSession,
    question: str,
    limit: int,
    document_ids: list[str] | None = None,
) -> list[Candidate]:
    """Postgres full-text search over the chunks' generated tsvector column."""
    if not question.strip():
        return []

    tsquery = or_tsquery(question)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery)
    stmt = (
        select(Chunk, Document.filename, rank.label("rank"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.tsv.op("@@")(tsquery))
        .order_by(rank.desc())
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
            scores={"keyword": float(r)},
        )
        for chunk, filename, r in rows
    ]
