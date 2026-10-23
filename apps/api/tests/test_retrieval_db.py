"""Retrieval against a real Postgres + pgvector database, with synthetic embeddings (no Gemini calls)."""

import pytest
from sqlalchemy import func, select

from app.db import AsyncSessionLocal
from app.models import Chunk, Document, DocumentStatus
from app.services.retrieval import pipeline
from app.services.retrieval.keyword import keyword_search
from app.services.retrieval.vector import vector_search
from tests.conftest import axis_vector

CHUNKS = {
    "runbook": [
        ("ERR-4012: Gateway timeout. The processor did not respond within 8 seconds.", axis_vector((0, 1.0))),
        ("ERR-4021: OAuth token refresh failed. Re-sync the secret in the vault.", axis_vector((0, 0.3), (1, 1.0))),
    ],
    "catalog": [
        ("NV-220X wireless headset. Battery life: 28 hours.", axis_vector((2, 1.0))),
        ("SP-30 conference speakerphone for rooms of up to 12 people.", axis_vector((3, 1.0), (0, 0.5))),
    ],
}


@pytest.fixture(scope="module")
async def doc_ids():
    ids: dict[str, str] = {}
    async with AsyncSessionLocal() as db:
        for name, chunks in CHUNKS.items():
            doc = Document(filename=f"{name}.md", status=DocumentStatus.ready, chunk_count=len(chunks))
            db.add(doc)
            await db.flush()
            ids[name] = doc.id
            db.add_all(
                Chunk(id=f"{doc.id}_{i}", document_id=doc.id, chunk_index=i, content=text, embedding=emb)
                for i, (text, emb) in enumerate(chunks)
            )
        await db.commit()
    yield ids
    async with AsyncSessionLocal() as db:
        for doc_id in ids.values():
            doc = await db.get(Document, doc_id)
            if doc:
                await db.delete(doc)
        await db.commit()


async def test_keyword_search_finds_exact_error_code(doc_ids):
    async with AsyncSessionLocal() as db:
        hits = await keyword_search(db, "What should on-call do about ERR-4021?", limit=5)
    assert hits[0].content.startswith("ERR-4021")
    assert hits[0].scores["keyword"] > 0


async def test_keyword_search_matches_postgres_tokens_for_codes(doc_ids):
    # Postgres stores "ERR-4021" as 'err' + '-4021' and "NV-220X" as 'nv-220x', 'nv', '220x'
    async with AsyncSessionLocal() as db:
        err = await keyword_search(db, "ERR-4021?", limit=5)
        nv = await keyword_search(db, "battery of the NV-220X", limit=5)
    assert err[0].content.startswith("ERR-4021")
    assert err[0].scores["keyword"] > err[1].scores["keyword"]  # ERR-4012 only shares 'err'
    assert nv[0].content.startswith("NV-220X")


async def test_keyword_search_returns_nothing_for_unrelated_or_stopword_queries(doc_ids):
    async with AsyncSessionLocal() as db:
        assert await keyword_search(db, "quarterly marketing budget", limit=5) == []
        assert await keyword_search(db, "what is the", limit=5) == []
        assert await keyword_search(db, "   ", limit=5) == []


async def test_vector_search_orders_by_cosine_similarity(doc_ids):
    async with AsyncSessionLocal() as db:
        hits = await vector_search(db, axis_vector((2, 1.0)), limit=2)
    assert hits[0].content.startswith("NV-220X")
    assert hits[0].scores["vector"] == pytest.approx(1.0)


async def test_document_filter_limits_results(doc_ids):
    async with AsyncSessionLocal() as db:
        hits = await vector_search(db, axis_vector((0, 1.0)), limit=10, document_ids=[doc_ids["catalog"]])
    assert {h.filename for h in hits} == {"catalog.md"}


async def test_hybrid_fixes_a_vector_miss(doc_ids, monkeypatch):
    # The query embedding sits closest to ERR-4012, then SP-30; ERR-4021 (the one the question
    # names) is only 3rd by vector, so vector-only top-2 misses it. Keyword search ranks it 1st.
    async def fake_embed(_question: str) -> list[float]:
        return axis_vector((0, 1.0))

    monkeypatch.setattr(pipeline, "embed_query", fake_embed)
    question = "What should on-call do about ERR-4021?"

    vector_only = await pipeline.retrieve(question, k=2, mode="vector")
    hybrid = await pipeline.retrieve(question, k=2, mode="hybrid")

    assert not any(c.content.startswith("ERR-4021") for c in vector_only)
    assert any(c.content.startswith("ERR-4021") for c in hybrid)


async def test_reranker_reorders_candidates(doc_ids, monkeypatch):
    pytest.importorskip("sentence_transformers")

    async def fake_embed(_question: str) -> list[float]:
        return axis_vector((2, 1.0))  # points at the headset chunk

    monkeypatch.setattr(pipeline, "embed_query", fake_embed)
    hits = await pipeline.retrieve(
        "How many people fit in a room with the conference speakerphone?", k=1, mode="hybrid_rerank"
    )
    assert hits[0].content.startswith("SP-30")
    assert "rerank" in hits[0].scores


async def test_deleting_document_cascades_to_chunks():
    async with AsyncSessionLocal() as db:
        doc = Document(filename="temp.md", status=DocumentStatus.ready)
        db.add(doc)
        await db.flush()
        db.add(Chunk(id=f"{doc.id}_0", document_id=doc.id, chunk_index=0, content="temp", embedding=axis_vector((5, 1.0))))
        await db.commit()

        await db.delete(doc)
        await db.commit()
        remaining = await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id))
    assert remaining == 0


async def test_rerank_puts_the_named_error_code_first(doc_ids, monkeypatch):
    pytest.importorskip("sentence_transformers")

    async def fake_embed(_question: str) -> list[float]:
        return axis_vector((0, 1.0))  # closest to ERR-4012, the wrong code

    monkeypatch.setattr(pipeline, "embed_query", fake_embed)
    hits = await pipeline.retrieve("What should on-call do about ERR-4021?", k=1, mode="hybrid_rerank")
    assert hits[0].content.startswith("ERR-4021")
