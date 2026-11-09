"""The Qdrant store, in-process (QDRANT_URL=:memory:), filled from Postgres the way the evals fill it."""

import pytest

from app.db import AsyncSessionLocal
from app.models import Chunk, Document, DocumentStatus
from app.services.retrieval import pipeline, qdrant_store
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
    assert await qdrant_store.rebuild() == 4
    yield ids
    async with AsyncSessionLocal() as db:
        for doc_id in ids.values():
            doc = await db.get(Document, doc_id)
            if doc:
                await db.delete(doc)
        await db.commit()
    await qdrant_store.rebuild()


async def test_dense_search_matches_pgvector_scores(doc_ids):
    hits = await qdrant_store.dense_search(axis_vector((2, 1.0)), limit=2)
    assert hits[0].content.startswith("NV-220X")
    assert hits[0].scores["vector"] == pytest.approx(1.0)
    assert hits[0].filename == "catalog.md"


async def test_bm25_finds_exact_error_code(doc_ids):
    hits = await qdrant_store.sparse_search("What should on-call do about ERR-4021?", limit=5)
    assert hits[0].content.startswith("ERR-4021")
    assert hits[0].scores["keyword"] > hits[1].scores["keyword"]  # ERR-4012 only shares 'err'


async def test_bm25_returns_nothing_for_unrelated_or_stopword_queries(doc_ids):
    assert await qdrant_store.sparse_search("quarterly marketing budget", limit=5) == []
    assert await qdrant_store.sparse_search("what is the", limit=5) == []


async def test_document_filter_limits_results(doc_ids):
    hits = await qdrant_store.dense_search(axis_vector((0, 1.0)), limit=10, document_ids=[doc_ids["catalog"]])
    assert {h.filename for h in hits} == {"catalog.md"}


async def test_hybrid_fixes_a_vector_miss(doc_ids, monkeypatch):
    # Same setup as the pgvector test: ERR-4021 is only 3rd by vector, 1st by keyword
    async def fake_embed(_question: str) -> list[float]:
        return axis_vector((0, 1.0))

    monkeypatch.setattr(pipeline, "embed_query", fake_embed)
    question = "What should on-call do about ERR-4021?"

    vector_only = await pipeline.retrieve(question, k=2, mode="vector", store="qdrant")
    hybrid = await pipeline.retrieve(question, k=2, mode="hybrid", store="qdrant")

    assert not any(c.content.startswith("ERR-4021") for c in vector_only)
    assert any(c.content.startswith("ERR-4021") for c in hybrid)
    assert "rrf" in hybrid[0].scores


async def test_hybrid_with_only_stopwords_falls_back_to_dense(doc_ids):
    hits = await qdrant_store.hybrid_search("what is the", axis_vector((2, 1.0)), limit=1, candidates=4)
    assert hits[0].content.startswith("NV-220X")


async def test_delete_document_removes_its_points(doc_ids):
    temp = qdrant_store.ChunkPoint("temp_0", "temp", "temp.md", 0, "temporary chunk", axis_vector((5, 1.0)))
    await qdrant_store.upsert_chunks([temp])
    assert (await qdrant_store.dense_search(axis_vector((5, 1.0)), limit=1))[0].chunk_id == "temp_0"

    await qdrant_store.delete_document("temp")
    assert all(h.chunk_id != "temp_0" for h in await qdrant_store.dense_search(axis_vector((5, 1.0)), limit=10))


async def test_unknown_store_is_rejected():
    with pytest.raises(ValueError, match="Unknown vector store"):
        await pipeline.retrieve("anything", store="faiss")
