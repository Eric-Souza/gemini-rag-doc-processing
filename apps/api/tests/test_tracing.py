"""Langfuse tracing: a no-op without keys, and nested spans per retrieval stage with them."""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.services import tracing
from app.services.rag import _run_config
from app.services.retrieval import pipeline
from tests.conftest import axis_vector
from tests.test_retrieval_db import doc_ids  # noqa: F401  (fixture)


def test_tracing_is_a_noop_without_keys(monkeypatch):
    monkeypatch.setattr(tracing, "get_langfuse", lambda: None)

    with tracing.trace("request", tags=["mode:rag"]) as root, tracing.observe("step") as span:
        assert span.update(output="ignored") is span
        root.update(output="ignored")
    assert tracing.langchain_callbacks() == []
    assert _run_config("rag")["callbacks"] == []
    tracing.flush()


@pytest.fixture
def spans(monkeypatch):
    """Route Langfuse spans to an in-memory exporter instead of a server."""
    from langfuse import Langfuse

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://localhost:1",
        span_exporter=exporter,
    )
    monkeypatch.setattr(tracing, "get_langfuse", lambda: client)
    yield client, exporter
    client.shutdown()


async def test_retrieval_stages_are_nested_spans(doc_ids, monkeypatch, spans):  # noqa: F811
    async def fake_embed(_question: str) -> list[float]:
        return axis_vector((0, 1.0))

    monkeypatch.setattr(pipeline, "embed_query", fake_embed)
    with tracing.trace("doc-qa-rag", tags=["mode:rag"], metadata={"mode": "rag"}):
        await pipeline.retrieve("What should on-call do about ERR-4021?", k=2, mode="hybrid")

    client, exporter = spans
    client.flush()
    finished = {s.name: s for s in exporter.get_finished_spans()}
    assert set(finished) == {"doc-qa-rag", "retrieve", "embed_query", "vector_search", "keyword_search", "rrf"}
    root = finished["doc-qa-rag"]
    assert finished["retrieve"].parent.span_id == root.context.span_id
    for stage in ("embed_query", "vector_search", "keyword_search", "rrf"):
        assert finished[stage].parent.span_id == finished["retrieve"].context.span_id
        assert finished[stage].context.trace_id == root.context.trace_id
    assert finished["retrieve"].attributes["langfuse.observation.type"] == "retriever"
    assert "ERR-4021" in finished["retrieve"].attributes["langfuse.observation.output"]
