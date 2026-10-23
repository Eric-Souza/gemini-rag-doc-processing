"""Hybrid retrieval: Postgres full-text + pgvector, RRF fusion, cross-encoder rerank."""

from app.services.retrieval.pipeline import RetrievalMode, retrieve
from app.services.retrieval.types import Candidate

__all__ = ["Candidate", "RetrievalMode", "retrieve"]
