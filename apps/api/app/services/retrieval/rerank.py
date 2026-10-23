import asyncio
from functools import lru_cache

from app.config import get_settings
from app.services.retrieval.types import Candidate


@lru_cache
def _get_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # depends on the optional "rerank" extra
        raise RuntimeError(
            'Reranking needs the "rerank" extra: pip install -e ".[rerank]"'
        ) from exc
    return CrossEncoder(get_settings().reranker_model)


def warm_up_reranker() -> None:
    """Load the model ahead of the first query (the first load downloads it)."""
    _get_cross_encoder()


def _score(question: str, candidates: list[Candidate]) -> list[float]:
    model = _get_cross_encoder()
    pairs = [(question, c.content) for c in candidates]
    return [float(s) for s in model.predict(pairs)]


async def rerank(question: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    """Score (question, chunk) pairs with a local cross-encoder and keep the best top_k."""
    if not candidates:
        return []
    # Model inference is CPU-bound; keep it off the event loop.
    scores = await asyncio.to_thread(_score, question, candidates)
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.scores["rerank"] = score
    ranked = sorted(candidates, key=lambda c: c.scores["rerank"], reverse=True)
    return ranked[:top_k]
