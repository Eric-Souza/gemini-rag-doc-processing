from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.types import Candidate


def _candidate(chunk_id: str, **scores: float) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id="doc",
        filename="doc.md",
        chunk_index=0,
        content=chunk_id,
        scores=scores,
    )


def test_rrf_rewards_chunks_found_by_both_searches():
    vector = [_candidate("a", vector=0.9), _candidate("b", vector=0.8), _candidate("c", vector=0.7)]
    keyword = [_candidate("c", keyword=0.5), _candidate("d", keyword=0.4)]

    fused = reciprocal_rank_fusion([vector, keyword], k=60)

    # "c" is 3rd in vector and 1st in keyword, so it beats "a" (1st in vector only)
    assert [c.chunk_id for c in fused] == ["c", "a", "b", "d"]
    assert fused[0].scores["rrf"] == 1 / 63 + 1 / 61
    assert fused[0].scores["vector"] == 0.7
    assert fused[0].scores["keyword"] == 0.5


def test_rrf_does_not_mutate_inputs():
    vector = [_candidate("a", vector=0.9)]
    reciprocal_rank_fusion([vector, [_candidate("a", keyword=1.0)]])
    assert vector[0].scores == {"vector": 0.9}


def test_candidate_score_prefers_latest_stage():
    assert _candidate("a", vector=0.4).score == 0.4
    assert _candidate("a", vector=0.4, rrf=0.03).score == 0.03
    assert _candidate("a", vector=0.4, rrf=0.03, rerank=5.2).score == 5.2

