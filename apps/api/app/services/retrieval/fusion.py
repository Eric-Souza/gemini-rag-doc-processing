from app.services.retrieval.types import Candidate


def reciprocal_rank_fusion(result_lists: list[list[Candidate]], k: int = 60) -> list[Candidate]:
    """Merge ranked lists with Reciprocal Rank Fusion: score = sum(1 / (k + rank)).

    RRF only uses ranks, so it can combine scores on different scales
    (cosine similarity and ts_rank_cd) without normalizing them.
    """
    fused: dict[str, Candidate] = {}
    for results in result_lists:
        for rank, candidate in enumerate(results, start=1):
            merged = fused.get(candidate.chunk_id)
            if merged is None:
                merged = Candidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    filename=candidate.filename,
                    chunk_index=candidate.chunk_index,
                    content=candidate.content,
                    scores={"rrf": 0.0},
                )
                fused[candidate.chunk_id] = merged
            merged.scores.update(candidate.scores)
            merged.scores["rrf"] += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda c: c.scores["rrf"], reverse=True)
