from dataclasses import dataclass, field


@dataclass
class Candidate:
    """A retrieved chunk plus the scores each retrieval stage gave it."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    content: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """The score from the last stage that ranked this chunk."""
        for key in ("rerank", "rrf", "vector", "keyword"):
            if key in self.scores:
                return self.scores[key]
        return 0.0
