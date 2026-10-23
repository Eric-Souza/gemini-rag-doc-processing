from evals.run_evals import evidence_rank, score_contains
from tests.test_fusion import _candidate


def test_evidence_rank_ignores_markdown_and_whitespace():
    hits = [_candidate("Shipping is free."), _candidate("Refunds within **30\n days** of delivery.")]
    assert evidence_rank(hits, "30 days") == 2
    assert evidence_rank(hits, "60 days") is None


def test_score_contains_accepts_any_phrase():
    assert score_contains("The deductible is $1,000.", ["1000", "$1,000"])
    assert not score_contains("The deductible is $2,000.", ["1000", "$1,000"])
