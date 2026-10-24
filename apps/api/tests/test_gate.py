from evals.gate import _report, check

BASELINE = {
    "retrieval": {"tolerance": 0.01, "metrics": {"hit_at_1": 0.909, "mrr": 0.936}},
    "ragas": {"tolerance": 0.1, "metrics": {"faithfulness": 1.0, "context_precision": 0.905}},
}


def _current(**overrides: float) -> dict[str, dict[str, float]]:
    current = {
        "retrieval": {"hit_at_1": 0.9091, "mrr": 0.9364},
        "ragas": {"faithfulness": 1.0, "context_precision": 0.905},
    }
    for key, value in overrides.items():
        group, metric = key.split("__")
        current[group][metric] = value
    return current


def _failed(rows) -> list[str]:
    return [f"{group}.{metric}" for group, metric, _, _, ok in rows if not ok]


def test_gate_passes_at_baseline_and_within_tolerance():
    assert _failed(check(BASELINE, _current())) == []
    # RAGAS uses an LLM judge, so a small dip is noise
    assert _failed(check(BASELINE, _current(ragas__context_precision=0.85))) == []


def test_one_retrieval_case_slipping_fails_the_gate():
    # 19/22 instead of 20/22 at hit@1
    assert _failed(check(BASELINE, _current(retrieval__hit_at_1=19 / 22))) == ["retrieval.hit_at_1"]


def test_ragas_drop_beyond_tolerance_fails_the_gate():
    rows = check(BASELINE, _current(ragas__context_precision=0.755))
    assert _failed(rows) == ["ragas.context_precision"]
    assert "| ragas | context_precision | 0.905 | 0.755 | -0.150 | FAIL |" in _report(rows)
    assert "+0.000 | pass |" in _report(rows)  # rounding noise isn't shown as -0.000
