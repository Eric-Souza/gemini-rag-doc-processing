from evals.gate import _report, check

BASELINE = {
    "retrieval": {
        "tolerance": 0.01,
        "modes": {
            "hybrid": {"hit_at_1": 0.864, "mrr": 0.924},
            "hybrid_rerank": {"hit_at_1": 0.909, "mrr": 0.936},
        },
    },
    "ragas": {"tolerance": 0.1, "metrics": {"faithfulness": 1.0, "context_precision": 0.905}},
}


def _current(overrides: dict[tuple[str, str], float] | None = None) -> dict[str, dict[str, float]]:
    current = {
        "retrieval:hybrid": {"hit_at_1": 19 / 22, "mrr": 0.9242},
        "retrieval:hybrid_rerank": {"hit_at_1": 20 / 22, "mrr": 0.9364},
        "ragas": {"faithfulness": 1.0, "context_precision": 0.905},
    }
    for (group, metric), value in (overrides or {}).items():
        current[group][metric] = value
    return current


def _failed(rows) -> list[str]:
    return [f"{group}.{metric}" for group, metric, _, _, ok in rows if not ok]


def test_gate_passes_at_baseline_and_within_tolerance():
    assert _failed(check(BASELINE, _current())) == []
    # RAGAS uses an LLM judge, so a small dip is noise
    assert _failed(check(BASELINE, _current({("ragas", "context_precision"): 0.85}))) == []


def test_one_retrieval_case_slipping_fails_the_gate():
    rows = check(BASELINE, _current({("retrieval:hybrid_rerank", "hit_at_1"): 19 / 22}))
    assert _failed(rows) == ["retrieval:hybrid_rerank.hit_at_1"]


def test_a_broken_stage_fails_even_when_the_final_mode_holds():
    # Reverting keyword search to websearch_to_tsquery drops hybrid to these numbers, but the
    # reranker still fixes the final ranking on a small corpus, so only the hybrid rows catch it
    rows = check(BASELINE, _current({("retrieval:hybrid", "hit_at_1"): 18 / 22, ("retrieval:hybrid", "mrr"): 0.902}))
    assert _failed(rows) == ["retrieval:hybrid.hit_at_1", "retrieval:hybrid.mrr"]


def test_ragas_drop_beyond_tolerance_fails_the_gate():
    rows = check(BASELINE, _current({("ragas", "context_precision"): 0.755}))
    assert _failed(rows) == ["ragas.context_precision"]
    assert "| ragas | context_precision | 0.905 | 0.755 | -0.150 | FAIL |" in _report(rows)
    assert "+0.000 | pass |" in _report(rows)  # rounding noise isn't shown as -0.000
