"""Quality gate: compare the latest eval results with evals/baseline.json.

Fails (exit 1) when a metric drops more than its tolerance below the baseline:

  retrieval   hit@1 and MRR from results/latest.json (python -m evals.run_evals --retrieval-only)
  ragas       mean RAGAS scores from results/ragas_latest.json (python -m evals.ragas.run_ragas)

Retrieval is deterministic, so its tolerance only absorbs float noise: one case falling from
#1 to #2 (1/22 of hit@1) fails the gate. RAGAS uses an LLM judge, so it gets more slack.

Usage (from apps/api):

    python -m evals.gate                    # check
    python -m evals.gate --update-baseline  # accept the latest results as the new baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
BASELINE_PATH = EVALS_DIR / "baseline.json"
RETRIEVAL_RESULTS = EVALS_DIR / "results" / "latest.json"
RAGAS_RESULTS = EVALS_DIR / "results" / "ragas_latest.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{path.name} not found. Run the evals first (see python -m evals.gate --help).")
    return json.loads(path.read_text(encoding="utf-8"))


def current_values(baseline: dict[str, Any]) -> dict[str, dict[str, float]]:
    retrieval = _load(RETRIEVAL_RESULTS)
    mode = baseline["retrieval"]["mode"]
    by_mode = {r["mode"]: r for r in retrieval["retrieval"]}
    if mode not in by_mode:
        raise SystemExit(f"latest.json has no retrieval results for mode {mode!r}.")

    ragas = _load(RAGAS_RESULTS)
    expected = baseline["ragas"]
    if (ragas["subset"], ragas["cases_scored"], ragas["mode"]) != (
        expected["subset"],
        expected["cases_scored"],
        expected["mode"],
    ):
        raise SystemExit(
            f"ragas_latest.json scored {ragas['cases_scored']} {ragas['subset']} cases in {ragas['mode']} mode; "
            f"the baseline uses {expected['cases_scored']} {expected['subset']} cases in {expected['mode']} mode. "
            "Rerun python -m evals.ragas.run_ragas without --limit."
        )
    return {
        "retrieval": {m: by_mode[mode][m] for m in baseline["retrieval"]["metrics"]},
        "ragas": {m: ragas["metrics"][m] for m in baseline["ragas"]["metrics"]},
    }


def check(baseline: dict[str, Any], current: dict[str, dict[str, float]]) -> list[tuple[str, str, float, float, bool]]:
    rows = []
    for group in ("retrieval", "ragas"):
        tolerance = baseline[group]["tolerance"]
        for metric, expected in baseline[group]["metrics"].items():
            value = current[group][metric]
            rows.append((group, metric, expected, value, value >= expected - tolerance))
    return rows


def _report(rows: list[tuple[str, str, float, float, bool]]) -> str:
    lines = ["| Group | Metric | Baseline | Current | Change | |", "|---|---|---|---|---|---|"]
    for group, metric, expected, value, ok in rows:
        change = round(value - expected, 3) + 0.0  # + 0.0 turns -0.0 into 0.0
        lines.append(
            f"| {group} | {metric} | {expected:.3f} | {value:.3f} | {change:+.3f} | {'pass' if ok else 'FAIL'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-baseline", action="store_true", help="write the latest results into baseline.json")
    args = parser.parse_args()

    baseline = _load(BASELINE_PATH)
    current = current_values(baseline)

    if args.update_baseline:
        for group in ("retrieval", "ragas"):
            baseline[group]["metrics"] = {m: round(v, 3) for m, v in current[group].items()}
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {BASELINE_PATH.name}")
        return

    rows = check(baseline, current)
    report = _report(rows)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Eval quality gate\n\n" + report + "\n")

    failed = [f"{group}.{metric}" for group, metric, _, _, ok in rows if not ok]
    if failed:
        print(f"\nQuality gate failed: {', '.join(failed)} dropped below the baseline.")
        sys.exit(1)
    print("\nQuality gate passed.")


if __name__ == "__main__":
    main()
