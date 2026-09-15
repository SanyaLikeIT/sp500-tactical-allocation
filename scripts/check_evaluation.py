"""Record the split plan and check synthetic scores against the official reference.

Run from the repository root with python -m scripts.check_evaluation.
Only date_id is read from train.csv. No real return series is scored.
"""

import argparse
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.metrics import adjusted_sharpe
from src.validation import development_folds, final_holdout_fold


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/stage2",
                        help="Directory for validation boundaries and reference check results.")
    args = parser.parse_args()
    dates = pd.read_csv(ROOT / "data/train.csv", usecols=["date_id"])["date_id"]
    folds = [(f"development_{i}", f) for i, f in enumerate(development_folds(dates), 1)]
    folds.append(("final_holdout_reserved", final_holdout_fold(dates)))
    records = []
    for name, fold in folds:
        records.append({
            "split": name, "train_start": int(dates.iloc[fold.train[0]]),
            "train_end": int(dates.iloc[fold.train[-1]]), "train_rows": len(fold.train),
            "validation_start": int(dates.iloc[fold.validation[0]]),
            "validation_end": int(dates.iloc[fold.validation[-1]]),
            "validation_rows": len(fold.validation),
            "gap_rows": int(fold.validation[0] - fold.train[-1] - 1),
        })
    reference_path = ROOT / "tests/fixtures/official_metric_cases.json"
    reference = json.loads(reference_path.read_text())
    checks = []
    for case in reference["cases"]:
        actual = adjusted_sharpe(case["forward_returns"], case["risk_free_rate"], case["allocations"])
        expected = case["adjusted_sharpe"]
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Reference mismatch for {case['name']}: {actual} vs {expected}")
        checks.append({"case": case["name"], "official_score": expected,
                       "project_score": actual, "absolute_error": abs(actual - expected)})
    report = {
        "stage": 2, "python_version": platform.python_version(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
        "source_url": reference["source_url"], "source_version": reference["version"],
        "source_cell_sha256": reference["source_cell_sha256"],
        "fixture_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "reference_checks": checks, "all_reference_checks_passed": True,
        "real_data_usage": "date_id only; holdout boundaries recorded, no real returns scored",
        "local_model_metrics": None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "validation_folds.csv", index=False)
    (args.output_dir / "evaluation_check.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Passed {len(checks)} official reference cases. Five development folds and reserved holdout recorded.")
    print(f"Largest absolute reference error: {max(c['absolute_error'] for c in checks):.3g}")
    print(f"Results written to {args.output_dir}. No model or real-return evaluation performed.")


if __name__ == "__main__":
    main()
