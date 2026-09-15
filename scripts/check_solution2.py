"""Audit persisted Solution 2 predictions, searches, metrics, and provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reproduce_solution1 import METRICS, ROOT, summarize_folds
from scripts.tune_solution1 import write_json
from src.metrics import evaluate_predictions


def check(directory):
    status = json.loads((directory / "run_status.json").read_text())
    assert status["status"] == "complete"
    smoke = status["smoke"]
    folds, length = (1, 52) if smoke else (5, 180)
    source_audit = json.loads((ROOT / "results/stage1/audit.json").read_text())
    digest = hashlib.sha256((ROOT / "data/train.csv").read_bytes()).hexdigest()
    assert digest == source_audit["dataset"]["sha256"]
    development = pd.read_csv(ROOT / "data/train.csv", nrows=8868)
    audit = json.loads((directory / "preprocessing_and_searches.json").read_text())
    assert len(audit) == folds
    counts, extreme = [], []
    for record in audit:
        assert len(record["selected_features"]) == 50
        assert not {"date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"} & set(record["selected_features"])
        offsets = [0, 50] if smoke else [0, 50, 100, 150]
        assert [e["offset"] for e in record["search_events"]] == offsets
        for event in record["search_events"]:
            start = 7968 + (record["fold"]-1)*180 + event["offset"]
            assert event["training_end"] == start-2
            for boundary in event["inner_boundaries"]:
                assert boundary["train_end"] == boundary["validation_start"]-2
                assert boundary["validation_end"] <= event["training_end"]
            for name, model in event["models"].items():
                assert all(t["state"] == "COMPLETE" for t in model["trials"])
                assert model["completed_trials"] == len(model["trials"])
                assert 0 < model["completed_trials"] <= model["requested_trials"]
                counts.append({"fold": record["fold"], "offset": event["offset"], "model": name,
                               "requested": model["requested_trials"], "completed": model["completed_trials"]})
    for variant in ["baseline", "frozen_control"]:
        summary = json.loads((directory / f"{variant}_summary.json").read_text())
        assert summary["dataset_sha256"] == digest
        assert summary["holdout"]["status"] == "not_evaluated_for_solution2"
        for path, expected in summary["source_hashes"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
        frame = pd.read_csv(directory / f"{variant}_predictions.csv")
        metrics = pd.read_csv(directory / f"{variant}_folds.csv")
        assert len(frame) == folds*length and frame.date_id.is_unique
        assert frame.allocation.between(0, 2).all()
        assert (frame.training_end <= frame.date_id-2).all()
        assert (frame.training_end-frame.training_start == 799).all()
        recalculated = []
        for number, group in frame.groupby("fold"):
            start = 7968+(number-1)*180
            np.testing.assert_array_equal(group.date_id, np.arange(start, start+length))
            expected_end = group.date_id.to_numpy()-2 if variant == "baseline" else np.full(length, start-2)
            np.testing.assert_array_equal(group.training_end, expected_end)
            rows = development.iloc[group.date_id]
            for actual, original in [("target", "market_forward_excess_returns"), ("forward_returns", "forward_returns"), ("risk_free_rate", "risk_free_rate")]:
                np.testing.assert_allclose(group[actual], rows[original], rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(group.prediction, .3*group.prediction_enet+.35*group.prediction_xgb+.35*group.prediction_lgb, atol=1e-14)
            # Independent risk/allocation recomputation from observable target slices.
            previous = 0.
            preprocessing = audit[int(number)-1]
            for row in group.itertuples():
                history = development.market_forward_excess_returns.iloc[row.training_end-19:row.training_end+1]
                v1 = None
                if "V1" in preprocessing["initial_medians"]:
                    v1 = development.V1.iloc[row.date_id]
                    if not np.isfinite(v1):
                        v1 = preprocessing["initial_medians"]["V1"]
                proxy = v1 if v1 is not None else (history.std() or .01)
                risk = max(np.sqrt(.3*np.var(history.to_numpy())+.7*proxy**2), .01)
                np.testing.assert_allclose(row.risk_volatility, risk, atol=1e-12)
                multiplier = 600. if v1 is not None and v1 < preprocessing["initial_v1_median"] else 400.
                assert row.signal_multiplier == multiplier
                signal = np.clip(row.prediction*row.signal_multiplier, 0, 2)
                allocation = (.75*np.clip(signal/(1.2*row.risk_volatility), 0, 2)+.25*previous)*.99997
                np.testing.assert_allclose(row.allocation, allocation, atol=1e-12)
                assert len(history) == 20
                previous = row.allocation
            np.testing.assert_allclose(group.strategy_return, group.risk_free_rate*(1-group.allocation)+group.forward_returns*group.allocation, atol=1e-14)
            actual = evaluate_predictions(targets=group.target, predictions=group.prediction,
                                           forward_returns=group.forward_returns, risk_free_rate=group.risk_free_rate,
                                           allocations=group.allocation)
            saved = metrics.loc[metrics.fold == number].iloc[0]
            for name in METRICS:
                if actual[name] is None:
                    assert pd.isna(saved[name])
                else:
                    np.testing.assert_allclose(actual[name], saved[name], rtol=1e-9, atol=1e-12)
            if actual["adjusted_sharpe"] is not None and abs(actual["adjusted_sharpe"]) > 3:
                extreme.append({"variant": variant, "fold": int(number), "score": actual["adjusted_sharpe"]})
            recalculated.append(actual)
        for name, values in summarize_folds(recalculated).items():
            for field in ["mean", "std"]:
                expected = summary["cv_metrics"][name][field]
                if expected is None:
                    assert values[field] is None
                else:
                    np.testing.assert_allclose(values[field], expected, rtol=1e-9, atol=1e-12)
    report = {"status": "passed", "smoke": smoke, "prediction_rows_per_variant": folds*length,
              "checks": ["source_and_dataset_hashes", "dates_and_target_alignment", "window_and_gap",
                         "inner_search_boundaries", "weighted_predictions", "risk_from_observable_labels", "allocation_recursion",
                         "strategy_returns", "fold_metrics", "summary_metrics", "holdout_exclusion"],
              "search_counts": counts, "extreme_adjusted_sharpe_folds": extreme}
    write_json(directory / "verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "search_counts"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/solution2")
    check(parser.parse_args().results_dir)
