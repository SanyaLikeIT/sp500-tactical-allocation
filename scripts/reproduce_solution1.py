"""Reproduce Baseline 1 on development CV only; final holdout is never scored."""

import argparse
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from src.metrics import evaluate_predictions, strategy_returns
from src.solution1 import AUTHOR_PARAMETERS, TemporalPreprocessor, baseline_model, binary_allocations
from src.validation import HOLDOUT_SIZE, TARGET, development_folds


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("adjusted_sharpe", "raw_sharpe", "r2", "spearman", "rmse",
           "strategy_volatility", "cumulative_return", "max_drawdown")


def summarize_folds(records: list[dict]) -> dict:
    """Use sample std across folds; never silently omit an undefined fold."""
    summary = {}
    for metric in METRICS:
        values = [r[metric] for r in records]
        count = sum(v is not None and np.isfinite(v) for v in values)
        summary[metric] = {
            "mean": float(np.mean(values)) if count == len(values) else None,
            "std": float(np.std(values, ddof=1)) if count == len(values) and count > 1 else None,
            "valid_folds": int(count), "total_folds": len(records),
        }
    return summary


def run(*, smoke: bool = False, n_jobs: int = 4, output_dir: Path | None = None) -> dict:
    data_path = ROOT / "data/train.csv"
    original_audit = json.loads((ROOT / "results/stage1/audit.json").read_text())
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if dataset_hash != original_audit["dataset"]["sha256"]:
        raise ValueError("Dataset hash differs from Stage 1; audit the change before reproduction.")
    dates = pd.read_csv(data_path, usecols=["date_id"])["date_id"]
    folds = development_folds(dates)
    # Only development rows' feature and target values are loaded into pandas.
    development = pd.read_csv(data_path, nrows=len(dates) - HOLDOUT_SIZE)
    if smoke:
        folds = folds[:1]
    destination = output_dir or ROOT / "results/solution1" / ("smoke" if smoke else "")
    destination.mkdir(parents=True, exist_ok=True)
    # Prevent a smoke invocation from overwriting the measured baseline by accident.
    existing = destination / "baseline_summary.json"
    if existing.exists() and json.loads(existing.read_text())["smoke"] != smoke:
        raise ValueError("Smoke and full experiments require separate output directories.")
    records, prediction_frames, preprocessing = [], [], []
    start_time = time.perf_counter()
    params = baseline_model(n_jobs=n_jobs, smoke=smoke).get_params()
    progress = {"status": "running", "smoke": smoke, "completed_folds": 0,
                "expected_folds": len(folds), "model_parameters": params}
    progress_path = destination / "run_status.json"
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
    try:
        for index, fold in enumerate(folds, 1):
            fold_start = time.perf_counter()
            training = development.iloc[fold.train]
            history = development.iloc[:fold.validation[-1] + 1]
            preprocessor = TemporalPreprocessor()
            X_train = preprocessor.fit_transform(training)
            X_validation = preprocessor.transform(history).iloc[fold.validation]
            y_train = training[TARGET].to_numpy()
            if not np.isfinite(y_train).all():
                raise ValueError("Training targets must be finite; targets are never imputed.")
            model = baseline_model(n_jobs=n_jobs, smoke=smoke)
            print(f"Fold {index}/{len(folds)}: {X_train.shape[0]} training rows, "
                  f"{X_train.shape[1]} features, {model.n_estimators} estimators.", flush=True)

            def report_iteration(env):
                if (env.iteration + 1) % 1000 == 0:
                    print(f"Fold {index}: completed {env.iteration + 1} boosting iterations.", flush=True)

            model.fit(X_train, y_train, callbacks=[report_iteration])
            predictions = model.predict(X_validation)
            allocations = binary_allocations(predictions)
            evaluation = development.iloc[fold.validation]
            metrics = evaluate_predictions(
                targets=evaluation[TARGET], predictions=predictions,
                forward_returns=evaluation["forward_returns"], risk_free_rate=evaluation["risk_free_rate"],
                allocations=allocations,
            )
            record = {"fold": index, "train_start": int(training.date_id.iloc[0]),
                      "train_end": int(training.date_id.iloc[-1]), "train_rows": len(training),
                      "validation_start": int(evaluation.date_id.iloc[0]),
                      "validation_end": int(evaluation.date_id.iloc[-1]),
                      "validation_rows": len(evaluation), "feature_count": X_train.shape[1],
                      "boosting_iterations": model.booster_.current_iteration(),
                      "allocation_mean": float(allocations.mean()), **metrics}
            records.append(record)
            prediction_frames.append(pd.DataFrame({
                "fold": index, "date_id": evaluation.date_id.to_numpy(),
                "target": evaluation[TARGET].to_numpy(), "prediction": predictions,
                "allocation": allocations, "forward_returns": evaluation.forward_returns.to_numpy(),
                "risk_free_rate": evaluation.risk_free_rate.to_numpy(),
                "strategy_return": strategy_returns(evaluation.forward_returns, evaluation.risk_free_rate, allocations),
            }))
            preprocessing.append({"fold": index, "fit_start": preprocessor.start_date_,
                                  "fit_end": preprocessor.train_end_date_,
                                  "medians": preprocessor.medians_.to_dict()})
            serializable_records = [{**r, "metric_errors": json.dumps(r["metric_errors"])} for r in records]
            pd.DataFrame(serializable_records).to_csv(destination / "baseline_folds.csv", index=False)
            pd.concat(prediction_frames, ignore_index=True).to_csv(destination / "baseline_predictions.csv", index=False)
            progress["completed_folds"] = index
            progress_path.write_text(json.dumps(progress, indent=2) + "\n")
            print(f"Fold {index} done in {time.perf_counter() - fold_start:.1f}s: "
                  f"Adjusted Sharpe={metrics['adjusted_sharpe']}, R2={metrics['r2']}, "
                  f"Spearman={metrics['spearman']}", flush=True)
    except Exception as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        raise
    result = {
        "experiment": "solution1_baseline_smoke" if smoke else "solution1_baseline",
        "smoke": smoke, "status": "complete", "dataset_sha256": dataset_hash,
        "dataset_rows": len(dates), "development_rows": len(development),
        "validation": {"n_splits": len(folds), "test_size": 180, "gap": 1, "std_ddof": 1},
        "holdout": {"status": "not_evaluated", "date_id_start": int(dates.iloc[-180]),
                    "date_id_end": int(dates.iloc[-1]),
                    "reason": "Reserved until baseline and improved configurations are frozen."},
        "model_parameters": params, "author_parameters": AUTHOR_PARAMETERS,
        "author_saved_mean_spearman": 0.06156511055277297,
        "feature_names": preprocessor.columns_, "cv_metrics": summarize_folds(records),
        "methodological_corrections": [
            "Training-only fallback medians, reused at inference; no global or inference-window medians.",
            "Rolling features shifted one observation; original notebook includes current row.",
            "Five development folds with 180 validation rows and gap 1; final holdout excluded.",
            "Full causal feature history replaces the notebook's 85-row inference slice.",
        ],
        "fit_policy": "Author final-fit parameters, fixed estimator count, no early stopping or tuning.",
        "prediction_policy": "Fixed model within each fold; causal feature matrices allow equivalent batched prediction.",
        "extreme_sharpe_folds": [r["fold"] for r in records if r["adjusted_sharpe"] is not None and abs(r["adjusted_sharpe"]) > 3],
        "elapsed_seconds": time.perf_counter() - start_time,
        "python_version": platform.python_version(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
            "src/solution1.py", "src/metrics.py", "src/validation.py", "scripts/reproduce_solution1.py",
            "baseline/hull-eda-training-pipeline.ipynb")},
        "command": ".venv/bin/python -m scripts.reproduce_solution1" + (" --smoke" if smoke else "") + f" --n-jobs {n_jobs}",
    }
    (destination / "baseline_preprocessing.json").write_text(json.dumps(preprocessing, indent=2, allow_nan=False) + "\n")
    existing.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    progress.update(status="complete")
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
    print(json.dumps(result["cv_metrics"], indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Check one fold with 32 estimators; not a baseline result.")
    parser.add_argument("--n-jobs", type=int, default=4, help="CPU threads per LightGBM model (default: 4).")
    parser.add_argument("--output-dir", type=Path, help="Optional separate destination for experiment artifacts.")
    args = parser.parse_args()
    run(smoke=args.smoke, n_jobs=args.n_jobs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
