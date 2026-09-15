"""Seeded Solution 1 search on development data; freeze selection before holdout access."""

import argparse
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import time

import numpy as np
import optuna
import pandas as pd

from scripts.reproduce_solution1 import ROOT, summarize_folds
from src.solution1_tuning import SEARCH_SPACE, SEED, PATIENCE, final_iterations, fit_fold, prepare_fold, suggest_parameters
from src.validation import development_folds


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Two trials, five folds, 64-tree ceiling; no final holdout.")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.n_trials < 1 or args.n_jobs < 1:
        parser.error("Trial count and thread count must be positive.")
    destination = args.output_dir or ROOT / "results/solution1" / ("tuning_smoke" if args.smoke else "tuning")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Use an empty output directory to preserve previous and failed experiments.")
    dataset = ROOT / "data/train.csv"
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    baseline = json.loads((ROOT / "results/solution1/baseline_summary.json").read_text())
    if digest != baseline["dataset_sha256"]:
        raise ValueError("Dataset must match the reproduced baseline.")
    dates = pd.read_csv(dataset, usecols=["date_id"]).date_id
    development = pd.read_csv(dataset, nrows=len(dates) - 180)
    folds = [prepare_fold(development, fold) for fold in development_folds(dates)]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.NopPruner())
    trial_count = 2 if args.smoke else args.n_trials
    metadata = {"status": "running", "smoke": args.smoke, "seed": SEED,
                "n_trials": trial_count, "n_folds": 5, "search_space": SEARCH_SPACE,
                "early_stopping": {"inner_rows": 180, "gap": 1, "patience": PATIENCE, "metric": "rmse"},
                "objective": "Mean outer-development Adjusted Sharpe; all five folds required.",
                "selection_rule": "Maximum mean CV Adjusted Sharpe; no threshold tuning.",
                "deployment_tree_rule": "Integer median of selected trial's five inner best iterations.",
                "dataset_sha256": digest, "development_rows": len(development),
                "holdout_accessed": False,
                "fold_boundaries": [f["bounds"] for f in folds],
                "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
                "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                    "src/metrics.py", "src/solution1.py", "src/validation.py", "src/solution1_tuning.py", "scripts/tune_solution1.py")}}
    write_json(destination / "search_manifest.json", metadata)
    trial_rows = []
    best_value = -np.inf
    started = time.perf_counter()

    def objective(trial):
        nonlocal best_value
        params = suggest_parameters(trial)
        if args.smoke:
            params["n_estimators"] = 64
        records, prediction_frames = [], []
        for number, fold in enumerate(folds, 1):
            metrics, predictions = fit_fold(fold, params, n_jobs=args.n_jobs)
            records.append({"fold": number, **metrics})
            prediction_frames.append(predictions.assign(fold=number))
            trial_rows.append({"trial": trial.number, "fold": number, **metrics})
        trial.set_user_attr("fold_metrics", records)
        values = [r["adjusted_sharpe"] for r in records]
        if any(v is None or not np.isfinite(v) for v in values):
            trial.set_user_attr("rejection_reason", "Undefined primary score in at least one fold.")
            raise optuna.TrialPruned("Undefined primary score; no NaN objective and no skipped folds.")
        value = float(np.mean(values))
        if value > best_value:
            best_value = value
            pd.concat(prediction_frames, ignore_index=True).to_csv(destination / "improved_predictions.csv", index=False)
            pd.DataFrame([{**r, "metric_errors": json.dumps(r["metric_errors"])} for r in records]).to_csv(destination / "improved_folds.csv", index=False)
        return value

    def checkpoint(study, trial):
        exported = [{"number": t.number, "state": t.state.name, "value": t.value,
                     "parameters": t.params, "attributes": t.user_attrs} for t in study.trials]
        write_json(destination / "trials.json", exported)
        study.trials_dataframe().to_csv(destination / "trials.csv", index=False)
        pd.DataFrame([{**r, "metric_errors": json.dumps(r["metric_errors"])} for r in trial_rows]).to_csv(destination / "trial_folds.csv", index=False)
        print(f"Completed trial {trial.number + 1}/{trial_count}: {trial.state.name}, objective={trial.value}", flush=True)

    try:
        study.optimize(objective, n_trials=trial_count, callbacks=[checkpoint])
        best = study.best_trial
        selected = dict(best.params)
        if args.smoke:
            selected["n_estimators"] = 64
        iterations = final_iterations([r["best_iteration"] for r in best.user_attrs["fold_metrics"]])
        frozen = {"status": "frozen", "smoke": args.smoke, "trial": best.number,
                  "parameters": selected, "final_n_estimators": iterations,
                  "threshold": 0.0, "dataset_sha256": digest,
                  "cv_metrics": summarize_folds(best.user_attrs["fold_metrics"]),
                  "holdout_accessed": False, "source_hashes": metadata["source_hashes"]}
        write_json(destination / "selected_configuration.json", frozen)
        metadata.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                        best_trial=best.number, best_value=best.value)
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(destination / "search_manifest.json", metadata)
    print(json.dumps(frozen, indent=2), flush=True)


if __name__ == "__main__":
    main()
