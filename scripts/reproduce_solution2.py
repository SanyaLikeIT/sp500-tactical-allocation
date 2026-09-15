"""Sequential Baseline 2 reproduction with a frozen-source diagnostic control."""

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
from scripts.tune_solution1 import write_json
from src.metrics import evaluate_predictions, strategy_returns
from src.solution2 import EnsemblePreprocessor, WINDOW, WEIGHTS, allocate, fit_ensemble, predict_ensemble, select_features
from src.solution2_search import author_search
from src.validation import TARGET, available_training_indices, development_folds, validate_dates


def training_indices(data, prediction_position):
    indices = available_training_indices(prediction_position, WINDOW)
    return indices[data.date_id.iloc[indices].to_numpy() >= 37]


def forecast_fold(development, start, length, *, fold_number=1, smoke=False, n_jobs=1, on_search=None):
    """Forecast one row at a time; current/gap labels never enter training or risk."""
    data = development.reset_index(drop=True)
    validate_dates(data.date_id)
    if start + length > len(data) or length < 1:
        raise ValueError("Prediction range must be contained in supplied development rows.")
    initial = training_indices(data, start)
    if len(initial) < 12:
        raise ValueError("Insufficient training history.")
    window = data.iloc[initial]
    processor = EnsemblePreprocessor()
    initial_matrix = processor.fit_transform(window)
    selected = select_features(initial_matrix, window[TARGET], n_jobs=n_jobs)
    seed = 42 + fold_number * 10_000
    print(f"Fold {fold_number}: startup author search (30/30/20 trials; smoke={smoke}).", flush=True)
    parameters, search = author_search(window, seed=seed, n_jobs=n_jobs, smoke=smoke)
    events = [{"offset": 0, "training_end": int(window.date_id.iloc[-1]), **search}]
    if on_search:
        on_search(events[-1])
    fitted = fit_ensemble(initial_matrix[selected], window[TARGET], parameters, n_jobs=n_jobs)
    frozen_fitted = fitted
    # This matrix uses cross-sectional inputs and causal I-column forward filling only.
    featured = processor.transform(data.iloc[initial[0]:start+length])
    v1_median = float(initial_matrix.V1.median()) if "V1" in initial_matrix else 0.
    records = {"baseline": [], "frozen_control": []}
    previous = {name: 0. for name in records}
    for offset in range(length):
        position = start + offset
        indices = training_indices(data, position)
        if offset:
            if offset % 50 == 0:
                print(f"Fold {fold_number}: author online search at offset {offset} (10 trials per model).", flush=True)
                parameters, search = author_search(data.iloc[indices], current=parameters,
                                                    seed=seed+offset*10, n_jobs=n_jobs, smoke=smoke)
                events.append({"offset": offset, "training_end": int(data.date_id.iloc[indices[-1]]), **search})
                if on_search:
                    on_search(events[-1])
            fitted = fit_ensemble(featured.loc[indices, selected], data.iloc[indices][TARGET], parameters, n_jobs=n_jobs)
        row = featured.loc[[position], selected]
        v1 = float(featured.loc[position, "V1"]) if "V1" in featured else None
        for name, models, risk_indices in [("baseline", fitted, indices), ("frozen_control", frozen_fitted, initial)]:
            prediction, components = predict_ensemble(models, row)
            allocation, volatility, multiplier = allocate(prediction, v1, v1_median,
                                                          data.iloc[risk_indices][TARGET], previous[name])
            previous[name] = allocation
            records[name].append({"date_id": int(data.date_id.iloc[position]), "prediction": prediction,
                                  "allocation": allocation, "risk_volatility": volatility,
                                  "signal_multiplier": multiplier,
                                  "training_start": int(data.date_id.iloc[risk_indices[0]]),
                                  "training_end": int(data.date_id.iloc[risk_indices[-1]]),
                                  **{f"prediction_{k}": v for k, v in components.items()}})
        if (offset + 1) % 25 == 0:
            print(f"Fold {fold_number}: {offset+1}/{length} sequential predictions completed.", flush=True)
    # Target/scoring columns are attached only after the entire forecast stream.
    evaluation = data.iloc[start:start+length]
    outputs = {}
    for name, rows in records.items():
        frame = pd.DataFrame(rows)
        frame["target"] = evaluation[TARGET].to_numpy()
        frame["forward_returns"] = evaluation.forward_returns.to_numpy()
        frame["risk_free_rate"] = evaluation.risk_free_rate.to_numpy()
        frame["strategy_return"] = strategy_returns(frame.forward_returns, frame.risk_free_rate, frame.allocation)
        outputs[name] = frame
    audit = {"fold": fold_number, "selected_features": selected, "initial_medians": processor.medians_.to_dict(),
             "initial_v1_median": v1_median, "last_parameters": parameters, "search_events": events}
    return outputs, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="First fold, 52 predictions, two trials per component/search.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("n-jobs must be positive.")
    destination = args.output_dir or ROOT / "results/solution2" / ("smoke" if args.smoke else "")
    destination.mkdir(parents=True, exist_ok=True)
    if any(p.name != "smoke" for p in destination.iterdir()):
        raise ValueError("Use an empty output directory; existing runs are never overwritten.")
    digest = hashlib.sha256((ROOT / "data/train.csv").read_bytes()).hexdigest()
    if digest != json.loads((ROOT / "results/stage1/audit.json").read_text())["dataset"]["sha256"]:
        raise ValueError("Dataset differs from audited baseline dataset.")
    dates = pd.read_csv(ROOT / "data/train.csv", usecols=["date_id"]).date_id
    data = pd.read_csv(ROOT / "data/train.csv", nrows=len(dates)-180)
    folds = development_folds(dates)
    if args.smoke:
        folds = folds[:1]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    status = {"status": "running", "smoke": args.smoke, "completed_folds": 0}
    write_json(destination / "run_status.json", status)
    collected = {name: {"frames": [], "metrics": []} for name in ["baseline", "frozen_control"]}
    audits = []
    started = time.perf_counter()
    try:
        for number, fold in enumerate(folds, 1):
            event_path = destination / f"fold_{number}_searches.jsonl"

            def checkpoint(event):
                with event_path.open("a") as handle:
                    handle.write(json.dumps(event, allow_nan=False) + "\n")

            outputs, audit = forecast_fold(data, int(fold.validation[0]), 52 if args.smoke else 180,
                                           fold_number=number, smoke=args.smoke, n_jobs=args.n_jobs, on_search=checkpoint)
            audits.append(audit)
            for name, frame in outputs.items():
                metrics = evaluate_predictions(targets=frame.target, predictions=frame.prediction,
                                               forward_returns=frame.forward_returns, risk_free_rate=frame.risk_free_rate,
                                               allocations=frame.allocation)
                collected[name]["frames"].append(frame.assign(fold=number))
                collected[name]["metrics"].append({"fold": number, "validation_start": int(frame.date_id.iloc[0]),
                                                   "validation_end": int(frame.date_id.iloc[-1]), **metrics})
                pd.concat(collected[name]["frames"], ignore_index=True).to_csv(destination / f"{name}_predictions.csv", index=False)
                pd.DataFrame([{**r, "metric_errors": json.dumps(r["metric_errors"])} for r in collected[name]["metrics"]]).to_csv(destination / f"{name}_folds.csv", index=False)
                print(f"Fold {number}, {name}: Adjusted Sharpe={metrics['adjusted_sharpe']}, R2={metrics['r2']}", flush=True)
            write_json(destination / "preprocessing_and_searches.json", audits)
            status["completed_folds"] = number
            write_json(destination / "run_status.json", status)
        common = {"status": "complete", "smoke": args.smoke, "dataset_sha256": digest,
                  "holdout": {"status": "not_evaluated_for_solution2", "date_id_start": 8868, "date_id_end": 9047},
                  "window": WINDOW, "ensemble_weights": WEIGHTS, "top_features": 50,
                  "seed_rule": "42 + 10000*fold + 10*offset + component_index",
                  "elapsed_seconds": time.perf_counter()-started,
                  "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
                  "source_hashes": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                      "src/solution2.py", "src/solution2_search.py", "scripts/reproduce_solution2.py", "src/metrics.py", "src/validation.py",
                      "baseline/hull-market-prediction-just-improved.ipynb")}}
        for name, results in collected.items():
            write_json(destination / f"{name}_summary.json", {**common, "variant": name, "cv_metrics": summarize_folds(results["metrics"]),
                "description": "Minimally repaired intended online ensemble." if name == "baseline" else
                "Diagnostic control: source's disabled online branch; same corrected startup preprocessing/search."})
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(destination / "run_status.json", status)


if __name__ == "__main__":
    main()
