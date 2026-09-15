"""Evaluate frozen Solution 1 baseline and tuned models once on the final holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reproduce_solution1 import ROOT, METRICS
from scripts.tune_solution1 import write_json
from src.metrics import evaluate_predictions, strategy_returns
from src.solution1 import TemporalPreprocessor, baseline_model, binary_allocations
from src.solution1_tuning import tuned_model
from src.validation import final_holdout_fold, TARGET


def validate_frozen(configuration, digest):
    if configuration.get("status") != "frozen" or configuration.get("smoke"):
        raise ValueError("Final evaluation requires a frozen non-smoke configuration.")
    if configuration["dataset_sha256"] != digest or configuration["holdout_accessed"]:
        raise ValueError("Dataset changed or configuration already used holdout information.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, default=ROOT / "results/solution1/tuning/selected_configuration.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/solution1/final")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    frozen = json.loads(args.configuration.read_text())
    digest = hashlib.sha256((ROOT / "data/train.csv").read_bytes()).hexdigest()
    validate_frozen(frozen, digest)
    for path, expected in frozen["source_hashes"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed after selection: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Final output directory must be empty; never overwrite a holdout evaluation.")
    status = {"status": "running", "configuration_sha256": hashlib.sha256(args.configuration.read_bytes()).hexdigest(),
              "completed_models": [], "dataset_sha256": digest,
              "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output_dir / "evaluation_status.json", status)
    try:
        data = pd.read_csv(ROOT / "data/train.csv")
        fold = final_holdout_fold(data.date_id)
        training = data.iloc[fold.train]
        processor = TemporalPreprocessor()
        X_train = processor.fit_transform(training)
        X_holdout = processor.transform(data).iloc[fold.validation]
        evaluation = data.iloc[fold.validation]
        baseline_summary = json.loads((ROOT / "results/solution1/baseline_summary.json").read_text())
        models = {"baseline": baseline_model(n_jobs=args.n_jobs),
                  "improved": tuned_model(frozen["parameters"], n_jobs=args.n_jobs,
                                           iterations=frozen["final_n_estimators"])}
        comparison = []
        for name, model in models.items():
            print(f"Final {name}: training through {training.date_id.iloc[-1]}, "
                  f"{model.n_estimators} estimators; no holdout early stopping.", flush=True)

            def progress(env):
                if (env.iteration + 1) % 1000 == 0:
                    print(f"Final {name}: {env.iteration + 1} iterations.", flush=True)

            model.fit(X_train, training[TARGET], callbacks=[progress])
            predictions = model.predict(X_holdout)
            allocations = binary_allocations(predictions)
            metrics = evaluate_predictions(targets=evaluation[TARGET], predictions=predictions,
                                           forward_returns=evaluation.forward_returns,
                                           risk_free_rate=evaluation.risk_free_rate, allocations=allocations)
            frame = evaluation[["date_id", TARGET, "forward_returns", "risk_free_rate"]].rename(columns={TARGET: "target"}).copy()
            frame["prediction"] = predictions
            frame["allocation"] = allocations
            frame["strategy_return"] = strategy_returns(evaluation.forward_returns, evaluation.risk_free_rate, allocations)
            frame.to_csv(args.output_dir / f"{name}_holdout_predictions.csv", index=False)
            summary = {"model": name, "metrics": metrics, "parameters": model.get_params(),
                       "train_end": int(training.date_id.iloc[-1]), "gap_date_id": 8867,
                       "holdout_start": int(evaluation.date_id.iloc[0]), "holdout_end": int(evaluation.date_id.iloc[-1]),
                       "actual_iterations": model.booster_.current_iteration(),
                       "configuration_sha256": status["configuration_sha256"]}
            write_json(args.output_dir / f"{name}_holdout_summary.json", summary)
            cv = baseline_summary["cv_metrics"] if name == "baseline" else frozen["cv_metrics"]
            comparison.append({"model": f"Solution 1 {name}", "cv_adjusted_sharpe_mean": cv["adjusted_sharpe"]["mean"],
                               "cv_adjusted_sharpe_std": cv["adjusted_sharpe"]["std"],
                               **{f"holdout_{key}": metrics[key] for key in METRICS}})
            status["completed_models"].append(name)
            write_json(args.output_dir / "evaluation_status.json", status)
            print(f"Final {name} metrics: {metrics}", flush=True)
        pd.DataFrame(comparison).to_csv(args.output_dir / "comparison.csv", index=False)
        write_json(args.output_dir / "preprocessing.json", processor.medians_.to_dict())
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output_dir / "evaluation_status.json", status)


if __name__ == "__main__":
    main()
