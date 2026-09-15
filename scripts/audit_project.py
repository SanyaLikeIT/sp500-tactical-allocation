"""Audit the supplied dataset and notebook outputs without executing notebooks."""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re


ROOT = Path(__file__).resolve().parents[1]
RETURN_COLUMNS = (
    "forward_returns", "risk_free_rate", "market_forward_excess_returns"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_data(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    if not columns or not rows:
        raise ValueError("The dataset must contain a header and data rows.")
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate column names found.")
    if not {"date_id", *RETURN_COLUMNS}.issubset(columns):
        raise ValueError("Required date or return columns are missing.")
    missing = dict.fromkeys(columns, 0)
    nonfinite = dict.fromkeys(columns, 0)
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row found.")
        for column, value in row.items():
            if value.strip().lower() in {"", "nan", "na", "null"}:
                missing[column] += 1
            elif not math.isfinite(float(value)):
                nonfinite[column] += 1
    dates = [int(row["date_id"]) for row in rows]
    checks = {
        "expected_shape": (len(rows), len(columns)) == (9048, 98),
        "expected_contiguous_dates": dates == list(range(9048)),
        "complete_return_columns": all(missing[c] == 0 for c in RETURN_COLUMNS),
        "no_infinite_values": not any(nonfinite.values()),
        "unique_dates": len(set(dates)) == len(dates),
    }
    if not all(checks.values()):
        raise ValueError(f"Dataset checks failed: {checks}")
    features = [c for c in columns if c not in {"date_id", *RETURN_COLUMNS}]
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "rows": len(rows), "columns_count": len(columns), "columns": columns,
        "feature_count": len(features),
        "feature_groups": {p: sum(c.startswith(p) for c in features)
                           for p in ("D", "E", "I", "M", "P", "S", "V")},
        "target": "market_forward_excess_returns",
        "return_columns": list(RETURN_COLUMNS),
        "date_id_min": dates[0], "date_id_max": dates[-1],
        "missing_counts": missing,
        "columns_with_missing_values": sum(v > 0 for v in missing.values()),
        "duplicate_rows": len(rows) - len({tuple(r[c] for c in columns) for r in rows}),
        "checks": checks,
        "planned_development": {"rows": 8868, "date_id_start": 0, "date_id_end": 8867},
        "reserved_final_holdout": {"rows": 180, "date_id_start": 8868, "date_id_end": 9047},
        "holdout_usage": "Schema/completeness audit only; no fitting or model evaluation.",
    }


def audit_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
    best_trials = {}
    counts = {}
    saved_shapes = []
    study = "unnamed"
    trial_pattern = re.compile(r"Trial (\d+) finished with value: ([^ ]+) and parameters: (\{.*?\})\.")
    for cell_index, cell in enumerate(notebook["cells"]):
        for output in cell.get("outputs", []):
            value = output.get("text", output.get("data", {}).get("text/plain", ""))
            value = "".join(value) if isinstance(value, list) else value
            for line in value.splitlines():
                if "Shape (Rows, Columns):" in line:
                    saved_shapes.append({"cell_index": cell_index, "text": line})
                if "A new study created in memory with name:" in line:
                    study = line.split("name: ", 1)[1]
                match = trial_pattern.search(line)
                if not match:
                    continue
                metric = "mean_spearman" if path.name.startswith("hull-eda") else "mean_mse"
                score = float(match[2])
                counts[study] = counts.get(study, 0) + 1
                previous = best_trials.get(study)
                better = previous is None or (score > previous["value"] if metric == "mean_spearman"
                                              else score < previous["value"])
                if better:
                    best_trials[study] = {
                        "cell_index": cell_index, "trial": int(match[1]),
                        "metric": metric, "value": score,
                        "parameters": ast.literal_eval(match[3]), "saved_output_line": line,
                    }
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "cell_count": len(notebook["cells"]), "metadata": notebook.get("metadata", {}),
        "source_urls_in_cells": sorted(set(re.findall(r"https?://[^\s<>\"']+", sources))),
        "provenance_status": "TODO: confirm original author and notebook URL; absent from supplied metadata and source cells.",
        "saved_shapes": saved_shapes, "saved_best_trials": best_trials,
        "saved_completed_trial_lines": counts,
        "metric_status": "External saved tuning outputs only; not locally reproduced results.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage1/audit.json",
                        help="Destination for the machine-readable audit.")
    args = parser.parse_args()
    report = {
        "stage": 1, "python_version": platform.python_version(),
        "dataset": audit_data(ROOT / "data/train.csv"),
        "notebooks": [audit_notebook(p) for p in sorted((ROOT / "baseline").glob("*.ipynb"))],
        "local_model_metrics": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Dataset checks passed: 9048 rows, 98 columns, date_id 0..9047.")
    print(f"Audit written to {args.output}")
    print("Notebooks were read, not executed. No local model metrics were produced.")


if __name__ == "__main__":
    main()
