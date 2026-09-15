# Solution 1: baseline reproduction

## Scope

Stage 3 implements the external single-LightGBM solution in `src/solution1.py`
and evaluates it through `scripts/reproduce_solution1.py`. It is a
**methodologically corrected local reproduction**, not an improved model and not
a literal replay of the original notebook's contaminated CV. The notebook remains
unchanged. No new Optuna search or decision-threshold selection occurs here.

The target is the supplied `market_forward_excess_returns`. The scoring columns
`forward_returns` and `risk_free_rate` are used only after predictions have been
made. The allocation rule is `prediction > 0 -> 1`, otherwise 0. This matches the
source's scalar inference behavior; its inactive affine mapping is not used.

## Fixed model settings

These are the exact saved parameters from author trial 10 (not freshly tuned):

| Parameter | Value |
|---|---:|
| n_estimators | 6917 |
| learning_rate | 0.021484317145938295 |
| max_depth | 8 |
| num_leaves | 2662 |
| reg_lambda | 0.0072045470966688365 |
| reg_alpha | 0.0013718972580079596 |
| colsample_bytree | 0.7659097201144509 |
| subsample | 0.7373863632419442 |
| random_state | 42 |

`subsample_freq=0` is intentionally retained: the author's final fit omitted the
search's fixed `subsample_freq=1`. Thus the recorded subsample fraction does not
activate row bagging. The unusual leaves/depth combination is also preserved.
There is no early stopping because the original deployed final model was fitted
with the selected estimator count; the author's earlier hyperparameter search
used early stopping, but that search is not repeated in Stage 3.

CPU settings are `n_jobs=4`, `deterministic=True`, `force_col_wise=True`, and
`verbosity=-1`. They control repeatability, resources, and logs rather than adding
regularization. LightGBM 4.7.0 is pinned. Other model settings remain library
defaults and are recorded explicitly in each result manifest.

## Features and required methodological corrections

The current schema yields 89 retained raw market columns and 224 derived columns:
**313 total inputs**. Raw market columns with the source's exclusions are retained;
IDs and realized target/scoring columns are never passed into feature engineering.

The source's fourteen selected columns, six lags (1, 3, 5, 7, 14, 20), and five
mean/std windows (2, 5, 10, 20, 60) are unchanged. Rolling calculations retain
`min_periods=1` and sample standard deviation (`ddof=1`).

| Original behavior | Local reproduction | Reason |
|---|---|---|
| Rolling features include current observation | Shift raw series by one row before rolling | Follow the project's strictly past temporal-feature rule |
| Whole-dataset median filling before CV | Compute fallback medians on training rows only | Exclude validation/holdout information |
| Recompute medians on an 85-row inference slice | Reuse fixed training medians | Keep preprocessing fitted on training only |
| Truncate inference history to 85 rows | Use full causal feature history | Preserve lag/rolling and forward-fill state consistently |
| Four broad folds without a gap | Five 180-row development folds with a one-row gap | Use the shared protocol and reserve the final holdout |
| Kaggle paths, serialized models, inference server | Local CPU fits and prediction CSVs | Run without Kaggle-only files/runtime |

Derived features are built from raw, unfilled series, as in the source. The
resulting raw and engineered matrix is forward-filled, then remaining missing
values use the training medians (zero if the training column is entirely missing).
Raw contemporaneous features are available at prediction time and remain inputs;
the past-only restriction applies to derived temporal features. The unlabeled gap
row can supply observed features but cannot supply training targets.

No model updates occur within a validation period for Solution 1. We compute the
causal validation feature matrix in a batch and predict with a frozen model.
Tests verify equivalence to processing chronological prefixes one at a time,
including missing observations. Future feature suffixes and validation targets
cannot change earlier inputs or predictions. Filling medians over the training
portion is permitted; no claim is made that fitting a model reproduces the online
information set of each historical training observation.

## Commands and artifacts

Run from the repository root:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.reproduce_solution1 --smoke --n-jobs 4
.venv/bin/python -m scripts.reproduce_solution1 --n-jobs 4
```

The smoke run uses the full first fold with only 32 estimators. It verifies the
execution path, not baseline quality; it writes to `results/solution1/smoke/`.
The real experiment fits five models with the full 6917-estimator configuration.
No trial budget is reduced in the real experiment.

An independent smoke repeat in `/tmp/pmdl-solution1-smoke-repeat` produced
byte-identical prediction CSVs, fold metrics, and preprocessing medians. The full
test suite passed 63 tests, including separate deterministic fitting checks.

Each output directory contains:

- `baseline_summary.json`: parameters, feature names, CV means/sample standard
  deviations, source/data hashes, environment, methodology, and holdout status.
- `baseline_folds.csv`: per-fold metrics, date boundaries, actual boosting
  iteration counts, and mean allocations.
- `baseline_predictions.csv`: out-of-fold predictions, allocations, targets,
  scoring inputs, and strategy returns, aligned by `date_id` and fold.
- `baseline_preprocessing.json`: per-fold training median values for auditing.
- `run_status.json`: whether the run is running, failed, or complete, and how
  many folds have completed. Exceptions are recorded before propagating.

`--output-dir` supports an independent repeat; smoke and full runs may not share
the same destination. Rerunning a completed configuration overwrites its files;
use a separate directory when retaining experiment variants. Inspect the run
status before using partial artifacts. Runtime can differ across repeats; numeric
result comparisons should exclude the elapsed-time field.

CV statistics weight each fold equally and use sample standard deviation
(`ddof=1`). If a metric is undefined in any fold, its mean/std remain null with
valid/total fold counts; invalid folds are never silently discarded. Mean
cumulative return and mean maximum drawdown summarize separate 180-day periods,
not a compounded 900-day portfolio. Per-row artifacts support separate pooled
analysis later without changing the CV objective.

## Holdout and provenance limitations

The script reads all date IDs to establish the split, then loads only the first
8868 rows' values. It hashes the complete CSV to verify the supplied dataset;
hashing is an integrity check, not model access to holdout returns. The last 180
rows are neither fitted, predicted, nor scored. Baseline and improved holdout
evaluation is deferred until the configuration-selection stage is complete.

The saved author mean Spearman (0.06156511055277297) came from a different dataset
snapshot and validation/preprocessing protocol; it is not directly comparable to
our current CV mean. The inherited author configuration was selected on a
9021-row source dataset. Its history may overlap today's holdout, although exact
overlapping row values have not been verified. Therefore the holdout is untouched
by **local selection**, but complete independence from the external author's
historical choices cannot be claimed.

Future improvements must use this same corrected feature protocol and shared
metric implementation to make a baseline-versus-improved comparison meaningful.
