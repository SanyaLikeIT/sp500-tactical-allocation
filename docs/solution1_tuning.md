# Solution 1: regularization search and frozen final evaluation

## Stage 4 hypothesis

The reproduced baseline used 6917 trees and achieved negative R² on every
development fold, with unstable trading results. Test whether lower tree
complexity, regularization, row/column sampling, and early stopping reduce
overfitting while retaining the existing single-LightGBM model, target, 313-feature
construction, and positive-prediction binary allocation policy.

No features, model families, or risk logic are added. The shared competition scorer
and Stage 3 preprocessing module remain unchanged. No threshold search is run.
The label "improved" identifies the tuned candidate; success depends on measured
results rather than its filename.

## Predeclared search

Thirty sequential Optuna TPE trials, sampler seed 42, maximize the unweighted mean
Adjusted Sharpe over the same five development folds used for Baseline 1. Each
LightGBM fit uses four CPU threads, seed 42, deterministic column-wise training.
The TPE sampler uses its default startup behavior. No performance-based pruning
or timeout reduces the configured trial budget.

| Parameter | Search range |
|---|---|
| max_depth | Integer 3–8 |
| num_leaves | Integer 8–min(128, 2**max_depth) |
| n_estimators (ceiling) | Integer 300–2000 |
| learning_rate | 0.01–0.07, logarithmic |
| min_child_samples | Integer 20–300 |
| min_split_gain | 0–0.001 |
| reg_alpha | 0.00001–0.1, logarithmic |
| reg_lambda | 0.001–10, logarithmic |
| subsample | 0.6–1.0 |
| colsample_bytree | 0.6–1.0 |

`subsample_freq=1` activates the searched row-sampling fraction. This is an
explicit regularization change from the reproduced author's final fit, which
retained the default zero. The search-space constants are in
`src/solution1_tuning.py` and recorded in `search_manifest.json`.

## Nested early stopping

For each outer development fold:

1. Reserve the last 180 **outer-training** rows as the early-stopping segment.
   Leave a one-row gap before that segment.
2. Fit an inner preprocessor on the earlier inner-training rows only. Use its
   medians for the early-stopping segment.
3. Fit LightGBM up to the sampled tree ceiling, monitoring RMSE on that inner
   segment with patience 50. Outer validation labels are not available to this fit.
4. Fit a separate preprocessor on all outer-training rows and refit the model on
   those rows using the selected tree count, with no validation callbacks.
5. Predict the outer validation segment and record all eight project metrics,
   penalties, constant-prediction status, allocation mean, and selected tree count.

Outer validation rows select hyperparameters through Optuna, so the winning CV
score is a selection score and may be optimistic. They do not directly select
early-stopping iterations. Holdout values do not enter either operation.

If a fold has an undefined Adjusted Sharpe, its trial is explicitly pruned with a
reason; undefined folds are not skipped or replaced by NaN objectives. Constant
predictions are allowed if the trading metric is defined, while undefined Spearman
is recorded as null. All trial records, including rejected trials, are preserved.

## Frozen deployment rule and holdout

The best mean-CV configuration is written to `selected_configuration.json` before
any holdout values are loaded. Its final tree count is the integer median of its
five inner early-stopping counts; this rule was defined before searching. It does
not depend on holdout performance. The file also stores dataset and source hashes.

The separate final evaluator rejects smoke/unfrozen configurations and changed
dataset/source hashes. It fits both the fixed author's baseline and the tuned
candidate through `date_id=8866`, leaves 8867 as the gap, and scores the same 180
holdout rows 8868–9047. Neither final model uses holdout early stopping. The baseline
retains 6917 estimators; the tuned model uses its frozen median tree count. Training
medians and temporal features follow exactly the same Stage 3 protocol.

The output directory must be empty, preventing accidental overwrites. Final
evaluation status is written before loading the complete CSV; completed model
results are saved individually. Do not run another search in response to holdout
results. The historical Stage 3 baseline summary remains unchanged and correctly
records that no holdout evaluation occurred during that earlier stage; the new
final-evaluation files contain the Stage 4 measurements.

## Commands

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.tune_solution1 --smoke --n-jobs 4
.venv/bin/python -m scripts.tune_solution1 --n-trials 30 --n-jobs 4
.venv/bin/python -m scripts.evaluate_solution1 --n-jobs 4
```

Search commands require empty output directories to retain previous experiments.
Use `--output-dir` for an independent repeat; final evaluation additionally accepts
`--configuration` pointing to that repeat's frozen selection. Repeating for
reproducibility must retain the same choices, not initiate further selection from
holdout results.

The smoke run is two trials on all five folds with a 64-tree ceiling, saved under
`results/solution1/tuning_smoke/`. Its Optuna sampled `n_estimators` entries describe
the proposed ceiling; smoke execution explicitly overrides them to 64, as the
selected-configuration file records. Its results are execution checks only. The
initial smoke run used LightGBM's deprecated `eval_set` adapter; the real run uses
the equivalent supported `eval_X`/`eval_y` API, verified by the tests.

## Artifacts

`results/solution1/tuning/` contains:

- `search_manifest.json`: budget, ranges, environment, boundaries, source hashes,
  status, elapsed time, and selection policy.
- `trials.json`, `trials.csv`: all trial parameters, states, objective values, and
  recorded fold diagnostics.
- `trial_folds.csv`: one row per evaluated trial/fold.
- `selected_configuration.json`: frozen best parameters, deployment tree count,
  and CV mean/sample-standard-deviation summaries.
- `improved_folds.csv`, `improved_predictions.csv`: winning trial's development
  fold metrics and all 900 out-of-fold predictions/allocations.

The smoke directory has the same artifact structure. `results/solution1/final/`
contains baseline/improved holdout summary JSONs and prediction CSVs,
`comparison.csv`, `preprocessing.json`, and `evaluation_status.json`.

## Checks and interpretation

Measured outcome: all 30 trials completed. Trial 10 was frozen with 61 final
estimators. Selected CV Adjusted Sharpe rose from 0.464577 to 1.007705, but final
holdout Adjusted Sharpe fell from baseline 0.955166 to tuned 0.431994. The proposed
improvement did not generalize on the measured holdout. All results, including
inferior trials and this unsuccessful final comparison, are retained. No follow-up
search used holdout feedback.

Tests cover depth/leaf constraints, inner gaps, median-tree selection, immutable
training/stopping inputs under outer-label perturbation, unchanged predictions,
and final-evaluation guards, in addition to the existing causality and metric
tests. Any extreme score requires inspection of splits, source hashes, target
isolation, allocation bounds, and persisted predictions before accepting it.

Compare mean and per-fold results, not only the best fold. A better selected CV
mean is not proof of generalization; final holdout results and stability must be
reported even if tuning hurts performance. The inherited author's source history
may overlap today's holdout, as documented in the Stage 3 protocol, limiting the
claim of historical independence. See the root README for measured outcomes.
