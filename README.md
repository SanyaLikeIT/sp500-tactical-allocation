# S&P 500 Tactical Allocation

Practical Machine Learning / Deep Learning project based on the
Kaggle competition **Hull Tactical - Market Prediction**. The task is to predict
market returns and convert predictions into allocations between 0 and 2.
We will reproduce two external Kaggle solutions, improve each independently
without changing its model families or main strategy, and compare them under
the same chronological evaluation protocol.

## Current status

**Stages 1–5 completed.** Solution 1's 30-trial development-only Optuna search
improved the selected CV score but **underperformed the baseline on final holdout**.
The failed improvement is retained. Solution 2's corrected online ensemble has
completed five development folds: Adjusted Sharpe **0.216599 ± 1.319487**.
Its frozen-model diagnostic control scored **0.318871 ± 0.729500**.
Competition scoring has been verified against
the official source; 79 tests pass. No holdout values selected configurations, and
no further search was run after final evaluation.
The original notebooks and dataset are preserved unchanged.

## Dataset

The checked local file is `data/train.csv`:

| Property | Verified value |
|---|---|
| Rows / columns | 9,048 / 98 |
| Ordered, unique, consecutive `date_id` | 0 through 9,047 |
| Candidate market features | 94, in groups D, E, I, M, P, S, V |
| Prediction target in both notebooks | `market_forward_excess_returns` |
| Realized market return for scoring | `forward_returns` |
| Cash return for scoring | `risk_free_rate` |
| Columns containing missing values | 85 |
| Missing values in the three return columns | 0 |
| Duplicate rows | 0 |

Return/scoring columns must not be passed directly as contemporaneous predictors.
The target is used as supplied; it is not silently replaced by
`forward_returns - risk_free_rate`.

Baseline 1's saved output (cell 4, zero-based) reports **9,021 rows and 98 columns**.
Our file has 27 more rows. This verifies a shape difference, not identical values
throughout the overlapping history. Published or saved scores cannot substitute
for reproduction on our dataset. Dataset and notebook SHA-256 hashes, column
names, missing counts, and saved tuning evidence are in
[`results/stage1/audit.json`](results/stage1/audit.json).

## Shared evaluation

- Primary model-selection metric: competition-style **Adjusted Sharpe**, higher
  is better. `src/metrics.py` implements the formula from
  [Kaggle's official metric notebook, version 3](https://www.kaggle.com/code/metric/hull-competition-sharpe),
  retrieved on 2026-09-15. It uses geometric excess returns, sample standard
  deviation of total returns, volatility/underperformance penalties, and the
  official upper score cap. Full definitions, units, and source provenance are in
  [`docs/evaluation.md`](docs/evaluation.md).
- Diagnostics: raw Sharpe, R², Spearman, RMSE, annualized strategy volatility,
  cumulative return, and maximum drawdown. R² is interpreted normally; do not
  optimize its absolute value toward zero.
- Development data: first **8,868 rows**, `date_id` 0–8867.
- Reserved final holdout: last **180 rows**, `date_id` 8868–9047. Stage 1 checks
  schema/completeness only; it does not use holdout performance or target
  distributions to make model decisions.
- Implemented development CV: `TimeSeriesSplit(n_splits=5, test_size=180, gap=1)`.
  No random splits. Fit preprocessing and feature selection inside training folds.
- Features must respect observation availability. Temporal features will use past
  observations; any necessary change from source behavior will be recorded.
- Freeze configuration selection before final evaluation. Holdout data must never
  select hyperparameters, weights, thresholds, history length, or risk parameters.
  Sequential retraining may use only labels already observable at prediction time;
  holdout-triggered hyperparameter searches must be disabled.

The reported raw Sharpe is the competition's geometric Sharpe before penalties.
R² uses its standard definition; Spearman measures ranking quality; RMSE measures
prediction error. Volatility and cumulative return are fractions, and maximum
drawdown is a nonnegative peak-to-trough loss including initial equity. Zero
volatility makes the strict score undefined and raises an error; diagnostic
reports use JSON `null` with reasons. Constant predictions produce undefined
Spearman, not a fabricated zero correlation. Invalid allocations are rejected;
explicit policy clipping is a separate helper.

Stage 2 checks and reproducible artifacts:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.check_evaluation
```

- [`results/stage2/validation_folds.csv`](results/stage2/validation_folds.csv):
  five development folds plus reserved holdout boundaries; every split has a
  one-row gap. First validation: 7968–8147; last development validation: 8688–8867.
- [`results/stage2/evaluation_check.json`](results/stage2/evaluation_check.json):
  eight synthetic comparisons against official reference scores, all passed;
  maximum absolute difference **6.58e-13**, tolerance `rtol=atol=1e-10`.
  This file also records the resolved package versions and source/fixture hashes.
- [`tests/fixtures/official_metric_cases.json`](tests/fixtures/official_metric_cases.json):
  fixed synthetic inputs and expected scores generated by the inspected official
  source. These are testing evidence, not model-performance measurements.

The check command reads only `date_id` from the real CSV. Solution 1 adds
train-fitted preprocessing, causal temporal-feature checks, and separate frozen
holdout evaluation; later model stages must enforce the same principles.

## External Baseline 1: LightGBM with temporal features

Source: [`baseline/hull-eda-training-pipeline.ipynb`](baseline/hull-eda-training-pipeline.ipynb),
configuration/feature/training/inference cells 8–12 (zero-based).

- One LightGBM regressor predicts `market_forward_excess_returns`.
- Fourteen selected columns receive lag and rolling features:
  `M4, V13, S5, S2, D2, E19, P7, P6, P3, P13, P4, P5, M2, V5`.
- Lags: 1, 3, 5, 7, 14, 20; mean and standard-deviation windows:
  2, 5, 10, 20, 60. This creates 224 temporal columns in addition to 89 retained
  original market features (313 model inputs on the current schema).
- Original search: 200 trials / 5,400-second timeout, four chronological folds,
  mean Spearman objective, RMSE early stopping with patience 200. Model seed is
  42; the Optuna sampler is not seeded. Only 12 completed-trial lines are saved.
- The saved best settings include 6,917 estimators, learning rate
  0.021484317145938295, depth 8, and 2,662 leaves. The full settings are recorded
  in the audit JSON. Final training uses the selected estimator count without
  early stopping and does not retain the search's fixed `subsample_freq=1`.
- Although the signal helper describes 0/1/2 outputs, inference passes one scalar:
  its 75th percentile equals itself, yielding allocation 1 for a positive
  prediction and 0 otherwise. `BEST_C=0.5` belongs to commented-out logic.

The source computes medians before CV and uses current-row observations in rolling
statistics. Training and inference also calculate fallback medians on different
history lengths. The local reproduction explicitly corrects these behaviors.

### Local reproduction protocol

[`src/solution1.py`](src/solution1.py) preserves the author's saved final-fit
parameters and binary allocation rule. Rolling features are shifted one row,
medians are fitted only on training data and reused at inference, and full causal
history preserves forward-fill state. The same 313-feature protocol will be used
for subsequent baseline-versus-improved comparisons. This is a **methodologically
corrected reproduction**, not a literal replay of the source's original CV.

There are five independent fits, each with 6917 estimators, depth 8, 2662 leaves,
learning rate 0.021484317145938295, and seed 42. Other author parameters, every
local adaptation, and artifact definitions are documented in
[`docs/solution1.md`](docs/solution1.md). No early stopping or new search is used:
this follows the author's final deployed fit rather than rerunning its earlier
Optuna search. The final fit's `subsample_freq=0` default is preserved.

```bash
.venv/bin/python -m scripts.reproduce_solution1 --smoke --n-jobs 4
.venv/bin/python -m scripts.reproduce_solution1 --n-jobs 4
```

The smoke command checks the first fold with 32 trees and saves its results
separately under `results/solution1/smoke/`. It is not a measured baseline result.
The full command uses all five development folds and the full estimator count.
It loads only development rows' feature/target values, checks the original data
hash, and does not evaluate holdout. Model predictions within each fold are
batched using causal features; tests verify equivalence to sequential prefixes.

### Reproduced Baseline 1 results

Dataset: the checked 9048-row CSV, with 8868 development rows and 900 out-of-fold
predictions across five 180-row windows. Four CPU threads; LightGBM 4.7.0; all five
models completed **6917 boosting iterations**. Full CV took **415.1 seconds** in
this environment. The commands above reproduce the experiment.

| Metric | CV mean | CV sample std |
|---|---:|---:|
| Adjusted Sharpe (primary) | 0.464577 | 0.704304 |
| Raw geometric Sharpe | 0.676633 | 1.155523 |
| R² | -0.088330 | 0.033789 |
| Spearman | 0.052373 | 0.054372 |
| RMSE | 0.010629 | 0.003187 |
| Annualized strategy volatility | 0.118188 | 0.041677 |
| Cumulative return per fold | 0.072467 | 0.071948 |
| Maximum drawdown per fold | 0.072963 | 0.035470 |

| Fold | Validation date_id | Adjusted Sharpe | R² | Spearman |
|---|---|---:|---:|---:|
| 1 | 7968–8147 | 0.262498 | -0.097467 | 0.044538 |
| 2 | 8148–8327 | 0.440280 | -0.033492 | 0.073292 |
| 3 | 8328–8507 | 0.113218 | -0.081107 | 0.079710 |
| 4 | 8508–8687 | 1.662404 | -0.118204 | 0.102026 |
| 5 | 8688–8867 | -0.155513 | -0.111382 | -0.037703 |

The mean primary score is positive but unstable across periods, with a substantial
contribution from fold 4 and a negative result on fold 5. R² is negative in every
fold: squared prediction error exceeds the evaluation-window mean-prediction
benchmark. Rank correlation is modestly positive on average. This establishes the
baseline; it is not evidence of a robust improvement or live trading performance.
No fold exceeds an absolute Adjusted Sharpe of 3.

All return/volatility/drawdown entries are fractions (0.118188 volatility means
11.8188%). Mean cumulative return and drawdown summarize separate 180-row folds;
they are not a compounded 900-day return or drawdown. The author's saved Spearman
0.061565 remains an external result under a different dataset/protocol.

Machine-readable artifacts:

- [`baseline_summary.json`](results/solution1/baseline_summary.json): all metrics,
  exact parameters, environment, feature names, source hashes, holdout status.
- [`baseline_folds.csv`](results/solution1/baseline_folds.csv): per-fold diagnostics.
- [`baseline_predictions.csv`](results/solution1/baseline_predictions.csv): all
  900 predictions, allocations, targets, and strategy returns for auditing.
- [`baseline_preprocessing.json`](results/solution1/baseline_preprocessing.json):
  the medians fitted separately on each training fold.
- [`run_status.json`](results/solution1/run_status.json): completed, five of five
  folds. The smoke directory contains the same five artifact types separately.

All saved metrics were recomputed from the persisted predictions and matched
within numerical tolerance. The independent smoke repeat produced byte-identical
prediction, fold-metric, and preprocessing files. Dataset and original notebook
hashes still match Stage 1. Stage 3 deliberately deferred final holdout evaluation.
Its historical summary remains unchanged; Stage 4 evaluates the baseline and
tuned candidate after freezing selection, using separate final-result files.

## Tuned Solution 1

Hypothesis: reduce overfitting using tree-complexity controls, L1/L2 regularization,
row/column sampling, and early stopping while preserving LightGBM, the target,
all 313 features, and the zero-threshold binary allocation rule.

The search completed **30 of 30 trials** (150 outer fold evaluations, with one
inner stopping fit and one full-training refit each) in **74.5 seconds** after
feature preparation. Optuna 5.0.0 uses a seeded TPE sampler (42), and each model
uses four CPU threads. The objective is mean development-fold Adjusted Sharpe;
R², Spearman, and RMSE are diagnostics, not selection criteria.

The final 180 rows never enter tuning. Early stopping uses a separate trailing
180-row segment inside each outer training fold, separated by a one-row gap.
Inner and outer preprocessing are fitted separately. The outer model is then
refitted on all allowed training rows at the inner-selected tree count. The
winning CV score is a selection score, not an independent generalization estimate.

All search ranges, stopping rules, and artifacts are described in
[`docs/solution1_tuning.md`](docs/solution1_tuning.md). Ranges include depth 3–8,
leaves 8–min(128, 2**depth), estimator ceilings 300–2000, L1 1e-5–0.1, and L2
0.001–10. Row sampling is explicitly enabled with `subsample_freq=1`.

Selected **trial 10**:

| Parameter | Selected value |
|---|---:|
| max_depth / num_leaves | 3 / 8 |
| sampled n_estimators ceiling | 724 |
| inner best iterations, folds 1–5 | 118, 61, 222, 2, 3 |
| final n_estimators (predeclared median rule) | 61 |
| learning_rate | 0.028502386445081143 |
| min_child_samples | 150 |
| min_split_gain | 0.0008325076947404105 |
| reg_alpha | 0.000014209969789341111 |
| reg_lambda | 4.071570699987008 |
| subsample | 0.7200142681676905 |
| colsample_bytree | 0.7877444618791577 |

| Development fold | Baseline Adjusted Sharpe | Tuned Adjusted Sharpe |
|---|---:|---:|
| 1 | 0.262498 | 0.660478 |
| 2 | 0.440280 | 0.890079 |
| 3 | 0.113218 | 1.266677 |
| 4 | 1.662404 | 2.253616 |
| 5 | -0.155513 | -0.032323 |
| Mean | 0.464577 | 1.007705 |
| Sample std | 0.704304 | 0.841936 |

The selected candidate improves all five development-fold primary scores, but
dispersion remains high and the last fold is still negative. Tuned CV mean R² is
0.000614, Spearman 0.070162, and RMSE 0.010213. CV maximum drawdown averages 0.075720,
slightly worse than baseline's 0.072963. No evaluated trial/fold exceeds an
absolute Adjusted Sharpe of 3. No threshold calibration or additional search was
performed after selection.

```bash
.venv/bin/python -m scripts.tune_solution1 --smoke --n-jobs 4
.venv/bin/python -m scripts.tune_solution1 --n-trials 30 --n-jobs 4
.venv/bin/python -m scripts.evaluate_solution1 --n-jobs 4
```

The smoke search uses two trials and a 64-tree ceiling on all five folds, saved
separately in `results/solution1/tuning_smoke/`. Full-search evidence is under
[`results/solution1/tuning/`](results/solution1/tuning/): `search_manifest.json`,
`trials.json`, `trials.csv`, `trial_folds.csv`, `improved_folds.csv`,
`improved_predictions.csv`, and `selected_configuration.json`. The last file is
written and source-hashed before final evaluation. Undefined primary scores are
explicitly rejected; undefined correlations remain null diagnostics. All 30 real
trials completed successfully; unsuccessful parameter choices remain in the logs.

These commands refuse to overwrite nonempty result directories. For an independent
repeat, pass a fresh `--output-dir`; the final evaluator accepts `--configuration`
pointing to that repeat's selection file. Do not use repeats to tune on holdout.

### Final Solution 1 comparison

Both final models trained through `date_id=8866`, left 8867 as a gap, and were
evaluated once on the same 180 holdout rows (8868–9047). The tuned model's 61-tree
configuration was frozen first; baseline used its original 6917-tree configuration.
All metrics below the CV rows refer exclusively to this final holdout.

| Metric | Baseline | Tuned candidate |
|---|---:|---:|
| CV Adjusted Sharpe mean | 0.464577 | 1.007705 |
| CV Adjusted Sharpe sample std | 0.704304 | 0.841936 |
| Holdout Adjusted Sharpe | **0.955166** | **0.431994** |
| Holdout raw geometric Sharpe | 1.238748 | 0.747911 |
| Holdout R² | -0.016038 | -0.012326 |
| Holdout Spearman | 0.143319 | 0.145287 |
| Holdout RMSE | 0.010395 | 0.010376 |
| Holdout annualized strategy volatility | 0.104760 | 0.132026 |
| Holdout cumulative return | 0.129365 | 0.104626 |
| Holdout maximum drawdown | 0.086132 | 0.111552 |

Tuning improved the selected development scores and slightly improved holdout
regression diagnostics, but **did not improve holdout trading performance**.
Holdout return fell from 12.94% to 10.46%, maximum drawdown increased from 8.61%
to 11.16%, and the primary score declined. The baseline performed better on this
held-out period. Possible explanations include selection optimism on reused CV
folds and temporal instability; this experiment does not identify a causal
explanation. Neither the improved CV score nor one short holdout proves robustness.
No new search or threshold adjustment was performed after seeing these outcomes.

Final artifacts under [`results/solution1/final/`](results/solution1/final/):

- [`comparison.csv`](results/solution1/final/comparison.csv): the complete comparison.
- `baseline_holdout_summary.json`, `improved_holdout_summary.json`: measured
  metrics, parameters, training/holdout boundaries, and frozen-selection hash.
- `baseline_holdout_predictions.csv`, `improved_holdout_predictions.csv`:
  180 predictions, allocations, targets, and strategy returns per model.
- `preprocessing.json`: shared final training medians.
- `evaluation_status.json`: completed status and frozen-selection/source hashes.

All final metrics were independently recomputed from saved predictions. The
selection/source hashes still match, the shared scorer and Stage 3 feature
implementation are unchanged, and original notebook/data hashes match Stage 1.

## External Baseline 2: ensemble with volatility-aware allocation

Source: [`baseline/hull-market-prediction-just-improved.ipynb`](baseline/hull-market-prediction-just-improved.ipynb),
cell 0. The word "improved" is part of the external filename; this is our second
baseline, not a measured project improvement.

- Last 800 training rows after filtering `date_id >= 37`; exclude columns with
  more than 50% missing values.
- Market features plus five cross-sectional features: `U1`, `U2`, `V1_S1`,
  `M11_V1`, `I9_S1`. Several temporal features are computed in the source but
  removed by its final column selection, so they are not model inputs.
- StandardScaler and XGBoost importance selection (`n_estimators=100`, seed 42)
  choose 50 features. ElasticNet, XGBoost, and LightGBM then form the ensemble
  with fixed weights **0.30 / 0.35 / 0.35**.
- Active startup tuning minimizes mean MSE on five time-series folds: XGBoost
  30 trials / 300 seconds; LightGBM 30 / 300; ElasticNet 20 / 100. This replaces
  the dataclass's fallback model defaults. Samplers are unseeded.
- Volatility uses a 20-row return history and V1; signal multipliers are 600/400
  depending on V1 versus its training median, with volatility scaling 1.2.
  Position clipping is followed by 0.75/0.25 allocation smoothing and a factor
  of `1 - 0.00003`.
- Intended online behavior: refit each row and retune every 50 rows using three
  inner folds and 10 trials / 15 seconds per model. **The actual saved code
  disables that branch by dropping its trigger column from stored state.**

The source also selects features and imputes data before its inner CV. Further
online-update errors and the minimum reproduction requirements are documented in
[`docs/baseline_audit.md`](docs/baseline_audit.md).

### Local Solution 2 reproduction

[`docs/solution2.md`](docs/solution2.md) specifies all source settings, search
ranges, seeds, local repairs, and artifact definitions. The primary `baseline`
restores the intended daily rolling-window fits and searches at prediction offsets
50, 100, and 150. At date `t`, training and volatility history end at `t-2`.
Initial medians, missingness filters, top-50 selection, and the V1 median are fixed
per fold; the scaler and three regressors refit daily on 800 available rows.
Each outer fold resets model, preprocessing, search, and allocation state.

The `frozen_control` uses the same initial models and corrected preprocessing,
but keeps model/scaler/target history fixed. It diagnoses the source's disabled
online branch; it is neither a literal notebook replay nor an improved model.
Both variants preserve the three model families, five actual derived features,
original weights, clipping, volatility scaling, and smoothing.

Necessary adaptations are explicit: remove backward filling, fit preprocessing
and supervised selection separately inside each inner fold, enforce one-row gaps,
repair online label/date alignment and feature schemas, retain causal I-column
history, and guard undefined ratios/missing interaction inputs. Only available
market features are predictors. The source's discarded temporal columns remain
absent. Tree searches still use unscaled inputs while final fits use scaled inputs,
matching the source's discrepancy. No new Stage 6 tuning has been applied.

Startup reproduces the author's MSE searches with 30/30/20 trials; each online
event allows 10 trials per model. Source timeouts are preserved and actual counts
are saved. Local TPE seeds use `42 + 10000*fold + 10*offset + component_index`;
model seed is 42. Runs use one CPU thread and XGBoost CPU 3.4.1.

```bash
.venv/bin/python -m scripts.reproduce_solution2 --smoke --n-jobs 1
.venv/bin/python -m scripts.reproduce_solution2 --n-jobs 1
.venv/bin/python -m scripts.check_solution2
```

Smoke uses 52 sequential rows and two trials per component/search, including the
offset-50 update. It completed in 11.9 seconds; results are saved separately in
`results/solution2/smoke/` and are not baseline estimates. The full experiment
uses the same development intervals as Solution 1: five 180-row folds, 900
predictions per variant, from the audited 9048-row dataset. Earlier validation
labels can enter later rolling fits only after the required observation delay.
Output directories cannot be overwritten; pass a fresh `--output-dir` to repeat.

Results live under [`results/solution2/`](results/solution2/):
`baseline_summary.json`, `baseline_folds.csv`, `baseline_predictions.csv`, matching
`frozen_control_*` files, `preprocessing_and_searches.json`, per-fold
`fold_N_searches.jsonl` checkpoints, `run_status.json`, and `verification.json`.
They include component predictions, training boundaries, risk estimates, medians,
selected features, trial results, source hashes, and installed package versions.
The verification command independently recomputes risk, positions, returns, and
shared metrics from saved records and checks split and holdout boundaries.

Solution 2 holdout evaluation is deferred until Stage 6 choices are frozen.
Solution 1's holdout outcome is already known, but does not select settings here;
the final period cannot be described as blind to the project as a whole.

### Reproduced Baseline 2 results

The full run took **328.4 seconds**, completing all **850/850 trials** across
60 component studies (five startup events and fifteen online events). No timeout
reduced a search budget. This required 3350 inner model fits, 2700 sequential
ensemble-component fits, and 75 feature-selector fits. The frozen control reuses
each initial fitted ensemble. Both variants have 900 saved predictions.

| Metric | Online baseline CV mean ± sample std | Frozen control CV mean ± sample std |
|---|---:|---:|
| Adjusted Sharpe (primary) | 0.216599 ± 1.319487 | 0.318871 ± 0.729500 |
| Raw geometric Sharpe | -0.182850 ± 1.666914 | 0.424916 ± 0.936483 |
| R² | -0.006521 ± 0.003502 | -0.005261 ± 0.005379 |
| Spearman | -0.096924 ± 0.071170 | Undefined across all five folds |
| RMSE | 0.010258 ± 0.003240 | 0.010248 ± 0.003226 |
| Annualized strategy volatility | 0.130549 ± 0.033728 | 0.138780 ± 0.044956 |
| Cumulative return per fold | 0.014771 ± 0.170726 | 0.070315 ± 0.090738 |
| Maximum drawdown per fold | 0.127424 ± 0.062804 | 0.111500 ± 0.041065 |

| Fold | Validation date_id | Online Adjusted Sharpe | Frozen Adjusted Sharpe |
|---|---|---:|---:|
| 1 | 7968–8147 | -0.575268 | -0.395544 |
| 2 | 8148–8327 | -0.813503 | -0.047199 |
| 3 | 8328–8507 | 0.208955 | 0.411529 |
| 4 | 8508–8687 | 2.473460 | 1.515564 |
| 5 | 8688–8867 | -0.210648 | 0.110005 |

The online ensemble has three negative folds, negative R² and Spearman in every
fold, and a positive mean primary score dominated by fold 4. Restoring updates
does not improve mean Adjusted Sharpe over the frozen diagnostic control here.
The predeclared online variant remains our baseline; the control is not selected
as a replacement based on these scores. No successful model improvement is claimed.

Frozen predictions are constant in folds 1–3, so their Spearman values are null;
only two of five folds have a defined correlation. The overall Spearman mean/std
therefore remain null instead of dropping undefined folds. Position sizing can
still vary with V1 and smoothing even when the model prediction is constant.
Raw and adjusted Sharpe means aggregate separately after per-fold penalties;
penalties also reduce the magnitude of negative scores in the official formula.

Volatility, return and drawdown are fractions. Fold averages are not a compounded
900-day result. No fold exceeds absolute Adjusted Sharpe 3. The artifact audit
passed for both variants, and an independent smoke repeat produced byte-identical
predictions, fold metrics, preprocessing and search logs (six files recorded in
`results/solution2/smoke/repeat_check.json`). Original notebook/data hashes match
Stage 1; shared metric/validation code and Solution 1 results remain unchanged.
Solution 2 holdout performance and improvements have not been evaluated yet.

## Saved author outputs versus local results

These are extracted **external tuning outputs**, not reproduced scores or final
holdout performance. They use different objectives/protocols and must not be
compared directly as trading results.

| External saved run | Metric | Value |
|---|---|---:|
| Baseline 1, best saved trial 10 | Four-fold mean Spearman | 0.06156511055277297 |
| Baseline 2, XGBoost best saved trial 1 | Five-fold mean MSE | 0.0000757019534636102 |
| Baseline 2, LightGBM best saved trial 7 | Five-fold mean MSE | 0.00007568567810074348 |
| Baseline 2, ElasticNet best saved trial 0 | Five-fold mean MSE | 0.00007570195345492456 |

No Adjusted Sharpe or leaderboard score is reported in the inspected notebook text
outputs. Full saved best parameters and supporting log lines are in the audit JSON.

| Local experiment | Status |
|---|---|
| Solution 1 baseline | Complete; CV 0.464577 ± 0.704304; holdout Adjusted Sharpe 0.955166 |
| Solution 1 improved | Complete; CV 1.007705 ± 0.841936; holdout Adjusted Sharpe 0.431994; improvement not confirmed |
| Solution 2 baseline | Complete; CV 0.216599 ± 1.319487; holdout deferred |
| Solution 2 improved | Not run; Stage 6 |

## Setup and reproducibility

Stages 1–5 use **Python 3.12.3**. Stage 1 still needs only the standard library.
Run from the repository root:

```bash
python3 scripts/audit_project.py
```

This validates the supplied dataset and writes `results/stage1/audit.json` without
executing either notebook. Repeating the command in the same environment produces
the same audit. The local shell has `python3`; `python` is not available.

`requirements.txt` pins NumPy, pandas, SciPy, scikit-learn, pytest, LightGBM 4.7.0,
Optuna 5.0.0 (added in Stage 4), and XGBoost CPU 3.4.1 (added in Stage 5).
The local `.venv` was successfully created with the installed `virtualenv` command:

```bash
virtualenv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The packages were initially installed by name, then pinned to the validated
versions; installing the resulting requirements file was also checked.
`python3 -m venv .venv` failed locally because `ensurepip` is missing; `virtualenv`
provides the working alternative. On machines with standard venv support,
`python3 -m venv .venv` can be used to create the environment instead.
Dependency download requires network access; tests subsequently run offline.
Only direct dependencies are pinned; all resolved transitive versions are recorded
in experiment reports. Plotting packages will be added at their implementation
stage. Solution 1 and Solution 2 training commands are listed above.

## Testing

Run `.venv/bin/python -m pytest -q`: **79 tests passed** in Stage 5. Tests cover
official metric parity, geometric versus arithmetic conventions, penalties,
R²/Spearman/RMSE, compounding/drawdown, invalid and constant inputs, position bounds,
row-ID alignment, and nonmutation. Leakage-related checks cover chronological
order, exact one-row gaps, holdout exclusion, unchanged development data after
holdout-target perturbation, exclusion of direct return/target predictors, and
sequential label availability. Solution 1 tests additionally verify train-only
median fitting, invariance to future feature/target changes, past-only lag/rolling
statistics, all-missing training columns, the 313-feature schema, exact inherited
parameters, deterministic fitting, and batch-versus-sequential prediction parity.
Stage 4 tests add depth/leaf constraints, inner stopping gaps, outer-target
perturbation without changes to stopping or predictions, the median-tree rule,
and guards against evaluating smoke/unfrozen configurations on final holdout.
Stage 5 adds exact ensemble inputs, safe ratios, train-fitted missingness/medians,
causal forward filling, the source risk formula, rolling-window boundaries,
future-target/feature perturbation with real ensemble models, and online-search
scheduling. Saved-artifact verification covers both adaptive and frozen variants.

The Stage 1 audit also checks dataset integrity. Syntax checks, `pip check`,
whitespace validation, and repeated artifact generation supplement the test suite.

## Repository structure

```text
baseline/                    Original external notebooks; do not modify
data/train.csv               Supplied dataset; do not modify
docs/baseline_audit.md        Source behavior, bugs, and reproduction constraints
docs/evaluation.md            Metric definitions, source, edge cases, split policy
docs/solution1.md             Baseline 1 settings, adaptations, experiment protocol
docs/solution1_tuning.md      Search ranges, nested stopping, frozen final evaluation
docs/solution2.md             Ensemble settings, online repairs, author search protocol
src/metrics.py                Shared competition scorer and diagnostics
src/validation.py             Development/holdout splits and availability helpers
src/solution1.py              Baseline 1 features, preprocessing, LightGBM, allocation
src/solution1_tuning.py       Regularized LightGBM and nested stopping helpers
src/solution2.py              Ensemble preprocessing, models, risk, and allocation
src/solution2_search.py       Nested reproduction of original MSE searches
scripts/audit_project.py     Standard-library data and notebook audit CLI
scripts/check_evaluation.py  Synthetic reference and split audit CLI
scripts/reproduce_solution1.py  Baseline 1 development CV and separate smoke run
scripts/tune_solution1.py     Development-only Optuna search and frozen selection
scripts/evaluate_solution1.py  Final holdout comparison after configuration freeze
scripts/reproduce_solution2.py  Sequential ensemble CV and frozen diagnostic control
scripts/check_solution2.py    Saved predictions, searches, risk, and metric audit
results/stage1/audit.json    Verified audit and saved external tuning evidence
results/stage2/               Split boundaries and synthetic verification results
results/solution1/            Baseline 1 folds, predictions, settings, preprocessing
results/solution2/            Ensemble CV, search logs, preprocessing, and smoke run
tests/                       Metric/validation tests and official-score fixtures
requirements.txt             Dependencies required by implemented project code
codex_pmdl_master_prompt.txt  User-supplied staged project specification
.gitignore                   Caches, environments, model binaries, heavy artifacts
README.md                    Living project log
```

## Provenance and limitations

Both notebooks are external Kaggle solutions used as cited baselines. Their local
paths, metadata, execution timestamps, and hashes are recorded in the audit.
Both metadata records reference competition source ID 111543.

The Kaggle inference runtime, `test.csv`, and serialized model/parameter artifacts
are absent. The shared scorer now follows the verified public metric version 3;
this is not an audit of all live competition rules. Baseline findings are based on
local source inspection, not execution of those two notebooks. Source defects,
unseeded searches, dataset differences,
and future library-version differences limit exact reproduction. Methodological
corrections must be distinguished from model improvements in subsequent stages.
The inherited LightGBM settings were selected on an external 9021-row history,
which may overlap today's holdout. Holdout is untouched by local selection;
complete independence from the external author's choices cannot be asserted.
