# S&P 500 Tactical Allocation

Practical Machine Learning / Deep Learning project based on the
Kaggle competition **Hull Tactical - Market Prediction**. The task is to predict
market returns and convert predictions into allocations between 0 and 2.
We will reproduce two external Kaggle solutions, improve each independently
without changing its model families or main strategy, and compare them under
the same chronological evaluation protocol.

## Current status

**Stages 1–3 completed.** The first LightGBM baseline has been reproduced on all
five development folds. Competition scoring has been verified against the official
source. The suite now has 63 passing tests.
No project hyperparameter tuning or final holdout evaluation has been performed.
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

The check command reads only `date_id` from the real CSV. Holdout returns remain
unevaluated. Solution 1 adds train-fitted preprocessing and causal temporal-feature
checks; later model stages must enforce the same principles.

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
hashes still match Stage 1. **Final holdout metrics are deliberately pending**
until configuration selection is finished; there has been no holdout prediction
or scoring and no project Optuna tuning.

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
[`docs/baseline_audit.md`](docs/baseline_audit.md). No corrections have been
implemented in Stage 1.

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
| Solution 1 baseline | Stage 3 complete; CV Adjusted Sharpe 0.464577 ± 0.704304; holdout deferred |
| Solution 1 improved | Not run; Stage 4 |
| Solution 2 baseline | Not run; Stage 5 |
| Solution 2 improved | Not run; Stage 6 |

## Setup and reproducibility

Stages 1–3 use **Python 3.12.3**. Stage 1 still needs only the standard library.
Run from the repository root:

```bash
python3 scripts/audit_project.py
```

This validates the supplied dataset and writes `results/stage1/audit.json` without
executing either notebook. Repeating the command in the same environment produces
the same audit. The local shell has `python3`; `python` is not available.

`requirements.txt` pins NumPy, pandas, SciPy, scikit-learn, pytest, and LightGBM
(4.7.0, added in Stage 3).
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
in experiment reports. XGBoost, Optuna, and plotting packages will be added when
their implementation stages begin. Solution 1 training commands are listed above.

## Testing

Run `.venv/bin/python -m pytest -q`: **63 tests passed** in Stage 3. Tests cover
official metric parity, geometric versus arithmetic conventions, penalties,
R²/Spearman/RMSE, compounding/drawdown, invalid and constant inputs, position bounds,
row-ID alignment, and nonmutation. Leakage-related checks cover chronological
order, exact one-row gaps, holdout exclusion, unchanged development data after
holdout-target perturbation, exclusion of direct return/target predictors, and
sequential label availability. Solution 1 tests additionally verify train-only
median fitting, invariance to future feature/target changes, past-only lag/rolling
statistics, all-missing training columns, the 313-feature schema, exact inherited
parameters, deterministic fitting, and batch-versus-sequential prediction parity.

The Stage 1 audit also checks dataset integrity. Syntax checks, `pip check`,
whitespace validation, and repeated artifact generation supplement the test suite.

## Repository structure

```text
baseline/                    Original external notebooks; do not modify
data/train.csv               Supplied dataset; do not modify
docs/baseline_audit.md        Source behavior, bugs, and reproduction constraints
docs/evaluation.md            Metric definitions, source, edge cases, split policy
docs/solution1.md             Baseline 1 settings, adaptations, experiment protocol
src/metrics.py                Shared competition scorer and diagnostics
src/validation.py             Development/holdout splits and availability helpers
src/solution1.py              Baseline 1 features, preprocessing, LightGBM, allocation
scripts/audit_project.py     Standard-library data and notebook audit CLI
scripts/check_evaluation.py  Synthetic reference and split audit CLI
scripts/reproduce_solution1.py  Baseline 1 development CV and separate smoke run
results/stage1/audit.json    Verified audit and saved external tuning evidence
results/stage2/               Split boundaries and synthetic verification results
results/solution1/            Baseline 1 folds, predictions, settings, preprocessing
tests/                       Metric/validation tests and official-score fixtures
requirements.txt             Dependencies required by implemented project code
codex_pmdl_master_prompt.txt  User-supplied staged project specification
.gitignore                   Caches, environments, model binaries, heavy artifacts
README.md                    Living project log
```

## Provenance and limitations

Both notebooks are external Kaggle solutions used as cited baselines. Their local
paths, metadata, execution timestamps, and hashes are recorded in the audit.
Both metadata records reference competition source ID 111543. Neither metadata nor
source cells identify an author or original notebook URL.
**TODO: obtain and verify the original authors, notebook URLs, and reuse terms.**
Do not infer attribution from filenames or fabricate links.

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
