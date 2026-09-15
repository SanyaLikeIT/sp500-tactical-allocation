# S&P 500 Tactical Allocation

Practical Machine Learning / Deep Learning project based on the
Kaggle competition **Hull Tactical - Market Prediction**. The task is to predict
market returns and convert predictions into allocations between 0 and 2.
We will reproduce two external Kaggle solutions, improve each independently
without changing its model families or main strategy, and compare them under
the same chronological evaluation protocol.

## Current status

**Stages 1–2 completed: baseline audit, shared metrics, chronological validation,
and tests.** No local model training, tuning, real-return scoring, or holdout
evaluation has been performed. Competition scoring has been verified on synthetic
fixtures against the official source, with 53 passing tests.
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
unevaluated. Future model stages must enforce train-fitted preprocessing and
feature construction in addition to these split-level checks.

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
history lengths. These behaviors require explicit treatment in local reproduction.

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
| Solution 1 baseline | Not run; Stage 3 |
| Solution 1 improved | Not run; Stage 4 |
| Solution 2 baseline | Not run; Stage 5 |
| Solution 2 improved | Not run; Stage 6 |

## Setup and reproducibility

Stages 1–2 use **Python 3.12.3**. Stage 1 still needs only the standard library.
Run from the repository root:

```bash
python3 scripts/audit_project.py
```

This validates the supplied dataset and writes `results/stage1/audit.json` without
executing either notebook. Repeating the command in the same environment produces
the same audit. The local shell has `python3`; `python` is not available.

Stage 2 pins NumPy, pandas, SciPy, scikit-learn, and pytest in `requirements.txt`.
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
in the Stage 2 report. LightGBM, XGBoost, Optuna, and plotting packages will be
added when their implementation stages begin. No training commands exist yet.

## Testing

Run `.venv/bin/python -m pytest -q`: **53 tests passed** in Stage 2. Tests cover
official metric parity, geometric versus arithmetic conventions, penalties,
R²/Spearman/RMSE, compounding/drawdown, invalid and constant inputs, position bounds,
row-ID alignment, and nonmutation. Leakage-related checks cover chronological
order, exact one-row gaps, holdout exclusion, unchanged development data after
holdout-target perturbation, exclusion of direct return/target predictors, and
sequential label availability. They do not yet test model preprocessing, which
will be added with the models.

The Stage 1 audit also checks dataset integrity. Syntax checks, `pip check`,
whitespace validation, and repeated artifact generation supplement the test suite.

## Repository structure

```text
baseline/                    Original external notebooks; do not modify
data/train.csv               Supplied dataset; do not modify
docs/baseline_audit.md        Source behavior, bugs, and reproduction constraints
docs/evaluation.md            Metric definitions, source, edge cases, split policy
src/metrics.py                Shared competition scorer and diagnostics
src/validation.py             Development/holdout splits and availability helpers
scripts/audit_project.py     Standard-library data and notebook audit CLI
scripts/check_evaluation.py  Synthetic reference and split audit CLI
results/stage1/audit.json    Verified audit and saved external tuning evidence
results/stage2/               Split boundaries and synthetic verification results
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
