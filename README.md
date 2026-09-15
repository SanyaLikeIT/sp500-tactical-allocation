# S&P 500 Tactical Allocation

Practical Machine Learning / Deep Learning project based on the
Kaggle competition **Hull Tactical - Market Prediction**. The task is to predict
market returns and convert predictions into allocations between 0 and 2.
We will reproduce two external Kaggle solutions, improve each independently
without changing its model families or main strategy, and compare them under
the same chronological evaluation protocol.

## Current status

**Stage 1: project setup and baseline audit completed.** No local model training,
tuning, competition scoring, or holdout evaluation has been performed.
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

## Evaluation plan (implementation starts in Stage 2)

- Primary model-selection metric: competition-style **Adjusted Sharpe**, higher
  is better. Neither supplied notebook contains the competition scorer. Its exact
  public formula and source must be verified before implementation.
- Diagnostics: raw Sharpe, R², Spearman, RMSE, annualized strategy volatility,
  cumulative return, and maximum drawdown. R² is interpreted normally; do not
  optimize its absolute value toward zero.
- Development data: first **8,868 rows**, `date_id` 0–8867.
- Reserved final holdout: last **180 rows**, `date_id` 8868–9047. Stage 1 checks
  schema/completeness only; it does not use holdout performance or target
  distributions to make model decisions.
- Planned development CV: `TimeSeriesSplit(n_splits=5, test_size=180, gap=1)`.
  No random splits. Fit preprocessing and feature selection inside training folds.
- Features must respect observation availability. Temporal features will use past
  observations; any necessary change from source behavior will be recorded.
- Freeze configuration selection before final evaluation. Holdout data must never
  select hyperparameters, weights, thresholds, history length, or risk parameters.
  Sequential retraining may use only labels already observable at prediction time;
  holdout-triggered hyperparameter searches must be disabled.

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

Stage 1 was run with **Python 3.12.3**, using only the standard library. Run from
the repository root:

```bash
python3 scripts/audit_project.py
```

This validates the supplied dataset and writes `results/stage1/audit.json` without
executing either notebook. Repeating the command in the same environment produces
the same audit. The local shell has `python3`; `python` is not available.

`requirements.txt` intentionally has no third-party packages at this stage.
NumPy, pandas, SciPy, scikit-learn, LightGBM, XGBoost, Optuna, plotting libraries,
and pytest are not installed in the inspected interpreter. Required dependencies
will be added and pinned when their code is introduced. No model environment or
training commands have been validated yet. A future isolated environment can be
created with `python3 -m venv .venv`; this setup command has not been run in Stage 1.

## Testing

Stage 1 runs executable dataset assertions in the audit script and checks Python
syntax, deterministic audit output, whitespace errors, and unchanged source/data
hashes against Git. No model or metric tests exist yet. Stage 2 will introduce
tests for allocation bounds, chronological order, the one-day gap, holdout
exclusion, and metric edge cases. A test-suite command will be added when the
suite exists.

## Repository structure

```text
baseline/                    Original external notebooks; do not modify
data/train.csv               Supplied dataset; do not modify
docs/baseline_audit.md        Source behavior, bugs, and reproduction constraints
src/                         Future shared modules and separate model solutions
scripts/audit_project.py     Standard-library data and notebook audit CLI
results/stage1/audit.json    Verified audit and saved external tuning evidence
tests/                       Reserved for Stage 2 evaluation tests
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
are absent. This stage does not verify live competition rules or implement their
formula. The recorded findings are based on local source inspection, not execution
of the Kaggle notebooks. Source defects, unseeded searches, dataset differences,
and future library-version differences limit exact reproduction. Methodological
corrections must be distinguished from model improvements in subsequent stages.
