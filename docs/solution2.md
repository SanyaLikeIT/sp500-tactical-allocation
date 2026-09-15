# Baseline 2 reproduction protocol

Source: `baseline/hull-market-prediction-just-improved.ipynb`, cell 0.
The original notebook remains unchanged. This is a corrected reproduction of
its ensemble, not the project's Stage 6 improvement search.

## Features and models

Keep the most recent 800 available rows with `date_id >= 37`. At forecast date
`t`, target observations stop at `t-2`, preserving the one-row gap. Missingness
filtering (at most 50% missing), imputation medians, the V1 median, and the selected
feature list are fitted on each outer fold's initial training window and then
fixed. Each daily fit updates the scaler and three regressors on the rolling
800-row window. Independent outer folds reset all state, including allocation.

Use D/E/I/M/P/S/V market inputs and exactly five source-derived columns:
`U1=I2-I1`, `U2=M11/((I2+I9+I7)/3)`, `V1_S1=V1*S1`, `M11_V1=M11*V1`,
and `I9_S1=I9*S1`. They are calculated before imputing base inputs. The source's
temporal and target rolling columns are discarded by its final selection and
are deliberately absent here. Return columns and date IDs are not predictors.

I-columns are forward-filled from causal history, followed by train-fitted
medians. Backward filling is removed. A zero U2 denominator or infinity is treated
as missing. Missing interaction inputs produce a zero placeholder to preserve
schema; the source would fail when selecting a missing derived column. Fully
missing retained derived columns fall back to zero. Future feature rows cannot
alter an earlier transformed row; vectorized preprocessing is prefix-equivalent.

An XGBoost selector with 100 estimators and seed 42 ranks standardized inputs;
keep its top 50 (stable input-order tie breaking). Refit StandardScaler on these
inputs and fit ElasticNet, XGBoost, and LightGBM. Weights remain 0.30/0.35/0.35.
All model seeds are 42 where supported. LightGBM additionally uses deterministic
column-wise construction; runs use one CPU thread. `xgboost-cpu==3.4.1` supplies
the `xgboost` module. No GPU or Kaggle runtime is required.

## Reproduced author searches

The active notebook searches override its dataclass defaults; the local baseline
therefore repeats the searches instead of treating fallback values as fitted
parameters. These searches minimize mean MSE, as in the source. Stage 6 will
separately select improvements by Adjusted Sharpe. No outer-fold trading metric
selects settings in Stage 5.

| Component | Startup ranges | Trials / timeout |
|---|---|---|
| XGBoost | depth 3–8; learning rate 0.01–0.1 log; trees 150–350; L1/L2 0.001–10 log | 30 / 300 s |
| LightGBM | same, except trees 150–300 | 30 / 300 s |
| ElasticNet | alpha 0.001–0.1 log; l1_ratio 0.1–0.9; max_iter 1,000,000 | 20 / 100 s |

Startup uses five inner chronological folds. Every 50 prediction rows (offsets
50, 100, 150), repeat each component's search on the current window with three
inner folds, 10 trials and a 15-second timeout per component. Online ranges are
centered on the current settings: depth ±1 within 3–8; learning rate ×0.8–1.2;
trees ±50 within 100–400 for XGBoost or 100–350 for LightGBM; L1/L2 ×0.5–2;
ElasticNet alpha ×0.5–2 and l1_ratio ±0.1 within 0.1–0.9. Online sampling is linear.

Every inner split has a one-row gap. Missingness filtering, imputation, feature
selection, and scaling are fitted afresh on its training portion. This removes
the source's pre-CV imputation and supervised-selection leakage. As in the source,
tree searches use unscaled inputs; ElasticNet uses inner-fitted scaling, and all
three final models use scaled inputs. The discrepancy is retained and disclosed.

The source samplers are unseeded. Local TPE seeds are
`42 + 10000*fold_number + 10*prediction_offset + component_index`, with components
ordered XGBoost, LightGBM, ElasticNet. All trials, inner boundaries, selected
columns, best parameters, requested/completed counts, and timeouts are saved.
Timeouts can make the completed trial count hardware-dependent; no smaller
budget is silently substituted. Final trained binaries are not versioned.

## Risk and allocation

Use the most recent 20 available target observations and current processed V1:

```text
volatility = max(sqrt(0.3 * population_variance(target_history) + 0.7 * V1**2), 0.01)
multiplier = 600 if V1 < initial_training_V1_median else 400
signal = clip(weighted_prediction * multiplier, 0, 2)
position = clip(signal / (volatility * 1.2), 0, 2)
allocation = (0.75 * position + 0.25 * previous_allocation) * (1 - 0.00003)
```

When V1 is unavailable, its proxy is the recent sample target standard deviation,
or 0.01. Initial previous allocation is zero. The final factor is position
shrinkage, not an explicit turnover-based transaction cost. Shared metrics use
actual scoring returns, with no additional cost model.

## Online repairs and diagnostic control

The saved notebook drops the lagged-target column required to activate its online
branch. Restoring the trigger alone would still overwrite labels with nulls,
misalign dates, and mix feature schemas. The local **baseline** restores the
intended daily refits and 50-row searches using explicitly aligned observations
through `t-2`, a consistent selected-column schema, and finite unmodified targets.

The **frozen_control** keeps the same initial models, scaler, and risk-target
history throughout a fold. It measures the consequence of the disabled online
branch while sharing the corrected startup search/preprocessing. It still uses
current features and recursive allocation smoothing. It is a diagnostic control,
not a literal replay of the flawed notebook and not an improved model. Both
variants are predeclared; we do not choose one based on the observed scores.

The source processes a single test row at a time and can fall back to medians
instead of retaining I-column forward-fill history. Local inference consistently
uses causal history for training and prediction; this is another explicit
preprocessing adaptation. The original source behavior is audited separately in
[baseline_audit.md](baseline_audit.md).

## Evaluation, reproduction, and artifacts

Use the unchanged audited 9048-row dataset. Only the first 8868 rows' feature and
target values are loaded. Five outer validation intervals of 180 rows start at
7968, 8148, 8328, 8508, and 8688. At each step earlier validation targets may enter
training only after the prescribed delay: this evaluates an adaptive sequential
strategy, not one fixed fit per validation fold.

```bash
.venv/bin/python -m scripts.reproduce_solution2 --smoke --n-jobs 1
.venv/bin/python -m scripts.reproduce_solution2 --n-jobs 1
.venv/bin/python -m scripts.check_solution2
```

Smoke uses the first fold's first 52 rows and two trials per component per search;
tree ranges remain unchanged. It exercises the offset-50 update and writes to
`results/solution2/smoke/`. Smoke scores are not baseline performance estimates.
The full experiment writes to `results/solution2/`. Use a fresh `--output-dir`
for an independent repeat; existing runs are not overwritten.

For each variant, `*_summary.json` contains eight CV means/sample standard
deviations, source hashes, package versions and holdout status; `*_folds.csv`
contains per-fold scores; `*_predictions.csv` contains 900 predictions, component
predictions, allocations, risk estimates, training boundaries and realized
returns. `preprocessing_and_searches.json` records medians, selected columns and
all searches. `fold_N_searches.jsonl` checkpoints every completed search;
`run_status.json` records completion or failure. The smoke folder uses the same
artifact layout with one fold and two searches.

`verification.json` records the independent saved-artifact audit: hashes, exact
dates, observable target windows, inner gaps, ensemble arithmetic, risk estimates,
allocation recurrence, strategy returns, and recomputed shared metrics. Pass
`--results-dir results/solution2/smoke` to check the smoke artifacts instead.

Solution 2 holdout is deferred until Stage 6 freezes its choices. No holdout
hyperparameter searches will be permitted. Solution 1's already reported holdout
outcome is known; it does not select any settings here. The same dates therefore
cannot be claimed to be blind to the project as a whole.

Metrics and measured execution details are reported in the root README. Returns,
volatility, and drawdown are fractions; averaging independent folds does not
produce a compounded 900-day return. No claim of improvement or live trading
performance follows from this reproduction.
