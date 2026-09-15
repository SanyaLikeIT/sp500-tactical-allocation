# Baseline source audit

Stage 1 static inspection. Cell references are zero-based. Original notebooks are
preserved byte-for-byte; no training or reproduction has been run. Machine-readable
metadata, hashes, and saved metrics are in `results/stage1/audit.json`.

## Baseline 1

Source: `baseline/hull-eda-training-pipeline.ipynb`, cells 8–12.

### Exact feature and fit behavior

`COLS_TO_DROP` contains `forward_returns`, `risk_free_rate`, `excess_return`,
`E7`, `V10`, `S3`, `M1`, and `M14`; `excess_return` is absent from our CSV.
The target and `date_id` are excluded from model inputs. The 14 feature-engineering
columns and window settings are listed in README. Rolling operations use
`min_periods=1`; pandas rolling standard deviation uses its default sample
definition. There is no shift before rolling means or standard deviations.

Feature creation forward-fills the complete frame, then fills remaining missing
values using medians (or zero if a median is unavailable). During training this
happens before the four-fold split. Consequently validation observations affect
imputation of leading training gaps. Target filling would also be unsafe if target
values were missing; they are complete in the supplied file.

The original four-fold splitter has neither an explicit test size nor a gap. Its
Optuna objective maximizes mean Spearman while early stopping monitors RMSE.
The selected `num_leaves=2662` exceeds `2**max_depth=256`; record this author setting
without silently imposing improved-search constraints on the baseline.
Final fitting passes `study.best_params` and seed 42 only, so fixed search options
including `subsample_freq=1` are not retained. No early stopping is used in final
fitting. The `.cbm` filename contains a joblib-serialized LightGBM model, not CatBoost.

Inference appends the current feature row to history and recomputes features on
the last `60 + 20 + 5 = 85` rows. It calculates fallback medians again on that
slice, unlike training's whole-frame imputation. Its scalar quantile signal is
exactly a positive-prediction 0/1 rule for finite predictions. The commented-out
`1 + BEST_C * prediction` mapping is inactive.

### Reproduction requirements

- Use the source's saved selected parameters for the initial local reproduction;
  do not run the project's improvement search first.
- Replace global preprocessing with train-fitted values and isolate each fold's
  history. Record this necessary methodological correction.
- Reconcile current-row rolling features with the project's past-only temporal
  feature rule explicitly; do not silently label shifted features as identical.
- Respect the common development split and reserve final holdout evaluation
  until configuration selection is complete.
- Replace Kaggle paths/server with local sequential evaluation, preserving the
  LightGBM model and binary allocation policy.

## Baseline 2

Source: `baseline/hull-market-prediction-just-improved.ipynb`, cell 0.

### Actual model inputs and preprocessing

Load rows with `date_id >= 37`, keep the last 800, then filter columns at 50%
missingness. Although raw return columns initially survive loading,
`create_features` selects only prefixes `D/E/I/M/P/S/V` plus five derived columns:

| Feature | Source expression |
|---|---|
| U1 | I2 - I1 |
| U2 | M11 / ((I2 + I9 + I7) / 3) |
| V1_S1 | V1 * S1 |
| M11_V1 | M11 * V1 |
| I9_S1 | I9 * S1 |

The source does not guard the U2 denominator. I-prefixed columns are forward-filled
and backward-filled, then remaining base-feature gaps use medians. Derived features
are constructed before filling their inputs and are median-filled separately.
Preprocessing and XGBoost top-50 selection occur before the startup inner CV,
allowing validation information into preprocessing and supervised selection.

The code computes several lags, V1 rolling moments, EWMA, S1 slope, rolling
correlation, and `target_roll_std_5` during training. Its final `select_cols`
discards all these additional columns. In particular, the unshifted target rolling
statistic is not an actual model input; do not report it as demonstrated target
leakage or add it to the reproduction.

The selector uses 100 XGBoost estimators and seed 42 on standardized features.
Selected columns are standardized again for final fitting of all three models.
Startup tuning evaluates XGBoost/LightGBM on unscaled selected features while
ElasticNet uses fold-fitted scaling. Final fits therefore differ in preprocessing
from tree-model tuning. Dataclass defaults are superseded by active searches;
the fallback fitting block is an inert triple-quoted string.

### Allocation

With at least two recent observations, the volatility estimate is
`max(sqrt(0.3 * var(last_20_targets) + 0.7 * V1**2), 0.01)`, using NumPy's population
variance. If V1 is unavailable, the source falls back to a recent target standard
deviation (or 0.01). The V1 median is initialized once from training.

The raw prediction is `0.30 * ElasticNet + 0.35 * XGBoost + 0.35 * LightGBM`.
Multiply by 600 if current V1 is below its initial median, otherwise 400; clip
to [0, 2], divide by `volatility * 1.2`, and clip again. Return
`(0.75 * allocation + 0.25 * previous_allocation) * (1 - 0.00003)`.
The source calls the last factor a transaction cost, but it shrinks the position;
it does not debit realized strategy returns according to turnover.

### Online-update defects (static findings)

1. At the end of prediction, `previous_lagged` receives the already processed
   frame. `create_features(..., is_train=False)` drops
   `lagged_market_forward_excess_returns`. On subsequent calls the condition
   guarding updates is false. Actual source execution therefore leaves models
   fixed after startup, despite its documented online intent.
2. If the trigger were restored alone, the append path drops the assigned target
   in feature creation and then explicitly overwrites it with null. Such rows
   cannot train a regressor correctly.
3. It attempts to stack a top-50 append row onto a training frame retaining all
   engineered features, creating a schema mismatch.
4. The ordinary retraining branch uses the original `features` list, whereas
   inference selects `global_top_features`, creating a model/scaler width mismatch.
5. A lagged label must be paired with the feature row for the date it describes;
   copying it onto the prior stored row without checking its date alignment is
   not sufficient for a valid offline simulation.

Intended scheduling is refitting every row, with three model searches every 50
rows (three inner folds; 10 trials and 15 seconds per search). All Optuna samplers
are unseeded. Local reproduction must separately document literal frozen-model
behavior and any minimally repaired online variant. Restoring online behavior is
a reproduction adaptation, not evidence of a successful model improvement.

### Reproduction requirements

Fit missingness filtering, imputation, scaling, and feature selection using only
each allowed training portion. Remove future-dependent backward filling and
preserve target availability and the required gap. Use consistent selected-column
schemas and correctly aligned observed labels when repairing updates. Preserve
the recent-window concept, three model families, weights, and risk layer.
No holdout labels may drive hyperparameter adjustment: freeze selected parameters
for final sequential holdout evaluation, even if observed past labels are used
for scheduled refitting. Document this departure from the source's intended
periodic searches.

## Shared limitations

Neither notebook contains the public competition scoring formula. Kaggle-specific
imports, server files, `test.csv`, and cached artifacts are not in this repository.
Source URLs and authors are not present in source cells or metadata; attribution
remains an explicit TODO. Saved optimization metrics are evidence of external
runs only. They do not validate our future leakage-corrected reproductions.
