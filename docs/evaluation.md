# Shared evaluation specification

## Source and scope

The primary formula is from **Kaggle Competition Metrics, Hull Competition Sharpe,
version 3**, retrieved on 2026-09-15:

- [Competition evaluation](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview/evaluation)
- [Official metric notebook](https://www.kaggle.com/code/metric/hull-competition-sharpe)
- [Public source download](https://www.kaggle.com/api/v1/kernels/pull/metric/hull-competition-sharpe)

The competition's public `competitions.PageService/ListPages` response for
competition 111543, Evaluation page 520010, links to that notebook. The downloaded
notebook contains one code cell. Its source hash, author, version, and reference
scores are recorded in `tests/fixtures/official_metric_cases.json`. The retrieved
source was inspected before execution. Eight synthetic cases were scored with the
official function and saved as reference outputs; project tests do not regenerate
their expectations using our implementation. The original two baseline notebooks
contain no competition scorer.

The implementation in `src/metrics.py` is the shared source of truth for every
future baseline and improved experiment. Synthetic reference scores are test
fixtures, not financial performance evidence. The fixed seed is 42. The saved
cases cover market, cash, half exposure, leverage, binary and mixed allocations,
negative returns, and the official score cap. No train.csv return values enter
these fixtures.

## Formula and units

For each row, let `m` be `forward_returns`, `f` be `risk_free_rate`, and `p` be
the position in [0, 2]. Strategy total return is `s = f*(1-p) + p*m`.
All returns are fractions: 0.01 means 1%.

Let `g(x) = product(1+x)**(1/n) - 1` and let `sd` be sample standard deviation
(`ddof=1`). With 252 trading days per year:

```text
strategy_excess_mean = g(s - f)
market_excess_mean   = g(m - f)
raw_sharpe          = strategy_excess_mean / sd(s) * sqrt(252)
volatility_penalty  = 1 + max(0, sd(s) / sd(m) - 1.2)
return_gap          = max(0, (market_excess_mean - strategy_excess_mean) * 100 * 252)
return_penalty      = 1 + return_gap**2 / 100
adjusted_sharpe     = min(raw_sharpe / (volatility_penalty * return_penalty), 1_000_000)
```

The geometric mean is computed with `log1p` and `expm1` for numerical stability.
The raw score uses geometric excess returns divided by the volatility of total
strategy returns. It is not the usual arithmetic-mean excess-return Sharpe.
The formula has no turnover-based transaction costs. Negative raw scores remain
negative; division by penalties can move them closer to zero. We preserve this
public behavior rather than changing the objective between experiments.

| Output | Definition |
|---|---|
| adjusted_sharpe | Primary competition score; higher is better |
| raw_sharpe | The competition's Sharpe before penalties |
| r2 | Standard `1 - SSE/SST` against the supplied prediction target |
| spearman | Pearson correlation of average ranks, including tie handling |
| rmse | Square root of mean squared prediction error |
| strategy_volatility | `sd(s) * sqrt(252)`, fraction per square-root year |
| cumulative_return | `product(1+s) - 1`, total fractional return |
| max_drawdown | Largest fractional peak-to-trough equity loss, nonnegative |

The equity curve starts at 1 before the first observation, so a loss on the first
day counts toward drawdown. Reported volatility is fractional (0.20 means 20%);
the reference scorer's intermediate percentage volatilities cancel in its ratio.
Extra audit outputs include market volatility, both penalties, and metric errors.
R² is diagnostic, never an absolute-value optimization target. Constant predictions
can have valid R²/RMSE even when Spearman is undefined.

## Invalid and degenerate inputs

- Scoring rejects out-of-range positions; it never silently clips them. Policy
  code can explicitly call `clip_allocations` before scoring. Both paths reject
  NaN, infinity, nonnumeric inputs, empty vectors, and invalid dimensions.
- Scoring requires at least two observations and matching vector lengths.
  Array inputs must already be aligned chronologically. The DataFrame `score`
  adapter additionally requires matching, unique, nonmissing row IDs, ignores
  unrelated pandas index labels, and never mutates input frames.
- Returns at or below -100% are outside the supported positive-equity geometric
  domain. Nonfinite values and compounding overflow raise errors.
- With zero strategy or market volatility, the strict `adjusted_sharpe`/`score`
  APIs raise `UndefinedMetricError`, consistent with the reference's rejection
  of zero denominators. `trading_metrics` instead preserves the remaining
  diagnostics, returns `None` for undefined Sharpe values, and records reasons
  in `metric_errors`. Such values serialize to JSON `null`, not NaN.
- Constant targets produce `r2=None`; constant targets or predictions produce
  `spearman=None`. These are explicitly undefined, not evidence of a zero score.
- Future tuning code must mark undefined primary objectives as failed/pruned
  trials or use an explicitly documented finite rejection value. Never silently
  average away an invalid fold or treat it as successful zero-Sharpe performance.

Input validation and stable arithmetic are documented adaptations to the reference
implementation, which otherwise relies on pandas alignment and missing-value
behavior. They do not change the mathematical score for valid financial data.

## Chronological validation and leakage prevention

`src/validation.py` reserves the final 180 rows before constructing five development
folds with `TimeSeriesSplit(n_splits=5, test_size=180, gap=1)`. There is no shuffle.
Dates must be consecutive, ascending, unique integers; malformed order is rejected,
not silently sorted. Indices are positions in the original data.

| Split | Training date_id | Validation date_id | Gap date_id |
|---|---|---|---|
| Development 1 | 0–7966 | 7968–8147 | 7967 |
| Development 2 | 0–8146 | 8148–8327 | 8147 |
| Development 3 | 0–8326 | 8328–8507 | 8327 |
| Development 4 | 0–8506 | 8508–8687 | 8507 |
| Development 5 | 0–8686 | 8688–8867 | 8687 |
| Reserved final holdout | 0–8866 | 8868–9047 | 8867 |

An earlier validation period may enter a later expanding training fold; that is
valid chronological reuse. Each fold still excludes its own current/future labels.
The holdout cannot select feature sets, parameters, windows, thresholds, weights,
risk settings, or early-stopping iterations. Its configuration must be frozen
before final evaluation. Recording its date boundaries is not evaluating it.

For sequential prediction at row `t`, `available_training_indices` permits labels
through `t-2`, retaining the one-row gap; an optional recent window keeps the last
N allowed rows. Prior holdout labels may become available for scheduled refitting
of a frozen configuration, but must not trigger hyperparameter selection.

`partition_data` returns independent development/holdout copies; split generation
uses dates only. `market_feature_columns` restricts raw predictors to the observed
market feature naming convention, excluding IDs and target/scoring columns.
These helpers do not enforce how callers subsequently fit models. Future model
stages must add integration checks for fold-fitted imputation, scaling, supervised
selection, training-only thresholds, and strictly past temporal features. No such
preprocessing/model implementation exists in Stage 2, so the current tests do not
claim to prove end-to-end model leakage freedom.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.check_evaluation
```

The first command tests official score parity, analytic metric identities,
degenerate inputs, nonmutation/alignment, chronological boundaries, holdout target
perturbation, allocation bounds, and sequential label availability. The second
reads only `date_id` from the real dataset and writes split boundaries plus
synthetic reference comparisons and package versions under `results/stage2/`.
