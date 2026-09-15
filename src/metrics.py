"""Shared competition scoring and return/regression diagnostics.

Formula source: Kaggle Competition Metrics, Hull Competition Sharpe, version 3:
https://www.kaggle.com/code/metric/hull-competition-sharpe
Retrieved 2026-09-15; provenance and reference outputs are recorded under tests/fixtures.
Geometric excess means, total-return sample standard deviations, penalties, and
the upper score cap follow that source. Inputs are validated more strictly here.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, root_mean_squared_error


TRADING_DAYS = 252
SCORE_CAP = 1_000_000.0


class UndefinedMetricError(ValueError):
    """A valid return series has a zero denominator in the competition formula."""


def _vector(values, name: str, min_size: int = 2) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size < min_size or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric one-dimensional sequence of length >= {min_size}.")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _same_length(*arrays: np.ndarray) -> None:
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Inputs must have matching lengths and chronological row order.")


def validate_allocations(allocations) -> np.ndarray:
    positions = _vector(allocations, "allocations", min_size=1)
    if np.any((positions < 0) | (positions > 2)):
        raise ValueError("Allocations must lie in [0, 2]; scoring never clips invalid input.")
    return positions


def clip_allocations(allocations) -> np.ndarray:
    """Explicit policy-layer clipping; NaN and infinite signals remain errors."""
    return np.clip(_vector(allocations, "allocations", min_size=1), 0.0, 2.0)


def strategy_returns(forward_returns, risk_free_rate, allocations) -> np.ndarray:
    market = _vector(forward_returns, "forward_returns")
    cash = _vector(risk_free_rate, "risk_free_rate")
    positions = validate_allocations(allocations)
    _same_length(market, cash, positions)
    with np.errstate(over="ignore", invalid="ignore"):
        result = cash * (1.0 - positions) + positions * market
    if not np.isfinite(result).all():
        raise ValueError("Strategy returns overflowed.")
    return result


def _geometric_mean(returns: np.ndarray) -> float:
    if np.any(returns <= -1):
        raise ValueError("Geometric scoring requires each excess return to be greater than -1.")
    # Equivalent to product(1 + returns)**(1/n) - 1, with stable compounding.
    return float(np.expm1(np.log1p(returns).mean()))


def cumulative_return(returns) -> float:
    values = _vector(returns, "returns", min_size=1)
    if np.any(values <= -1):
        raise ValueError("Equity diagnostics require each total return to be greater than -1.")
    with np.errstate(over="ignore"):
        result = float(np.expm1(np.log1p(values).sum()))
    if not np.isfinite(result):
        raise ValueError("Cumulative return overflowed.")
    return result


def max_drawdown(returns) -> float:
    """Largest peak-to-trough loss as a nonnegative fraction, including initial equity 1."""
    values = _vector(returns, "returns", min_size=1)
    if np.any(values <= -1):
        raise ValueError("Equity diagnostics require each total return to be greater than -1.")
    log_equity = np.r_[0.0, np.cumsum(np.log1p(values))]
    return float(-np.expm1(np.min(log_equity - np.maximum.accumulate(log_equity))))


def trading_metrics(forward_returns, risk_free_rate, allocations) -> dict:
    """Report trading diagnostics; undefined Sharpe values are None with reasons.

    Volatility and cumulative return use fractional units, not percent. Raw Sharpe
    is the competition's geometric Sharpe before penalties, not arithmetic Sharpe.
    """
    market = _vector(forward_returns, "forward_returns")
    cash = _vector(risk_free_rate, "risk_free_rate")
    strategy = strategy_returns(market, cash, allocations)
    mean_strategy = _geometric_mean(strategy - cash)
    mean_market = _geometric_mean(market - cash)
    strategy_std = float(np.std(strategy, ddof=1))
    market_std = float(np.std(market, ddof=1))
    # Equality checks also catch roundoff in std for exactly constant series.
    strategy_constant = bool(np.all(strategy == strategy[0])) or strategy_std == 0
    market_constant = bool(np.all(market == market[0])) or market_std == 0
    errors = []
    raw = None
    adjusted = None
    vol_penalty = None
    return_gap = max(0.0, (mean_market - mean_strategy) * 100 * TRADING_DAYS)
    return_penalty = 1.0 + return_gap**2 / 100.0
    if strategy_constant:
        errors.append("Strategy standard deviation is zero.")
    else:
        raw = float(mean_strategy / strategy_std * np.sqrt(TRADING_DAYS))
    if market_constant:
        errors.append("Market standard deviation is zero.")
    else:
        vol_penalty = 1.0 + max(0.0, strategy_std / market_std - 1.2)
    if not errors:
        adjusted = min(float(raw / (vol_penalty * return_penalty)), SCORE_CAP)
    return {
        "adjusted_sharpe": adjusted, "raw_sharpe": raw,
        "strategy_volatility": strategy_std * float(np.sqrt(TRADING_DAYS)),
        "market_volatility": market_std * float(np.sqrt(TRADING_DAYS)),
        "cumulative_return": cumulative_return(strategy),
        "max_drawdown": max_drawdown(strategy),
        "volatility_penalty": vol_penalty, "return_penalty": return_penalty,
        "metric_errors": errors,
    }


def adjusted_sharpe(forward_returns, risk_free_rate, allocations) -> float:
    """Strict scalar scorer: undefined scores raise rather than silently becoming zero."""
    result = trading_metrics(forward_returns, risk_free_rate, allocations)
    if result["adjusted_sharpe"] is None:
        raise UndefinedMetricError(" ".join(result["metric_errors"]))
    return result["adjusted_sharpe"]


def regression_metrics(targets, predictions) -> dict:
    """Return standard R², rank correlation, and RMSE; undefined diagnostics are None."""
    target = _vector(targets, "targets")
    prediction = _vector(predictions, "predictions")
    _same_length(target, prediction)
    target_constant = bool(np.all(target == target[0]))
    prediction_constant = bool(np.all(prediction == prediction[0]))
    return {
        "r2": None if target_constant else float(r2_score(target, prediction)),
        "spearman": None if target_constant or prediction_constant else float(spearmanr(target, prediction).statistic),
        "rmse": float(root_mean_squared_error(target, prediction)),
    }


def evaluate_predictions(*, targets, predictions, forward_returns, risk_free_rate, allocations) -> dict:
    """Evaluate regression and trading using separately supplied target/return columns."""
    arrays = [_vector(v, name) for name, v in (
        ("targets", targets), ("predictions", predictions),
        ("forward_returns", forward_returns), ("risk_free_rate", risk_free_rate),
        ("allocations", allocations),
    )]
    _same_length(*arrays)
    return {**trading_metrics(*arrays[2:]), **regression_metrics(*arrays[:2])}


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "date_id") -> float:
    """DataFrame adapter with strict row-ID matching; neither input is mutated."""
    for frame in (solution, submission):
        if not frame.columns.is_unique or row_id_column_name not in frame:
            raise ValueError("Scoring frames must have unique columns and an explicit row-ID column.")
        if frame[row_id_column_name].isna().any() or not frame[row_id_column_name].is_unique:
            raise ValueError("Scoring row IDs must be nonmissing and unique.")
    if not np.array_equal(solution[row_id_column_name].to_numpy(), submission[row_id_column_name].to_numpy()):
        raise ValueError("Solution and submission row IDs must match in order.")
    required = {"forward_returns", "risk_free_rate"}
    if not required.issubset(solution.columns) or "prediction" not in submission:
        raise ValueError("Missing scoring return columns or submission prediction column.")
    return adjusted_sharpe(solution["forward_returns"], solution["risk_free_rate"], submission["prediction"])
