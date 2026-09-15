"""Scoring checks against official outputs, analytic identities, and invalid inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    UndefinedMetricError, adjusted_sharpe, clip_allocations, cumulative_return,
    evaluate_predictions, max_drawdown, regression_metrics, score,
    strategy_returns, trading_metrics, validate_allocations,
)


REFERENCE = json.loads((Path(__file__).parent / "fixtures/official_metric_cases.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_matches_official_v3_reference(case):
    result = adjusted_sharpe(case["forward_returns"], case["risk_free_rate"], case["allocations"])
    assert result == pytest.approx(case["adjusted_sharpe"], rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("position", [0.0, 1.0, 2.0])
def test_cash_market_and_leverage_returns(position):
    market = np.array([0.02, -0.01, 0.03])
    cash = np.array([0.001, 0.002, 0.001])
    expected = {0.0: cash, 1.0: market, 2.0: 2 * market - cash}[position]
    np.testing.assert_allclose(strategy_returns(market, cash, np.full(3, position)), expected)


@pytest.mark.parametrize("values", [[-1e-12, 1], [1, 2.00000001], [np.nan, 1], [np.inf, 1], ["1", "0"], [], [[1]], [True]])
def test_invalid_allocations_are_rejected(values):
    with pytest.raises(ValueError):
        validate_allocations(values)


def test_explicit_clipping_always_respects_bounds():
    signals = np.random.default_rng(42).normal(1, 10, 1000)
    positions = clip_allocations(signals)
    assert np.all((positions >= 0) & (positions <= 2))
    np.testing.assert_array_equal(clip_allocations([-10, 0, 1, 2, 10]), [0, 0, 1, 2, 2])
    with pytest.raises(ValueError):
        clip_allocations([np.nan])


def test_raw_sharpe_uses_geometric_excess_and_total_return_sample_std():
    market = np.array([0.1, -0.05])
    cash = np.array([0.01, 0.02])
    expected = (np.sqrt(1.09 * 0.93) - 1) / np.std(market, ddof=1) * np.sqrt(252)
    metrics = trading_metrics(market, cash, [1, 1])
    assert metrics["raw_sharpe"] == pytest.approx(expected)
    assert metrics["adjusted_sharpe"] == pytest.approx(expected)
    assert metrics["volatility_penalty"] == metrics["return_penalty"] == 1
    assert metrics["strategy_volatility"] == pytest.approx(np.std(market, ddof=1) * np.sqrt(252))


def test_volatility_and_underperformance_penalties():
    market = [.02, -.01, .03, -.01]
    leveraged = trading_metrics(market, [0]*4, [2]*4)
    assert leveraged["volatility_penalty"] == pytest.approx(1.8)
    half = trading_metrics(market, [0]*4, [.5]*4)
    assert half["return_penalty"] > 1
    assert half["adjusted_sharpe"] < half["raw_sharpe"]


def test_constant_market_or_strategy_has_explicit_undefined_score():
    for market, cash, positions in [([.01, -.01], [0, 0], [0, 0]), ([.01, .01], [0, 0], [0, 1])]:
        result = trading_metrics(market, cash, positions)
        assert result["adjusted_sharpe"] is None
        assert result["metric_errors"]
        json.dumps(result, allow_nan=False)
        with pytest.raises(UndefinedMetricError):
            adjusted_sharpe(market, cash, positions)


def test_regression_diagnostics_use_standard_r2_and_rank_order():
    assert regression_metrics([1, 2, 3], [1, 2, 3]) == {"r2": 1.0, "spearman": 1.0, "rmse": 0.0}
    mean = regression_metrics([1, 2, 3], [2, 2, 2])
    assert mean["r2"] == 0
    assert mean["spearman"] is None
    assert mean["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert regression_metrics([1, 2, 3], [3, 2, 1])["r2"] < 0
    assert regression_metrics([1, 2, 3], [3, 2, 1])["spearman"] == -1
    assert regression_metrics([1, 1, 1], [1, 1, 1])["r2"] is None
    assert regression_metrics([1, 1, 2], [2, 2, 1])["spearman"] == pytest.approx(-1)


def test_compounding_and_drawdown_include_starting_equity():
    assert cumulative_return([.1, -.2, .25]) == pytest.approx(.1)
    assert max_drawdown([.1, -.2, .25]) == pytest.approx(.2)
    assert max_drawdown([-.1, 0]) == pytest.approx(.1)
    assert max_drawdown([.1, .1]) == 0
    assert max_drawdown([0, 0]) == 0


@pytest.mark.parametrize("values", [[], [np.nan, .01], [-1, .01], [-1.1, .01]])
def test_invalid_equity_series(values):
    with pytest.raises(ValueError):
        cumulative_return(values)
    with pytest.raises(ValueError):
        max_drawdown(values)


def test_input_shapes_and_geometric_domain():
    with pytest.raises(ValueError):
        adjusted_sharpe([.1], [0], [1])
    with pytest.raises(ValueError):
        adjusted_sharpe([.1, -.1], [0], [1, 1])
    with pytest.raises(ValueError):
        adjusted_sharpe([-.6, .1], [0, 0], [2, 2])
    with pytest.raises(ValueError):
        regression_metrics([1, np.inf], [1, 1])
    with pytest.raises(ValueError):
        evaluate_predictions(targets=[1, 2, 3], predictions=[1, 2, 3],
                             forward_returns=[.1, -.1], risk_free_rate=[0, 0], allocations=[1, 1])


def test_combined_metrics_keep_target_separate_from_trading_returns():
    result = evaluate_predictions(targets=[9, 10, 11], predictions=[9, 10, 11],
                                  forward_returns=[.01, -.01, .02], risk_free_rate=[0, 0, 0], allocations=[1, 1, 1])
    assert result["r2"] == 1
    assert result["adjusted_sharpe"] == adjusted_sharpe([.01, -.01, .02], [0, 0, 0], [1, 1, 1])
    json.dumps(result, allow_nan=False)


def test_dataframe_adapter_checks_alignment_and_does_not_mutate():
    solution = pd.DataFrame({"date_id": [10, 11], "forward_returns": [.01, -.01], "risk_free_rate": [0, 0]}, index=[5, 6])
    submission = pd.DataFrame({"date_id": [10, 11], "prediction": [1, 1]}, index=[20, 21])
    original = solution.copy(deep=True)
    submitted = submission.copy(deep=True)
    assert np.isfinite(score(solution, submission))
    pd.testing.assert_frame_equal(solution, original)
    pd.testing.assert_frame_equal(submission, submitted)
    with pytest.raises(ValueError):
        score(solution, submission.iloc[::-1])
    submission["date_id"] = [10, 10]
    with pytest.raises(ValueError):
        score(solution, submission)
