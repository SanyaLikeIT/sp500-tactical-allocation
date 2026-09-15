"""Temporal causality, train-only preprocessing, and baseline prediction checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.solution1 import (
    AUTHOR_PARAMETERS, TEMPORAL_COLUMNS, TemporalPreprocessor,
    baseline_model, binary_allocations, temporal_features,
)


def history(rows=100):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({c: rng.normal(size=rows) for c in TEMPORAL_COLUMNS})
    frame.insert(0, "date_id", np.arange(rows))
    frame["market_forward_excess_returns"] = rng.normal(0, .01, rows)
    frame["forward_returns"] = rng.normal(0, .01, rows)
    frame["risk_free_rate"] = .0001
    return frame


def test_author_parameters_match_saved_output_and_final_fit_semantics():
    audit = json.loads(Path("results/stage1/audit.json").read_text())
    saved = next(iter(audit["notebooks"][0]["saved_best_trials"].values()))
    assert AUTHOR_PARAMETERS == saved["parameters"]
    parameters = baseline_model().get_params()
    assert parameters["random_state"] == 42
    assert parameters["subsample_freq"] == 0
    assert parameters["n_estimators"] == 6917


def test_temporal_windows_use_only_previous_observations():
    frame = history()
    frame["M4"] = np.arange(len(frame), dtype=float)
    features = temporal_features(frame)
    assert features.loc[10, "M4_lag_3"] == 7
    assert features.loc[10, "M4_roll_mean_2"] == 8.5
    assert features.loc[10, "M4_roll_std_2"] == pytest.approx(np.std([8, 9], ddof=1))
    assert features.loc[10, "M4"] == 10
    assert pd.isna(features.loc[0, "M4_lag_1"])
    changed = frame.copy()
    changed.loc[10:, "M4"] = 1e9
    temporal = [c for c in features if c.startswith("M4_")]
    pd.testing.assert_series_equal(features.loc[10, temporal], temporal_features(changed).loc[10, temporal])


def test_future_suffix_cannot_change_earlier_features_or_fitted_medians():
    frame = history()
    frame.loc[:5, "M4"] = np.nan
    processor = TemporalPreprocessor()
    training = processor.fit_transform(frame.iloc[:70])
    medians = processor.medians_.copy()
    full = processor.transform(frame)
    changed = frame.copy()
    changed.loc[80:, list(TEMPORAL_COLUMNS)] = 1e8
    transformed = processor.transform(changed)
    pd.testing.assert_frame_equal(full.iloc[:80], transformed.iloc[:80])
    pd.testing.assert_frame_equal(training, full.iloc[:70])
    pd.testing.assert_series_equal(medians, processor.medians_)
    assert full.loc[0, "M4"] == medians["M4"]


def test_all_missing_training_column_uses_zero_not_future_median():
    frame = history()
    frame.loc[:69, "M4"] = np.nan
    processor = TemporalPreprocessor()
    train = processor.fit_transform(frame.iloc[:70])
    assert processor.medians_["M4"] == 0
    assert (train["M4"] == 0).all()
    assert (processor.transform(frame).iloc[:70]["M4"] == 0).all()


def test_targets_and_scoring_values_cannot_enter_features():
    frame = history()
    expected = temporal_features(frame)
    changed = frame.copy()
    changed[["market_forward_excess_returns", "forward_returns", "risk_free_rate"]] = np.nan
    pd.testing.assert_frame_equal(expected, temporal_features(changed))
    assert "date_id" not in expected and "market_forward_excess_returns" not in expected


def test_batch_matches_sequential_prefix_features_and_model_predictions():
    frame = history(130)
    frame.loc[101:104, "M4"] = np.nan
    processor = TemporalPreprocessor()
    X_train = processor.fit_transform(frame.iloc[:100])
    model = baseline_model(n_jobs=1, smoke=True)
    model.fit(X_train, frame.iloc[:100].market_forward_excess_returns)
    batch = processor.transform(frame)
    # Training ends at 99, row 100 is the gap, prediction begins at 101.
    for date in [101, 105, 129]:
        sequential = processor.transform(frame.iloc[:date + 1]).iloc[[-1]]
        pd.testing.assert_frame_equal(batch.iloc[[date]], sequential)
        np.testing.assert_array_equal(model.predict(batch.iloc[[date]]), model.predict(sequential))


def test_model_fit_is_deterministic_and_validation_targets_are_unused():
    frame = history(130)
    processor = TemporalPreprocessor()
    X_train = processor.fit_transform(frame.iloc[:100])
    before = processor.transform(frame).iloc[101:]
    models = [baseline_model(n_jobs=1, smoke=True) for _ in range(2)]
    for model in models:
        model.fit(X_train, frame.iloc[:100].market_forward_excess_returns)
    frame.loc[100:, "market_forward_excess_returns"] = 1e15
    after = processor.transform(frame).iloc[101:]
    np.testing.assert_array_equal(models[0].predict(before), models[1].predict(after))


def test_real_schema_retains_89_original_and_224_derived_features():
    frame = pd.read_csv("data/train.csv", nrows=70)
    features = TemporalPreprocessor().fit_transform(frame)
    assert features.shape == (70, 313)
    assert np.isfinite(features.to_numpy()).all()
    assert not {"E7", "V10", "S3", "M1", "M14"}.intersection(features.columns)


def test_binary_rule_including_zero_and_invalid_predictions():
    np.testing.assert_array_equal(binary_allocations([-1, -1e-20, 0, 1e-20, 1]), [0, 0, 0, 1, 1])
    with pytest.raises(ValueError):
        binary_allocations([np.nan])


def test_history_and_schema_errors_are_explicit():
    processor = TemporalPreprocessor()
    with pytest.raises(ValueError):
        processor.transform(history())
    processor.fit_transform(history().iloc[:70])
    with pytest.raises(ValueError):
        processor.transform(history().iloc[1:])
    with pytest.raises(ValueError):
        processor.transform(history().drop(columns="M4"))
    with pytest.raises(ValueError):
        temporal_features(history().iloc[::-1])
