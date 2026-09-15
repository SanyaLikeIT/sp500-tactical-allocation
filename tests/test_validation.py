"""Chronology, holdout isolation, and label-availability checks."""

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    available_training_indices, development_folds, final_holdout_fold,
    market_feature_columns, partition_data, validate_dates,
)


def test_five_folds_have_exact_expected_boundaries_and_gap():
    folds = development_folds(np.arange(9048))
    assert len(folds) == 5
    for fold, start in zip(folds, [7968, 8148, 8328, 8508, 8688], strict=True):
        assert fold.train[0] == 0
        assert fold.train[-1] == start - 2
        np.testing.assert_array_equal(fold.validation, np.arange(start, start + 180))
        assert not np.intersect1d(fold.train, fold.validation).size
        assert fold.train.max() < fold.validation.min()
        assert fold.validation.max() < 8868
    assert folds[-1].validation[-1] == 8867


def test_final_holdout_has_its_own_gap():
    fold = final_holdout_fold(np.arange(9048))
    assert fold.train[-1] == 8866
    np.testing.assert_array_equal(fold.validation, np.arange(8868, 9048))
    assert 8867 not in fold.train


def test_holdout_target_changes_cannot_affect_development_data_or_splits():
    frame = pd.DataFrame({"date_id": np.arange(9048), "market_forward_excess_returns": np.zeros(9048)})
    before, holdout = partition_data(frame)
    frame.loc[8868:, "market_forward_excess_returns"] = 1e10
    after, changed_holdout = partition_data(frame)
    pd.testing.assert_frame_equal(before, after)
    assert len(holdout) == len(changed_holdout) == 180
    before.iloc[0, 1] = 123
    assert frame.iloc[0, 1] == 0
    for fold in development_folds(frame.date_id):
        assert np.all(frame.iloc[fold.train].market_forward_excess_returns == 0)
        assert np.all(frame.iloc[fold.validation].market_forward_excess_returns == 0)


@pytest.mark.parametrize("dates", [[], [1, 1], [2, 1], [1, 3], [1., 2.], ["1", "2"], [[1, 2]]])
def test_invalid_dates_are_rejected(dates):
    with pytest.raises(ValueError):
        validate_dates(dates)


def test_insufficient_history_is_rejected():
    with pytest.raises(ValueError):
        development_folds(np.arange(1081))
    with pytest.raises(ValueError):
        partition_data(pd.DataFrame({"date_id": np.arange(180)}))
    with pytest.raises(ValueError):
        final_holdout_fold(np.arange(181))


def test_online_history_excludes_current_and_gap_targets():
    for current in range(7968, 8148):
        train = available_training_indices(current, max_train_rows=800)
        assert len(train) == 800
        assert train[-1] == current - 2
        assert current not in train and current - 1 not in train
    np.testing.assert_array_equal(available_training_indices(5), [0, 1, 2, 3])


@pytest.mark.parametrize("index,window", [(1, None), (0, 800), (5, 0), (5, -1), (5, 2.5), (5.5, None), (True, None)])
def test_invalid_online_history_request(index, window):
    with pytest.raises(ValueError):
        available_training_indices(index, window)


def test_feature_selection_excludes_contemporaneous_targets_and_scoring_columns():
    frame = pd.DataFrame(columns=["date_id", "M1", "V1", "forward_returns", "risk_free_rate",
                                  "market_forward_excess_returns", "lagged_forward_returns", "is_scored"])
    assert market_feature_columns(frame) == ["M1", "V1"]


def test_real_dataset_split_uses_dates_only():
    dates = pd.read_csv("data/train.csv", usecols=["date_id"]).date_id
    assert len(development_folds(dates)) == 5
    assert final_holdout_fold(dates).validation[0] == 8868
