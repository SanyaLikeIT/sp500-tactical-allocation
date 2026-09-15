"""Nested stopping isolation, bounded search, and final-evaluation guards."""

import numpy as np
import optuna
import pandas as pd
import pytest

from scripts.evaluate_solution1 import validate_frozen
from src.solution1_tuning import final_iterations, fit_fold, prepare_fold, stopping_split, suggest_parameters
from src.validation import Fold
from tests.test_solution1 import history


def test_sampled_parameters_respect_tree_complexity():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    for _ in range(25):
        trial = study.ask()
        parameters = suggest_parameters(trial)
        assert parameters["num_leaves"] <= 2**parameters["max_depth"]
        assert parameters["num_leaves"] <= 128
        assert parameters["reg_alpha"] > 0 and parameters["reg_lambda"] > 0
        study.tell(trial, 0.)


def test_inner_stopping_has_gap_and_is_contained_in_outer_training():
    split = stopping_split(7967)
    assert split.train[-1] + 2 == split.validation[0]
    assert len(split.validation) == 180 and split.validation[-1] == 7966
    with pytest.raises(ValueError):
        stopping_split(181)


def test_outer_validation_labels_cannot_control_stopping_or_predictions():
    frame = history(550)
    fold = Fold(np.arange(400), np.arange(401, 501))
    original = prepare_fold(frame, fold)
    changed = frame.copy()
    changed.loc[400:, "market_forward_excess_returns"] = 123.
    changed.loc[501:, list(frame.columns[1:])] = -999.
    other = prepare_fold(changed, fold)
    for key in ["inner_X", "stopping_X", "outer_X", "validation_X"]:
        pd.testing.assert_frame_equal(original[key], other[key])
    for key in ["inner_y", "stopping_y", "outer_y"]:
        np.testing.assert_array_equal(original[key], other[key])
    params = {"max_depth": 3, "num_leaves": 8, "n_estimators": 10, "learning_rate": .03}
    first, pred1 = fit_fold(original, params, n_jobs=1)
    second, pred2 = fit_fold(other, params, n_jobs=1)
    assert first["best_iteration"] == second["best_iteration"]
    np.testing.assert_array_equal(pred1.prediction, pred2.prediction)


def test_final_tree_count_is_development_median_only():
    assert final_iterations([1, 20, 4, 10, 100]) == 10
    with pytest.raises(ValueError):
        final_iterations([0, 2])


@pytest.mark.parametrize("change", [{"status": "running"}, {"smoke": True}, {"holdout_accessed": True}, {"dataset_sha256": "other"}])
def test_final_evaluation_rejects_unfrozen_or_contaminated_configuration(change):
    configuration = {"status": "frozen", "smoke": False, "holdout_accessed": False, "dataset_sha256": "expected"}
    validate_frozen(configuration, "expected")
    configuration.update(change)
    with pytest.raises(ValueError):
        validate_frozen(configuration, "expected")
