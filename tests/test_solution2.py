"""Causal preprocessing, sequential label availability, and source risk policy."""

import numpy as np
import pandas as pd
import pytest

import scripts.reproduce_solution2 as runner
from src.solution2 import DERIVED, EnsemblePreprocessor, allocate, derive_features
from src.validation import TARGET, market_feature_columns


def sample_data(rows=960):
    rng = np.random.default_rng(48)
    frame = pd.DataFrame({name: rng.normal(size=rows) for name in
                          ["I1", "I2", "I7", "I9", "M11", "V1", "S1", "D1"]})
    frame.insert(0, "date_id", np.arange(rows))
    frame[TARGET] = rng.normal(0, .01, rows)
    frame["forward_returns"] = rng.normal(.001, .01, rows)
    frame["risk_free_rate"] = .0001
    return frame


def test_exact_features_and_no_target_inputs():
    data = sample_data(20)
    result = derive_features(data, market_feature_columns(data))
    assert set(result) == set(market_feature_columns(data)) | set(DERIVED)
    np.testing.assert_allclose(result.U1, data.I2-data.I1)
    np.testing.assert_allclose(result.U2, data.M11/((data.I2+data.I9+data.I7)/3))
    for name, left, right in [("V1_S1", "V1", "S1"), ("M11_V1", "M11", "V1"), ("I9_S1", "I9", "S1")]:
        np.testing.assert_allclose(result[name], data[left]*data[right])


def test_train_only_imputation_and_causal_prefix():
    data = sample_data(40)
    data.loc[0, "I1"] = np.nan
    data.loc[:11, "D1"] = np.nan
    data.loc[20:25, "I1"] = np.nan
    data.loc[5, ["I2", "I7", "I9"]] = 0.
    processor = EnsemblePreprocessor()
    train = processor.fit_transform(data.iloc[:20])
    assert "D1" not in train
    assert train.I1.iloc[0] == processor.medians_["I1"]
    assert train.U2.iloc[5] == processor.medians_["U2"]
    prefix = processor.transform(data.iloc[:26])
    changed = data.copy()
    changed.loc[26:, market_feature_columns(data)] = 1e9
    changed[TARGET] = 1e9
    pd.testing.assert_frame_equal(prefix, processor.transform(changed).iloc[:26])
    assert prefix.I1.iloc[25] == data.I1.iloc[19]
    assert np.isfinite(prefix.to_numpy()).all()


@pytest.mark.parametrize("prediction,v1,previous", [(1., .1, 0.), (-1., .1, 2.), (.0001, 0., .3)])
def test_source_risk_formula(prediction, v1, previous):
    history = np.linspace(-.03, .03, 20)
    volatility = max(np.sqrt(.3*np.var(history)+.7*v1**2), .01)
    multiplier = 600 if v1 < .2 else 400
    expected = (.75*np.clip(np.clip(prediction*multiplier, 0, 2)/(volatility*1.2), 0, 2)
                + .25*previous)*(1-.00003)
    actual, risk, signal = allocate(prediction, v1, .2, history, previous)
    assert actual == pytest.approx(expected)
    assert risk == pytest.approx(volatility)
    assert signal == multiplier
    assert 0 <= actual <= 2


def test_window_and_gap():
    data = sample_data()
    indices = runner.training_indices(data, 900)
    assert len(indices) == 800
    assert indices[0] == 99
    assert indices[-1] == 898


def cheap_search(window, **kwargs):
    tree = {"n_estimators": 5, "max_depth": 3, "learning_rate": .05,
            "reg_alpha": .001, "reg_lambda": .1}
    return {"enet": {"alpha": .001, "l1_ratio": .5}, "xgb": tree, "lgb": tree}, {}


def test_real_models_do_not_see_current_gap_or_future_targets(monkeypatch):
    monkeypatch.setattr(runner, "author_search", cheap_search)
    data = sample_data()
    before, _ = runner.forecast_fold(data, 900, 4)
    changed = data.copy()
    changed.loc[901:, TARGET] = 100.
    changed.loc[903:, market_feature_columns(data)] = -100.
    after, _ = runner.forecast_fold(changed, 900, 4)
    for variant in before:
        columns = ["prediction", "allocation", "risk_volatility", "training_end"]
        pd.testing.assert_frame_equal(before[variant].loc[:2, columns], after[variant].loc[:2, columns])
    assert before["baseline"].training_end.tolist() == [898, 899, 900, 901]
    assert before["frozen_control"].training_end.tolist() == [898]*4
    assert before["baseline"].prediction.iloc[3] != after["baseline"].prediction.iloc[3]


def test_online_search_schedule_and_window(monkeypatch):
    searches, fits = [], []

    def search(window, **kwargs):
        searches.append(window.date_id.to_list())
        return cheap_search(window)

    def fit(matrix, target, parameters, **kwargs):
        fits.append(matrix.index.to_list())
        return float(target.mean())

    monkeypatch.setattr(runner, "author_search", search)
    monkeypatch.setattr(runner, "select_features", lambda matrix, target, **kw: matrix.columns.to_list())
    monkeypatch.setattr(runner, "fit_ensemble", fit)
    monkeypatch.setattr(runner, "predict_ensemble", lambda fitted, row: (fitted, {k: fitted for k in runner.WEIGHTS}))
    outputs, audit = runner.forecast_fold(sample_data(), 900, 52)
    assert [e["offset"] for e in audit["search_events"]] == [0, 50]
    assert [s[-1] for s in searches] == [898, 948]
    assert all(len(s) == 800 for s in searches)
    assert len(fits) == 52
    assert [s[-1] for s in fits] == list(range(898, 950))
    assert outputs["frozen_control"].prediction.nunique() == 1
    assert outputs["baseline"].prediction.nunique() > 1
