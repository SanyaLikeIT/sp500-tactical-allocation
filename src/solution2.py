"""Source-preserving ensemble and minimal causal repairs for Baseline 2.

Source: baseline/hull-market-prediction-just-improved.ipynb, cell 0.
Only the five derived inputs returned by its feature function are retained.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.validation import market_feature_columns, validate_dates


WINDOW = 800
TOP_K = 50
WEIGHTS = {"enet": .30, "xgb": .35, "lgb": .35}
DERIVED = ("U1", "U2", "V1_S1", "M11_V1", "I9_S1")


def derive_features(frame, columns):
    """Cross-sectional transforms before imputation, matching the source order."""
    base = frame[columns].astype(float).copy()
    if {"I1", "I2", "I7", "I9", "M11"}.issubset(columns):
        base["U1"] = base.I2 - base.I1
        denominator = (base.I2 + base.I9 + base.I7) / 3
        base["U2"] = base.M11 / denominator.replace(0, np.nan)
    else:
        base["U1"] = 0.
        base["U2"] = 0.
    for name, left, right in [("V1_S1", "V1", "S1"), ("M11_V1", "M11", "V1"), ("I9_S1", "I9", "S1")]:
        base[name] = base[left] * base[right] if {left, right}.issubset(columns) else 0.
    base = base.replace([np.inf, -np.inf], np.nan)
    interest = [c for c in columns if c.startswith("I")]
    base[interest] = base[interest].ffill()
    return base


class EnsemblePreprocessor:
    """Fit missingness filtering/medians on training only; no backward filling."""

    def fit_transform(self, training):
        validate_dates(training.date_id)
        candidates = market_feature_columns(training)
        self.columns_ = [c for c in candidates if training[c].isna().mean() <= .5]
        if not self.columns_:
            raise ValueError("No usable market features in training.")
        self.start_ = int(training.date_id.iloc[0])
        matrix = derive_features(training, self.columns_)
        self.medians_ = matrix.median().fillna(0.)
        return matrix.fillna(self.medians_)

    def transform(self, history):
        if not hasattr(self, "medians_"):
            raise ValueError("Fit the preprocessor first.")
        validate_dates(history.date_id)
        if int(history.date_id.iloc[0]) != self.start_:
            raise ValueError("Provide causal feature history from the preprocessing start date.")
        return derive_features(history, self.columns_).fillna(self.medians_)


def select_features(matrix, target, *, n_jobs=1):
    scaled = StandardScaler().fit_transform(matrix)
    selector = XGBRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    selector.fit(scaled, target)
    importance = pd.Series(selector.feature_importances_, index=matrix.columns)
    return importance.sort_values(ascending=False, kind="stable").head(TOP_K).index.tolist()


def component_model(kind, parameters, *, n_jobs=1):
    if kind == "enet":
        return ElasticNet(**parameters, max_iter=1_000_000)
    if kind == "xgb":
        return XGBRegressor(**parameters, objective="reg:squarederror", random_state=42, n_jobs=n_jobs)
    if kind == "lgb":
        return LGBMRegressor(**parameters, objective="regression", random_state=42,
                             n_jobs=n_jobs, deterministic=True, force_col_wise=True, verbosity=-1)
    raise ValueError(f"Unknown model family: {kind}")


def fit_ensemble(matrix, target, parameters, *, n_jobs=1):
    values = np.asarray(target, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training targets must be finite; they are not imputed.")
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(matrix), columns=matrix.columns, index=matrix.index)
    models = {kind: component_model(kind, parameters[kind], n_jobs=n_jobs) for kind in WEIGHTS}
    for model in models.values():
        model.fit(scaled, values)
    return scaler, models


def predict_ensemble(fitted, row):
    scaler, models = fitted
    matrix = pd.DataFrame(scaler.transform(row), columns=row.columns, index=row.index)
    components = {kind: float(model.predict(matrix)[0]) for kind, model in models.items()}
    return sum(WEIGHTS[k] * components[k] for k in WEIGHTS), components


def allocate(prediction, v1, v1_median, past_targets, previous_allocation):
    """Original clipping, volatility scaling, and smoothing, with observable labels."""
    recent = np.asarray(past_targets, dtype=float)[-20:]
    if not len(recent) or not np.isfinite(recent).all():
        raise ValueError("Volatility history must contain finite observed targets.")
    has_v1 = v1 is not None and np.isfinite(v1)
    proxy = float(v1) if has_v1 else (float(np.std(recent, ddof=1)) if len(recent) > 1 else .01)
    if not has_v1:
        proxy = proxy or .01
    volatility = max(float(np.sqrt(.3 * np.var(recent) + .7 * proxy**2)), .01) if len(recent) > 1 else max(proxy, .01)
    multiplier = 600. if has_v1 and v1 < v1_median else 400.
    if not np.isfinite(prediction) or not 0 <= previous_allocation <= 2:
        raise ValueError("Prediction/state must be finite and allocation bounded.")
    signal = np.clip(prediction * multiplier, 0, 2)
    unsmoothed = np.clip(signal / (volatility * 1.2), 0, 2)
    allocation = float((.75 * unsmoothed + .25 * previous_allocation) * (1 - .00003))
    return allocation, volatility, multiplier
