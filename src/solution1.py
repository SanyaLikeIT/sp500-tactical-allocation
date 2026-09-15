"""Leakage-corrected reproduction of the external single-LightGBM baseline.

Source: baseline/hull-eda-training-pipeline.ipynb, cells 8--12.
Model parameters are the author's saved trial 10, used by the final fit.
Temporal windows are shifted one row and imputation is fitted on training only;
these methodological corrections are not hyperparameter improvements.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.validation import market_feature_columns, validate_dates


TEMPORAL_COLUMNS = ("M4", "V13", "S5", "S2", "D2", "E19", "P7", "P6", "P3", "P13", "P4", "P5", "M2", "V5")
LAGS = (1, 3, 5, 7, 14, 20)
WINDOWS = (2, 5, 10, 20, 60)
DROPPED_FEATURES = ("E7", "V10", "S3", "M1", "M14")
AUTHOR_PARAMETERS = {
    "n_estimators": 6917,
    "learning_rate": 0.021484317145938295,
    "max_depth": 8,
    "num_leaves": 2662,
    "reg_lambda": 0.0072045470966688365,
    "reg_alpha": 0.0013718972580079596,
    "colsample_bytree": 0.7659097201144509,
    "subsample": 0.7373863632419442,
}


def temporal_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build raw + lag/rolling inputs using a chronological feature history.

    Raw features at t are observed inputs. All derived temporal features use rows
    no later than t-1. Forward filling is causal; no medians are learned here.
    Full history from its starting date is required, including any gap rows.
    """
    if "date_id" not in history or not history.columns.is_unique:
        raise ValueError("History requires unique columns and date_id.")
    validate_dates(history["date_id"])
    columns = [c for c in market_feature_columns(history) if c not in DROPPED_FEATURES]
    if not set(TEMPORAL_COLUMNS).issubset(columns):
        raise ValueError("History is missing required temporal feature columns.")
    base = history[columns].astype(float)
    if np.isinf(base.to_numpy()).any():
        raise ValueError("Infinite market features are not supported.")
    derived = {}
    for column in TEMPORAL_COLUMNS:
        values = base[column]
        for lag in LAGS:
            derived[f"{column}_lag_{lag}"] = values.shift(lag)
        past = values.shift(1)
        for window in WINDOWS:
            rolling = past.rolling(window, min_periods=1)
            derived[f"{column}_roll_mean_{window}"] = rolling.mean()
            derived[f"{column}_roll_std_{window}"] = rolling.std(ddof=1)
    return pd.concat([base, pd.DataFrame(derived, index=history.index)], axis=1).ffill()


class TemporalPreprocessor:
    """Learn fallback medians on training rows, reuse them for every future row."""

    def fit_transform(self, training_history: pd.DataFrame) -> pd.DataFrame:
        matrix = temporal_features(training_history)
        self.start_date_ = int(training_history["date_id"].iloc[0])
        self.train_end_date_ = int(training_history["date_id"].iloc[-1])
        self.columns_ = matrix.columns.tolist()
        self.medians_ = matrix.median().fillna(0.0)
        return matrix.fillna(self.medians_)

    def transform(self, history: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "medians_"):
            raise ValueError("Fit the preprocessor on training history before transforming.")
        matrix = temporal_features(history)
        if int(history["date_id"].iloc[0]) != self.start_date_:
            raise ValueError("Transform requires full feature history from the training start date.")
        if matrix.columns.tolist() != self.columns_:
            raise ValueError("Feature columns and their order must match training.")
        return matrix.fillna(self.medians_)


def baseline_model(*, n_jobs: int = 4, smoke: bool = False) -> LGBMRegressor:
    """Build the fixed author configuration; smoke mode only checks execution.

    The original final fit omits subsample_freq, so its default zero is preserved.
    No early stopping or validation-based estimator selection occurs in Stage 3.
    """
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")
    params = dict(AUTHOR_PARAMETERS)
    if smoke:
        params["n_estimators"] = 32
    return LGBMRegressor(**params, random_state=42, n_jobs=n_jobs,
                         deterministic=True, force_col_wise=True, verbosity=-1)


def binary_allocations(predictions) -> np.ndarray:
    """Preserve the author's scalar inference rule: positive -> 1, otherwise 0."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Predictions must be a nonempty finite one-dimensional vector.")
    return (values > 0).astype(float)
