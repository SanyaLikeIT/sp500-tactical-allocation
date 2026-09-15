"""Chronological splits with a reserved holdout and explicit label-availability gap."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


HOLDOUT_SIZE = 180
N_SPLITS = 5
VALIDATION_SIZE = 180
GAP = 1
TARGET = "market_forward_excess_returns"
SCORING_COLUMNS = ("forward_returns", "risk_free_rate")


@dataclass(frozen=True)
class Fold:
    """Zero-based positional indices into the original chronological dataset."""

    train: np.ndarray
    validation: np.ndarray


def validate_dates(date_ids) -> np.ndarray:
    dates = np.asarray(date_ids)
    if dates.ndim != 1 or dates.size == 0 or dates.dtype.kind not in "iu":
        raise ValueError("date_id must be a nonempty one-dimensional integer sequence.")
    if not np.all(np.diff(dates.astype(np.int64)) == 1):
        raise ValueError("date_id must be ordered, unique, and consecutive; do not shuffle rows.")
    return dates


def partition_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent development/holdout copies without inspecting targets."""
    if not data.columns.is_unique or "date_id" not in data:
        raise ValueError("Data must have unique columns including date_id.")
    validate_dates(data["date_id"])
    if len(data) <= HOLDOUT_SIZE:
        raise ValueError("The dataset must have more rows than the final holdout.")
    return data.iloc[:-HOLDOUT_SIZE].copy(), data.iloc[-HOLDOUT_SIZE:].copy()


def development_folds(date_ids) -> tuple[Fold, ...]:
    """Generate five development folds; final 180 rows are never passed to CV."""
    dates = validate_dates(date_ids)
    development_size = len(dates) - HOLDOUT_SIZE
    if development_size <= N_SPLITS * VALIDATION_SIZE + GAP:
        raise ValueError("Insufficient development history for five folds and a one-row gap.")
    splitter = TimeSeriesSplit(n_splits=N_SPLITS, test_size=VALIDATION_SIZE, gap=GAP)
    return tuple(Fold(train, validation)
                 for train, validation in splitter.split(np.arange(development_size)))


def final_holdout_fold(date_ids) -> Fold:
    """Final evaluation indices; call only after configuration selection is frozen."""
    dates = validate_dates(date_ids)
    start = len(dates) - HOLDOUT_SIZE
    if start <= GAP:
        raise ValueError("Insufficient training history before the final holdout.")
    return Fold(np.arange(start - GAP), np.arange(start, len(dates)))


def available_training_indices(prediction_index: int, max_train_rows: int | None = None) -> np.ndarray:
    """At row t, allow labels through t-2, preserving the one-row gap.

    This only encodes availability, not permission to tune on holdout labels.
    Online holdout evaluation may refit frozen configurations on observed history.
    """
    if isinstance(prediction_index, bool) or not isinstance(prediction_index, (int, np.integer)):
        raise ValueError("prediction_index must be an integer.")
    if prediction_index <= GAP:
        raise ValueError("No training labels are available with the required gap.")
    if max_train_rows is not None and (
        isinstance(max_train_rows, bool)
        or not isinstance(max_train_rows, (int, np.integer)) or max_train_rows <= 0
    ):
        raise ValueError("max_train_rows must be a positive integer.")
    stop = prediction_index - GAP
    start = 0 if max_train_rows is None else max(0, stop - max_train_rows)
    return np.arange(start, stop)


def market_feature_columns(data: pd.DataFrame) -> list[str]:
    """Select original market inputs, excluding IDs, targets, and realized returns."""
    if not data.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed.")
    return [c for c in data.columns
            if isinstance(c, str) and len(c) > 1 and c[0] in "DEIMPSV" and c[1:].isdigit()]
