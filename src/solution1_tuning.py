"""Development-only regularization search for the unchanged Solution 1 architecture."""

import numpy as np
from lightgbm import LGBMRegressor, early_stopping

from src.metrics import evaluate_predictions, strategy_returns
from src.solution1 import TemporalPreprocessor, binary_allocations
from src.validation import Fold, TARGET


SEED = 42
PATIENCE = 50
SEARCH_SPACE = {
    "n_estimators": [300, 2000], "learning_rate": [0.01, 0.07, "log"],
    "max_depth": [3, 8], "num_leaves": [8, "min(128, 2**max_depth)"],
    "min_child_samples": [20, 300], "min_split_gain": [0.0, 0.001],
    "reg_alpha": [0.00001, 0.1, "log"], "reg_lambda": [0.001, 10.0, "log"],
    "subsample": [0.6, 1.0], "colsample_bytree": [0.6, 1.0],
}


def suggest_parameters(trial) -> dict:
    depth = trial.suggest_int("max_depth", 3, 8)
    return {
        "max_depth": depth, "num_leaves": trial.suggest_int("num_leaves", 8, min(128, 2**depth)),
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate": trial.suggest_float("learning_rate", .01, .07, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, .001),
        "reg_alpha": trial.suggest_float("reg_alpha", .00001, .1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", .001, 10, log=True),
        "subsample": trial.suggest_float("subsample", .6, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", .6, 1),
    }


def tuned_model(parameters: dict, *, n_jobs=4, iterations=None):
    params = dict(parameters)
    if iterations is not None:
        params["n_estimators"] = int(iterations)
    if params["num_leaves"] > 2**params["max_depth"]:
        raise ValueError("num_leaves must not exceed 2**max_depth.")
    if params["n_estimators"] < 1 or n_jobs < 1:
        raise ValueError("Estimator count and CPU threads must be positive.")
    return LGBMRegressor(**params, objective="regression", random_state=SEED,
                         subsample_freq=1, deterministic=True, force_col_wise=True,
                         n_jobs=n_jobs, verbosity=-1)


def stopping_split(training_size: int) -> Fold:
    """Reserve the last 180 outer-training rows for stopping, with a one-row gap."""
    start = training_size - 180
    if start <= 1:
        raise ValueError("Not enough history for inner early stopping.")
    return Fold(np.arange(start - 1), np.arange(start, training_size))


def prepare_fold(development, fold: Fold) -> dict:
    """Fit separate inner/outer imputers; this function never accepts holdout rows."""
    if fold.validation[-1] >= len(development) or fold.train[0] != 0:
        raise ValueError("Expected an expanding fold contained in development data.")
    if fold.train[-1] + 2 != fold.validation[0]:
        raise ValueError("A one-row outer gap is required.")
    training = development.iloc[fold.train]
    inner = stopping_split(len(training))
    inner_processor = TemporalPreprocessor()
    inner_X = inner_processor.fit_transform(training.iloc[inner.train])
    stopping_X = inner_processor.transform(training).iloc[inner.validation]
    outer_processor = TemporalPreprocessor()
    outer_X = outer_processor.fit_transform(training)
    validation_X = outer_processor.transform(development.iloc[:fold.validation[-1] + 1]).iloc[fold.validation]
    target = training[TARGET].to_numpy()
    if not np.isfinite(target).all():
        raise ValueError("Training targets must be finite and are never imputed.")
    return {
        "inner_X": inner_X, "inner_y": target[inner.train],
        "stopping_X": stopping_X, "stopping_y": target[inner.validation],
        "outer_X": outer_X, "outer_y": target, "validation_X": validation_X,
        "evaluation": development.iloc[fold.validation].copy(),
        "bounds": {"train_end": int(training.date_id.iloc[-1]),
                   "inner_train_end": int(training.date_id.iloc[inner.train[-1]]),
                   "stopping_start": int(training.date_id.iloc[inner.validation[0]]),
                   "stopping_end": int(training.date_id.iloc[inner.validation[-1]]),
                   "validation_start": int(development.date_id.iloc[fold.validation[0]]),
                   "validation_end": int(development.date_id.iloc[fold.validation[-1]])},
    }


def fit_fold(prepared: dict, parameters: dict, *, n_jobs=4) -> tuple[dict, object]:
    """Select tree count on inner data, then refit all outer training rows."""
    stopping_model = tuned_model(parameters, n_jobs=n_jobs)
    stopping_model.fit(prepared["inner_X"], prepared["inner_y"],
                       eval_X=prepared["stopping_X"], eval_y=prepared["stopping_y"],
                       eval_metric="rmse", callbacks=[early_stopping(PATIENCE, verbose=False)])
    iterations = max(1, stopping_model.best_iteration_)
    model = tuned_model(parameters, n_jobs=n_jobs, iterations=iterations)
    model.fit(prepared["outer_X"], prepared["outer_y"])
    predictions = model.predict(prepared["validation_X"])
    frame = prepared["evaluation"]
    allocations = binary_allocations(predictions)
    metrics = evaluate_predictions(targets=frame[TARGET], predictions=predictions,
                                   forward_returns=frame.forward_returns,
                                   risk_free_rate=frame.risk_free_rate, allocations=allocations)
    prediction_frame = frame[["date_id", TARGET, "forward_returns", "risk_free_rate"]].rename(columns={TARGET: "target"}).copy()
    prediction_frame["prediction"] = predictions
    prediction_frame["allocation"] = allocations
    prediction_frame["strategy_return"] = strategy_returns(frame.forward_returns, frame.risk_free_rate, allocations)
    return {**metrics, **prepared["bounds"], "best_iteration": iterations,
            "prediction_constant": bool(np.all(predictions == predictions[0])),
            "allocation_mean": float(allocations.mean())}, prediction_frame


def final_iterations(fold_iterations) -> int:
    """Predeclared deployment rule: integer median of selected trial's inner tree counts."""
    values = np.asarray(fold_iterations)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 1).any():
        raise ValueError("Inner stopping iterations must be positive and finite.")
    return int(np.median(values))
