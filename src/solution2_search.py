"""Reproduce Baseline 2's MSE searches with train-only nested preprocessing."""

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.solution2 import EnsemblePreprocessor, component_model, select_features
from src.validation import TARGET


def suggest(trial, kind, current=None):
    if kind == "enet":
        return {"alpha": trial.suggest_float("alpha", .001 if current is None else current["alpha"]*.5,
                                             .1 if current is None else current["alpha"]*2, log=current is None),
                "l1_ratio": trial.suggest_float("l1_ratio", .1 if current is None else max(.1, current["l1_ratio"]-.1),
                                                .9 if current is None else min(.9, current["l1_ratio"]+.1))}
    upper_trees = 350 if kind == "xgb" else 300
    if current is None:
        return {"max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", .01, .1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 150, upper_trees),
                "reg_alpha": trial.suggest_float("reg_alpha", .001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", .001, 10, log=True)}
    return {"max_depth": trial.suggest_int("max_depth", max(3, current["max_depth"]-1), min(8, current["max_depth"]+1)),
            "learning_rate": trial.suggest_float("learning_rate", current["learning_rate"]*.8, current["learning_rate"]*1.2),
            "n_estimators": trial.suggest_int("n_estimators", max(100, current["n_estimators"]-50), min(upper_trees+50, current["n_estimators"]+50)),
            "reg_alpha": trial.suggest_float("reg_alpha", current["reg_alpha"]*.5, current["reg_alpha"]*2),
            "reg_lambda": trial.suggest_float("reg_lambda", current["reg_lambda"]*.5, current["reg_lambda"]*2)}


def author_search(window, *, current=None, seed=42, n_jobs=1, smoke=False):
    """Original budgets/objective; fresh inner selection prevents supervised leakage."""
    splits = TimeSeriesSplit(n_splits=5 if current is None else 3, gap=1)
    cached, boundaries = [], []
    for train, validation in splits.split(window):
        processor = EnsemblePreprocessor()
        X_train = processor.fit_transform(window.iloc[train])
        selected = select_features(X_train, window.iloc[train][TARGET], n_jobs=n_jobs)
        X_validation = processor.transform(window.iloc[:validation[-1]+1]).iloc[validation]
        scaler = StandardScaler().fit(X_train[selected])
        cached.append((X_train[selected], X_validation[selected],
                       pd.DataFrame(scaler.transform(X_train[selected]), columns=selected),
                       pd.DataFrame(scaler.transform(X_validation[selected]), columns=selected),
                       window.iloc[train][TARGET].to_numpy(), window.iloc[validation][TARGET].to_numpy()))
        boundaries.append({"train_end": int(window.date_id.iloc[train[-1]]),
                           "validation_start": int(window.date_id.iloc[validation[0]]),
                           "validation_end": int(window.date_id.iloc[validation[-1]]), "selected_features": selected})
    best, reports = {}, {}
    for offset, kind in enumerate(("xgb", "lgb", "enet")):
        trials = (20 if kind == "enet" else 30) if current is None else 10
        timeout = (100 if kind == "enet" else 300) if current is None else 15
        budget = 2 if smoke else trials
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed+offset))

        def objective(trial):
            parameters = suggest(trial, kind, None if current is None else current[kind])
            errors = []
            for raw_train, raw_validation, scaled_train, scaled_validation, y_train, y_validation in cached:
                X_train, X_validation = (scaled_train, scaled_validation) if kind == "enet" else (raw_train, raw_validation)
                model = component_model(kind, parameters, n_jobs=n_jobs)
                model.fit(X_train, y_train)
                errors.append(float(mean_squared_error(y_validation, model.predict(X_validation))))
            trial.set_user_attr("fold_mse", errors)
            return float(np.mean(errors))

        study.optimize(objective, n_trials=budget, timeout=timeout)
        best[kind] = study.best_params
        reports[kind] = {"seed": seed+offset, "requested_trials": budget, "source_trial_budget": trials,
                         "timeout_seconds": timeout, "completed_trials": len(study.trials),
                         "best_mse": study.best_value, "best_parameters": study.best_params,
                         "trials": [{"number": t.number, "value": t.value, "state": t.state.name,
                                     "parameters": t.params, "fold_mse": t.user_attrs["fold_mse"]} for t in study.trials]}
    return best, {"models": reports, "inner_boundaries": boundaries}
