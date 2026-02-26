# ================================================
# model.py
# ================================================
from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def get_base_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )


def tune_model(preprocessor, X_train, y_train, n_iter: int = 40, cv: int = 3):
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    if cv < 2:
        raise ValueError("cv must be >= 2")

    # imbalance ratio
    neg, pos = np.bincount(y_train)
    if pos == 0:
        raise ValueError("y_train contains no positive class examples.")
    scale_pos_weight = neg / pos

    base_model = get_base_model(scale_pos_weight)

    pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", base_model)])

    param_dist = {
        "clf__max_depth": randint(3, 10),
        "clf__min_child_weight": randint(1, 8),
        "clf__gamma": uniform(0, 5),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__learning_rate": uniform(0.01, 0.2),
        "clf__n_estimators": randint(200, 800),
        "clf__reg_alpha": uniform(0, 2),
        "clf__reg_lambda": uniform(0, 3),
        "clf__scale_pos_weight": uniform(max(0.1, scale_pos_weight - 1), 3),
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    random_search.fit(X_train, y_train)

    print("Best AUC:", random_search.best_score_)
    print("Best Params:", random_search.best_params_)

    return random_search.best_estimator_
