from typing import Dict, Callable, Tuple, Union, Callable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import root_mean_squared_error
from sklearn.base import clone
from catboost import CatBoostRegressor
from .config import SEED, N_SPLITS, CAT_FEATURES


def _get_cv(
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    shuffle: bool =True,
    scoring: str = 'accuracy'
):
    if scoring in ['neg_median_absolute_error', 'neg_root_mean_squared_error', ]:
        cv_type='kfold'
    else:
        cv_type='stratified'

    if cv_type == "kfold":
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed,
        )
    elif cv_type == "stratified":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed,
        )
    else:
        raise ValueError(
            f"Unknown cv_type='{cv_type}'. Use 'kfold' or 'stratified'."
        )

def cv_scores(
    model,
    X,
    y,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    scoring: str = "accuracy",
    return_train_score: bool = False
) -> Tuple[float, float, np.ndarray, Union[float, np.ndarray]]:
    """Возвращает mean, std, и массив score'ов по StratifiedKFold, как в ноутбуке.[file:1]"""

    cv = _get_cv(n_splits=n_splits, shuffle=True, seed=seed, scoring=scoring)

    result = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )

    val_scores = result["test_score"]
    val_mean   = float(val_scores.mean())
    val_std    = float(val_scores.std())
    train_data = float(result["train_score"].mean()) if not return_train_score else result["train_score"]

    return val_mean, val_std, val_scores, train_data

def cv_scores_with_fit_params(
    factory,
    model_name,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Callable = root_mean_squared_error,
    metric_name: str = "RMSE",
    factory_kwargs: Optional[Dict] = None,
    return_train_score=False,
    transform_off=True,
    **fit_params,
) -> Tuple[float, float, np.ndarray, Union[float, np.ndarray]]:
    """
    Универсальная кросс-валидация, совместимая с build_catboost_model,
    build_model и любыми другими фабриками из train_pipeline.
    """
    factory_kwargs = factory_kwargs or {}
    val_scores = []
    train_scores = []

    cv = _get_cv(n_splits=N_SPLITS, shuffle=True, seed=SEED, scoring=metric_name)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model_clone = factory(model_name, X_train, transform_off=transform_off)

        model_clone.fit(X_train, y_train, **fit_params)

        y_pred_val = model_clone.predict(X_valid)
        y_pred_tr  = model_clone.predict(X_train)

        val_score = scoring(y_valid, y_pred_val)
        train_score = scoring(y_train, y_pred_tr)

        val_scores.append(val_score)
        train_scores.append(train_score)

    val_scores = np.array(val_scores)
    train_scores = np.array(train_scores)

    val_mean = float(val_scores.mean())
    val_std  = float(val_scores.std())

    train_data = float(train_scores.mean()) if not return_train_score else train_scores

    return val_mean, val_std, val_scores, train_data


def compare_models(
    builders: Dict[str, Callable[[pd.DataFrame], object]],
    X,
    y,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Обгоняет список моделей и возвращает таблицу с их mean/std score."""
    rows = []
    for name, builder in builders.items():
        model = builder(X)
        mean, std, _, _ = cv_scores(model, X, y, n_splits=n_splits, seed=seed, scoring=scoring)
        rows.append({"model": name, "mean": mean, "std": std})
    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return df