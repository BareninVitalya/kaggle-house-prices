# tuning.py
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.base import clone
from scipy.stats import loguniform
from .tuning_objectives import (
    LogregObjective,
    RidgeObjective,
    LassoObjective,
    ElasticNetObjective,
    KNNObjective,
    TreeObjective,
    RFObjective,
    CatBoostObjective,
    XGBObjective,
    LGBMObjective
)
import optuna
from .logging_utils import log_experiment
from .modeling import build_model
from .config import RANDOM_SEARCH_SPACE, SEED, N_SPLITS
from .evaluate import _get_cv

def tune_with_random_search(
    model_name: str,
    X,
    y,
    base_params: dict | None = None,
    n_iter: int = 30,
    scoring: str = "accuracy",
    random_state: int = SEED,
):
    base_model = build_model(model_name, X, params=base_params, transform_off=True)

    cv = _get_cv(n_splits=N_SPLITS, shuffle=True, seed=random_state, scoring=scoring)

    param_distributions = RANDOM_SEARCH_SPACE[model_name]

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=1,
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    best_params = search.best_params_
    return best_model, best_params, search

def tune_with_optuna(
    model_name: str,
    X,
    y,
    base_params: dict | None = None,
    n_trials: int = 50,
    scoring: str = "accuracy",
    random_state: int = SEED,
    transform_off: bool = True,
    log_trials: bool = True,
    logfile: str = "optuna.log",
):
    if model_name == 'knn':
        objective_class = KNNObjective
    elif model_name == "tree":
        objective_class = TreeObjective
    elif model_name in ("rf", "random_forest", "randomforest"):
        objective_class = RFObjective
    elif model_name in ("catboost", "cat"):
        objective_class = CatBoostObjective
    elif model_name in ("xgboost", "xgb"):
        objective_class = XGBObjective
    elif model_name == "lgbm":
        objective_class = LGBMObjective
    elif model_name == "ridge":
        objective_class = RidgeObjective
    elif model_name == "lasso":
        objective_class = LassoObjective
    elif model_name == "elasticnet":
        objective_class = ElasticNetObjective
    # elif model_name == 'logreg':
    else:
        base_params = base_params or {}
        base_params["penalty"] = "l2" # костыль который нужен чтобы работало на версии 1.8 sklearn
        objective_class = LogregObjective


    base_model = build_model(model_name, X, params=base_params, transform_off=transform_off)

    cv = _get_cv(n_splits=N_SPLITS, shuffle=True, seed=random_state, scoring=scoring)

    objective = objective_class(base_model, X, y, cv, scoring=scoring, log_trials=log_trials)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    best_model = clone(base_model)

    best_model.set_params(**study.best_trial.params)
    best_model.fit(X, y)

    log_experiment(
        name=f"optuna_{model_name}_best",
        score=float(study.best_value),
        std=0.0,
        params=study.best_trial.params,
        col_names=X.columns.tolist() if hasattr(X, "columns") else [],
        logfile=logfile,
        print_log=False,
    )

    return best_model, study


def tune_dnn_with_grid_search(
    X,
    y,
    model_name,
    base_params: dict | None = None,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = SEED,
    transform_off: bool = False,
    param_grid: dict | None = None,
    verbose: bool = True,
):
    """
    Grid search для DNN:
    - перебирает архитектуру сети (число/размер слоев)
    - одновременно перебирает параметры регуляризации

    Ожидается, что build_model("dnn", ...) возвращает sklearn-compatible estimator.
    """

    base_params = base_params or {}

    base_model = build_model(
        model_name,
        X,
        params=base_params,
        transform_off=transform_off,
    )

    cv = _get_cv(
        n_splits=N_SPLITS,
        shuffle=True,
        seed=random_state,
        scoring=scoring,
    )

    if param_grid is None:
        param_grid = {
            "model__hidden_dims": [
                (128,),
                (256,),
                (128, 64),
                (256, 128),
                (256, 128, 64),
            ],

            "model__dropout": [0.0, 0.1, 0.2, 0.3],
            "model__weight_decay": [0.0, 1e-5, 1e-4, 1e-3],

        }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=verbose,
    )

    search.fit(X, y)

    best_model = search.best_estimator_
    best_params = search.best_params_

    log_experiment(
        name=f"gridsearch_{model_name}_best",
        score=float(search.best_score_),
        std=0.0,
        params=best_params,
        col_names=X.columns.tolist() if hasattr(X, "columns") else [],
        logfile=f"gridsearch_{model_name}.log",
        print_log=False,
    )

    return best_model, best_params, search