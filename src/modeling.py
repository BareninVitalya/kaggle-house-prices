from typing import Optional, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from .nn_model import HouseDNNRegressor
import numpy as np


from .config import (
    SEED,
    DEFAULT_LOGREG_PARAMS,
    TARGET_COL,
    NUM_FEATURES,
    CAT_FEATURES,
    DEFAULT_TREE_PARAMS,
    DEFAULT_KNN_PARAMS,
    DEFAULT_RF_PARAMS,
    DEFAULT_CATBOOST_PARAMS,
    DEFAULT_LGBM_PARAMS,
    DEFAULT_XGB_PARAMS,
    DEFAULT_DNN_PARAMS,
    DEFAULT_RF_REG_PARAMS,
    MODELS_DIR
)

from .features import HousePricesTransformer
from .data import load_train, load_test, save_processed

def get_transformer(
        use_log_transform: bool = True,
        use_quality_map: bool = False,
        transform_off: bool = False
) -> TransformerMixin:
    if transform_off:
        return FunctionTransformer(lambda x: x)
    else:
        return HousePricesTransformer(
                use_log_transform=use_log_transform,
                use_quality_map=use_quality_map,
            )


def build_matual_info_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Строим общий препроцессор для числовых и категориальных признаков."""

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    return preprocessor

def build_preprocessor(X: Optional[pd.DataFrame] = None, transform_off: bool = False, sparse_output: bool = True) -> ColumnTransformer:
    """Строим общий препроцессор для числовых и категориальных признаков."""
    if not transform_off and X is not None:
        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(exclude="number").columns
    else:
        num_cols = NUM_FEATURES
        cat_cols = CAT_FEATURES

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    return preprocessor

def build_catboost_preprocessor(X: Optional[pd.DataFrame] = None, transform_off: bool = False) -> ColumnTransformer:
    """CatBoost препроцессор"""
    if transform_off and X is not None:
        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(exclude="number").columns
    else:
        num_cols = NUM_FEATURES
        cat_cols = CAT_FEATURES

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        verbose_feature_names_out=False
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor


def build_logreg_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_fare=True,
            use_age_bins=True,
            use_fare_bins=True,
            use_pclass_sex=True,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_LOGREG_PARAMS.copy()

    if params:
        base_params.update(params)


    clf = LogisticRegression(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_linreg_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = {}

    if params:
        base_params.update(params)


    clf = LinearRegression(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_ridge_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = {}

    if params:
        base_params.update(params)


    clf = Ridge(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_lasso_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = {}

    if params:
        base_params.update(params)


    clf = Lasso(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_elasticnet_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = {}

    if params:
        base_params.update(params)


    clf = ElasticNet(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_knn_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = {}

    if params:
        base_params.update(params)

    clf = KNeighborsRegressor(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_tree_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = {}

    if params:
        base_params.update(params)

    clf = DecisionTreeRegressor(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_rf_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = {}

    if params:
        base_params.update(params)

    clf = RandomForestRegressor(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_catboost_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_catboost_preprocessor(X, transform_off=transform_off)
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_catboost_preprocessor()
        cat_cols = CAT_FEATURES
        num_cols = list(NUM_FEATURES)

    base_params = DEFAULT_CATBOOST_PARAMS.copy()

    if params:
        base_params.update(params)

    cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    base_params['cat_features'] = cat_idx

    clf = CatBoostRegressor(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_rf_reg_model(X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_RF_REG_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = RandomForestRegressor(**base_params)
    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_lgbm_model(
    X: pd.DataFrame,
    params: Optional[Dict] = None,
    transform_off: bool = False
) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = DEFAULT_LGBM_PARAMS.copy()

    if params:
        base_params.update(params)

    clf = LGBMRegressor(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])

def build_xgb_model(
    X: pd.DataFrame,
    params: Optional[Dict] = None,
    transform_off: bool = False
) -> Pipeline:

    if transform_off:
        fe = get_transformer(transform_off=transform_off)
        pre = build_preprocessor(X)
    else:
        fe = get_transformer(
            use_log_transform=True,
            use_quality_map= False,
            transform_off=transform_off
        )
        pre = build_preprocessor()

    base_params = {}

    if params:
        base_params.update(params)

    clf = XGBRegressor(**base_params)

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])



def build_dnn_model(
    X: pd.DataFrame,
    params: Optional[Dict] = None,
    transform_off: bool = False
) -> Pipeline:
    """
    Создаёт TitanicDNNClassifier на основе DEFAULT_DNN_PARAMS + overrides из params.

    DNN ожидает числовой вход — используй transform_off=True в quick_experiment,
    чтобы передавать уже подготовленный X напрямую.

    Пример:
        model = build_model("dnn", X_proc, params={"hidden_dims": [128, 64], "dropout": 0.2})
        quick_experiment(X_proc, y, model_name="dnn", transform_off=True)
    """
    cfg = DEFAULT_DNN_PARAMS.copy()
    if params:
        cfg.update(params)

    fe = get_transformer(
        use_log_transform=True,
        use_quality_map=False,
        transform_off=transform_off
    )
    pre = build_preprocessor(sparse_output=False)

    clf = HouseDNNRegressor(
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        batchnorm=cfg["batchnorm"],
        optimizer=cfg["optimizer"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        scheduler=cfg["scheduler"],
        scheduler_params=cfg["scheduler_params"],
        loss_fn=cfg["loss_fn"],
        random_state=SEED,
    )

    return Pipeline([("feat", fe), ("prep", pre), ("model", clf)])


def build_model(name: str, X: pd.DataFrame, params: Optional[Dict] = None, transform_off: bool = False) -> Pipeline:
    """Фабрика моделей по имени."""
    name = name.lower()
    if name in ("logreg", "lr", "logistic"):
        return build_logreg_model(X, params, transform_off)
    if name in ('linreg'):
        return build_linreg_model(X, params, transform_off)
    if name in ('ridge'):
        return build_ridge_model(X, params, transform_off)
    if name in ('lasso'):
        return build_ridge_model(X, params, transform_off)
    if name in ('elasticnet'):
        return build_elasticnet_model(X, params, transform_off)
    if name in ("knn", "kneighbors", "k_neighbors"):
        return build_knn_model(X, params, transform_off)
    if name in ("tree", "dt", "decision_tree"):
        return build_tree_model(X, params, transform_off)
    if name in ("rf", "random_forest"):
        return build_rf_model(X, params, transform_off)
    if name in ("catboost", "cat"):
        return build_catboost_model(X, params, transform_off)
    if name in ("lgbm", "lightgbm"):
        return build_lgbm_model(X, params, transform_off)
    if name in ("xgb", "xgboost"):
        return build_xgb_model(X, params, transform_off)
    if name in ("dnn", "mlp", "nn"):
        return build_dnn_model(X, params, transform_off)
    if name in ("rf_reg"):
        return build_rf_reg_model(X, params, transform_off)
    raise ValueError(f"Unknown model name: {name}")

def train_model(model_name: str = 'logreg', params: Optional[Dict] = None, train_data: pd.DataFrame = None, transform_off: bool = False):
    if transform_off and train_data is not None:
        df_train = train_data
    else:
        df_train = load_train()
        transform_off = False

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = np.log1p(df_train[TARGET_COL])

    model = build_model(model_name, X_train, params, transform_off=transform_off)
    model.fit(X_train, y_train)

    return model

def predict_and_save_titanic(model, test_data: pd.DataFrame, file_name: str = "submission"):

    submission = pd.DataFrame({
        "Id": test_data["Id"],
        "SalePrice": np.expm1(model.predict(test_data)),
    })

    save_processed(submission, f"{file_name}.csv")

def predict_and_save_titanic2(model, ids_data: pd.DataFrame, test_data: pd.DataFrame, file_name: str = "submission"):

    submission = pd.DataFrame({
        "Id": ids_data["Id"],
        "SalePrice": np.expm1(model.predict(test_data)),
    })

    save_processed(submission, f"{file_name}.csv")