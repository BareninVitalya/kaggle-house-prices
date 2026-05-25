"""
Глобальная конфигурация проекта kaggle-titanic.

Содержит:
- пути к данным и директориям для артефактов обучения;
- основные константы (seed, целевая переменная, число фолдов);
- списки признаков для моделирования;
- дефолтные гиперпараметры для используемых моделей;
- настройки поиска гиперпараметров и др.
"""

from pathlib import Path
from scipy.stats import loguniform

# ── Общие пути и базовые настройки ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

SEED = 42

TARGET_COL = 'SalePrice'

N_SPLITS = 5

NUMERIC_AS_CATEGORICAL_MAX_UNIQUE = 16

QUALITY_COLS = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
]

LOG_COLS = [
    "LotArea",
    "LotFrontage",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
]

# ── Признаки, которые показали себя шумными в экспериментах и исключаются из моделирования.
NOISE_FEATURES = [
    "Utilities", "Id", "RoofMatl", "Street",
    "LowQualFinSF", "BsmtHalfBath", "MiscVal",
    "GarageYrBlt", "GarageArea", "PoolArea", "PoolQC",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath", "BsmtFullBath"
]

IDX_TO_DROP = [
    934, 1298, 249, 313, 335, 706, 691, 1182, 297,
    523, 635, 1190, 1061, 581, 495, 583, 1328, 197,
    1337, 451, 1396, 30, 496 #
]

# ── Список числовых признаков, используемых в текущей конфигурации моделей.
NUM_FEATURES = [
    "MSSubClass",
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "GrLivArea",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "MoSold",
    "YrSold",
    "HasGarage",
    "HasMasVnr",
    "HasBsmt",
    "TotalSF",
    "TotalBath",
]

# ── Категориальные признаки, используемые в текущей конфигурации моделей
CAT_FEATURES = [
    "MSZoning",
    "Alley",
    "LotShape",
    "LandContour",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "CentralAir",
    "Electrical",
    "Functional",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
]


# ── Дефолтные параметры моделей (из экспериментов в ноутбуке) ─────────────────
DEFAULT_LOGREG_PARAMS = {
    "max_iter":    2000,
    "C":           0.5,
    "l1_ratio":    0.0,
    "solver":      "lbfgs",
    "random_state": SEED,
}

DEFAULT_RIDGE_PARAMS = {
    'alpha': 15.9,
    'fit_intercept': True,
    "random_state": SEED,
}

DEFAULT_KNN_PARAMS = {
    "n_neighbors": 7,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2,
}

DEFAULT_TREE_PARAMS = {
    "criterion": "gini",
    "max_depth": 4,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": None,
    "random_state": SEED,
}

DEFAULT_RF_PARAMS = {
    "n_estimators":    300,
    "criterion":       "gini",
    "max_depth":       5,
    "min_samples_split": 10,
    "min_samples_leaf":  4,
    "random_state":    SEED,
}

DEFAULT_RF_REG_PARAMS = {
    "n_estimators":    300,
    "criterion":       "squared_error",
    "max_depth":       5,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "random_state":    SEED,
}

DEFAULT_GB_PARAMS = {
    "n_estimators":    300,
    "learning_rate":   0.03,
    "max_depth":       2,
    "min_samples_leaf": 3,
    "random_state":    SEED,
}

DEFAULT_CATBOOST_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.01,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    'random_strength': 1.0,
    'bagging_temperature': 0.5,
    'border_count': 128,
    'boosting_type': "Plain",
    'bootstrap_type': "Bayesian",
    "random_seed":    SEED,
    'verbose':  False
}


DEFAULT_LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": SEED,
    "n_jobs": -1,
}

RANDOM_SEARCH_SPACE = {
    "logreg": {
        "model__C": loguniform(1e-3, 1e3),
        "model__solver": ["lbfgs", "liblinear"],
    },
    # сюда потом можно добавить "rf", "xgb" и т.д.
}

# ── OpenFE ────────────────────────────────────────────────────────────────────
USE_OPENFE = False

OPENFE_PARAMS = {
    "n_features": 30,
    "corr_threshold": 0.95,
    "greedy_threshold": 0.001,
    "use_ablation": True,
    "ablation_step": 5,
    "n_jobs": 1,
}

DEFAULT_DNN_PARAMS = {
    # Архитектура
    "hidden_dims":      [128, ],   # список размеров скрытых слоёв
    "activation":       "sigmoid",     # relu | leakyrelu | gelu | tanh | selu | elu
    "dropout":          0.0,        # 0.0 = Dropout выключен
    "batchnorm":        False,      # BatchNorm1d после каждого скрытого слоя

    # Обучение
    "optimizer":        "adam",     # adam | adamw | sgd
    "lr":               1e-3,
    "weight_decay":     0.0,        # L2-регуляризация для AdamW/SGD
    "batch_size":       32,
    "epochs":           60,

    # Scheduler (None = не использовать)
    "scheduler":        None,       # None | "cosine" | "step"
    "scheduler_params": {},         # доп. параметры для scheduler'а

    # Loss
    "loss_fn":          "mse",
}