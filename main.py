from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from src.data import load_train, load_test, save_submission
from src.config import (
    TARGET_COL,
    N_SPLITS,
    SEED,
    DEFAULT_DNN_PARAMS,
    DEFAULT_CATBOOST_PARAMS,
    DEFAULT_RIDGE_PARAMS
)
from src.evaluate import cv_scores
from src.ensemble import SoftEnsemble


def main():
    df = load_train()
    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    y_log = np.log1p(y)

    model_names = ["ridge", "cat", "dnn"]

    model_params = {
        "ridge": DEFAULT_RIDGE_PARAMS,
        "cat": DEFAULT_CATBOOST_PARAMS,
        "dnn": DEFAULT_DNN_PARAMS,
    }

    ensemble = SoftEnsemble(
        model_names=model_names,
        model_params=model_params,
        weights={
            "ridge": 1.0,
            "cat": 2.0,
            "dnn": 1.5,
        },
        transform_off=False,
    )

    val_mean, val_std, val_scores, train_data = cv_scores(
        model=ensemble,
        X=X_raw,
        y=y_log,
        n_splits=N_SPLITS,
        seed=SEED,
        scoring="neg_root_mean_squared_error",
        return_train_score=False,
    )
    print(
        f"CV SoftEnsemble: "
        f"mean={val_mean} "
        f"std={val_std} "
        f"gap={float(train_data - val_std)}"
    )

    print(f'Fit and predict ...')
    ensemble.fit_build(X_raw, y_log)
    df_submission = ensemble.predict_test(load_test())

    save_submission(df_submission)


if __name__ == "__main__":
    main()