import pandas as pd
import numpy as np
from .config import NOISE_FEATURES, QUALITY_COLS, LOG_COLS, IDX_TO_DROP
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


class HousePricesFeaturesBase:

    def _fill_no(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols is None:
            df[cols] = df[cols].fillna("No")
        return df

    def _fill_zero(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols is None:
            df[cols] = df[cols].fillna(0)
        return df

    def _impute_basement(self, df):
        bsmt_cat_cols = [
            "BsmtQual", "BsmtCond", "BsmtExposure",
            "BsmtFinType1", "BsmtFinType2"
        ]

        bsmt_num_cols = [
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
            "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"
        ]
        df = self._fill_no(df, bsmt_cat_cols)
        df = self._fill_zero(df, bsmt_num_cols)
        return df

    def _impute_garage(self, df):
        garage_cat_cols = [
            "GarageType", "GarageFinish", "GarageQual", "GarageCond"
        ]

        garage_num_cols = [
            "GarageYrBlt", "GarageCars", "GarageArea"
        ]
        df = self._fill_no(df, garage_cat_cols)
        df = self._fill_zero(df, garage_num_cols)
        return df

    def _impute_masvnr(self, df):
        df = self._fill_no(df, ['MasVnrType'])
        df = self._fill_zero(df, ['MasVnrArea'])
        return df

    def _add_has_garage_feature(self, df):
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        return df

    def _add_has_basement_feature(self, df):
        df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
        return df

    def _add_has_masvnr_feature(self, df):
        df["HasMasVnr"] = (df["MasVnrArea"] > 0).astype(int)
        return df

    def _impute_other_cat_cols(self, df):
        none_cat_cols = [
            "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"
        ]
        df = self._fill_no(df, none_cat_cols)
        return df

    def _impute_lot_frontage(self, df):
        df["LotFrontage"] = df["LotFrontage"].fillna(
            df.groupby("Neighborhood")["LotFrontage"].transform("median")
        )
        return df

    def _lot_shape_transform(self, df):
        df["LotShape"] = df["LotShape"].replace({
            "IR2": "IR2_IR3",
            "IR3": "IR2_IR3"
        })
        return df

    def _add_total_sf(self, df):
        df["TotalSF"] = (df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"])
        return df

    def _add_total_bath(self, df):
        df["TotalBath"] = (
                df["FullBath"]
                + 0.5 * df["HalfBath"]
                + df["BsmtFullBath"]
                + 0.5 * df["BsmtHalfBath"]
        )
        return df

    def _cols_log_transform(self, df, num_cols):
        for col in num_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        return df

    def _cols_quality_transform(self, df, qua_cols):
        qual_map = {
            "Po": 1,
            "Fa": 2,
            "TA": 3,
            "Gd": 4,
            "Ex": 5
        }
        for col in qua_cols:
            if col in df.columns:
                df[col] = df[col].map(qual_map).fillna(0)
        return df

    def _drop(self, df, columns):
        return df.drop(columns=[c for c in columns if c in df.columns], errors="ignore")

    def _base_transform(self, df):
        """Шаги, общие для обеих задач."""
        df = df.copy()
        df = self._impute_basement(df)
        df = self._add_has_basement_feature(df)
        df = self._impute_garage(df)
        df = self._add_has_garage_feature(df)
        df = self._impute_masvnr(df)
        df = self._add_has_masvnr_feature(df)
        df = self._impute_other_cat_cols(df)
        df = self._impute_lot_frontage(df)
        df = self._lot_shape_transform(df)
        df = self._add_total_sf(df)
        df = self._add_total_bath(df)
        return df


class HousePricesFeatures(HousePricesFeaturesBase):
    """
    Препроцессинг для основной задачи: предсказание выживания.
    Использует внутреннюю модель (age_model) для заполнения Age.
    """

    def __init__(
        self,
        use_log_transform: bool = True,
        use_quality_map: bool = False,
    ):
        self.use_log_transform = use_log_transform
        self.use_quality_map = use_quality_map

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._base_transform(df)

        if self.use_log_transform:
            df = self._cols_log_transform(df, set(df.columns).intersection(LOG_COLS))

        if self.use_quality_map:
            df = self._cols_quality_transform(df, set(df.columns).intersection(QUALITY_COLS))

        cols = set(NOISE_FEATURES.copy())

        df = self._drop(df, cols)

        return df

# ──────────────────────────────────────────────
# Sklearn-совместимые трансформеры
# ──────────────────────────────────────────────

class HousePricesTransformerMixin(BaseEstimator, TransformerMixin):
    """Добавляет fit() — stateful-логики нет, просто возвращает self."""

    def fit(self, X, y=None):
        return self

class HousePricesTransformer(HousePricesFeatures, HousePricesTransformerMixin):
    """Sklearn Pipeline-совместимый трансформер для задачи Survived."""
    pass
