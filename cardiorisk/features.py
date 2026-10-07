"""Feature engineering: one sklearn ColumnTransformer used by every model.

Because the transformer is part of the saved Pipeline, the exact same imputation,
scaling and encoding learned during training is applied at prediction time.
Only stock sklearn classes are used, so the pickled model can be loaded by the
API without importing any custom code.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cardiorisk import config as C


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # robust to skew (chol, oldpeak)
        ("scale", StandardScaler()),                      # needed by LR / SVM, harmless for RF
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),  # already 0/1, no scaling
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # codes like cp=1..4 are labels, not quantities -> one-hot.
        # handle_unknown='ignore' means an unseen code becomes all-zeros, not a crash.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, C.NUMERIC_FEATURES),
            ("bin", binary, C.BINARY_FEATURES),
            ("cat", categorical, C.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X as float (same dtype the API sends), y as int."""
    X = df[C.FEATURES].astype(float)
    y = df[C.TARGET].astype(int)
    return X, y
