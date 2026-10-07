import numpy as np

from cardiorisk import config as C
from cardiorisk.features import build_preprocessor, split_features_target


def test_preprocessor_output_has_no_missing_values(clean_df):
    X, _ = split_features_target(clean_df)
    X.iloc[0, :] = np.nan  # a fully empty row must still be imputed
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()


def test_preprocessor_expected_width(clean_df):
    X, _ = split_features_target(clean_df)
    pre = build_preprocessor().fit(X)
    n_onehot = sum(X[c].nunique() for c in C.CATEGORICAL_FEATURES)
    expected = len(C.NUMERIC_FEATURES) + len(C.BINARY_FEATURES) + n_onehot
    assert pre.transform(X).shape == (len(X), expected)


def test_numeric_features_are_standardised(clean_df):
    X, _ = split_features_target(clean_df)
    Xt = build_preprocessor().fit_transform(X)
    num = Xt[:, :len(C.NUMERIC_FEATURES)]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(num.std(axis=0), 1, atol=1e-6)


def test_unseen_category_does_not_crash(clean_df):
    X, _ = split_features_target(clean_df)
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["thal"] = 5.0  # never seen in training
    assert pre.transform(row).shape[0] == 1


def test_split_features_target_types(clean_df):
    X, y = split_features_target(clean_df)
    assert list(X.columns) == C.FEATURES
    assert all(dt == float for dt in X.dtypes)
    assert y.dtype.kind == "i"
