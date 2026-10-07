"""Shared fixtures. Tests use a synthetic dataset shaped exactly like the UCI file,
so the suite runs offline (CI never depends on the UCI website being up)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cardiorisk import config as C
from cardiorisk.features import split_features_target


def make_raw_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows in raw UCI format: codes as floats, 'num' in 0..4,
    with a real signal so models can beat chance."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    df = pd.DataFrame({
        "age": np.clip(rng.normal(54, 9, n) + 3 * risk, 29, 77).round(),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": np.where(risk > 0.3, 4, rng.integers(1, 4, n)).astype(float),
        "trestbps": np.clip(rng.normal(131, 17, n), 94, 200).round(),
        "chol": np.clip(rng.normal(246, 50, n), 126, 564).round(),
        "fbs": (rng.random(n) < 0.15).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": np.clip(rng.normal(150, 20, n) - 12 * risk, 71, 202).round(),
        "exang": (risk + rng.normal(0, 0.8, n) > 0.5).astype(float),
        "oldpeak": np.clip(rng.gamma(1.2, 0.9, n) + 0.6 * risk, 0, 6.2).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": np.clip(np.round(risk + 1 + rng.normal(0, 0.6, n)), 0, 3),
        "thal": rng.choice([3.0, 6.0, 7.0], n, p=[0.55, 0.06, 0.39]),
    })
    sick = risk + rng.normal(0, 0.7, n) > 0.1
    df["num"] = np.where(sick, rng.integers(1, 5, n), 0).astype(float)
    return df[C.RAW_COLUMNS]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    from cardiorisk.data import clean_data
    return clean_data(raw_df)


@pytest.fixture
def raw_file(tmp_path, raw_df):
    """The synthetic data written exactly like the UCI file: no header, '?' for missing."""
    df = raw_df.astype(object)
    df.loc[3, "ca"] = "?"
    df.loc[7, "thal"] = "?"
    path = tmp_path / "processed.cleveland.data"
    df.to_csv(path, header=False, index=False)
    return path


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
    """A small but real Pipeline(preprocess + LR) saved with joblib, for API tests."""
    import json

    import joblib
    from cardiorisk.data import clean_data
    from cardiorisk.train import build_pipeline

    X, y = split_features_target(clean_data(make_raw_frame(seed=1)))
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    out = tmp_path_factory.mktemp("model")
    path = out / "model.joblib"
    joblib.dump(model, path)
    (out / "model_metadata.json").write_text(json.dumps({
        "model_name": "test_lr", "model_version": "test", "decision_threshold": 0.5}))
    return path
