"""End-to-end test of run_training(): split -> tune -> compare -> gate -> save.

Runs without MLflow (track=False), with a tiny grid, and with every output path redirected
to a temporary folder, so it never touches the real models/ or reports/ directories."""
import json

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from cardiorisk import config as C
from cardiorisk.train import QualityGateError, positive_proba, run_training

TINY_SPECS = {
    "logistic_regression": (LogisticRegression(solver="liblinear"), {"clf__C": [0.1, 1.0]}),
    "random_forest": (RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1),
                      {"clf__max_depth": [3]}),
}


@pytest.fixture
def isolated(tmp_path, monkeypatch, clean_df):
    """Point every output path at tmp_path and write the synthetic data there."""
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(C, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(C, "METADATA_PATH", tmp_path / "models" / "model_metadata.json")
    monkeypatch.setattr(C, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(C, "FIGURES_DIR", tmp_path / "reports" / "figures")
    data_path = tmp_path / "heart_clean.csv"
    clean_df.to_csv(data_path, index=False)
    return data_path


def test_run_training_end_to_end(isolated, clean_df):
    summary = run_training(data_path=isolated, min_test_auc=0.5, track=False, specs=TINY_SPECS)

    assert summary["selected_model"] in TINY_SPECS
    assert set(summary["comparison"]["model"]) == set(TINY_SPECS)

    # the packaged model loads and predicts from raw feature values
    model = joblib.load(C.MODEL_PATH)
    p = positive_proba(model, clean_df[C.FEATURES].astype(float).head(5))
    assert ((p >= 0) & (p <= 1)).all()

    meta = json.loads(C.METADATA_PATH.read_text())
    assert meta["model_name"] == summary["selected_model"]
    assert meta["features"] == C.FEATURES
    assert meta["tracked_with_mlflow"] is False
    assert set(meta["test_metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

    assert (C.REPORTS_DIR / "model_comparison.csv").exists()
    for name in TINY_SPECS:
        assert (C.FIGURES_DIR / f"{name}_confusion_matrix.png").exists()


def test_quality_gate_blocks_saving(isolated):
    with pytest.raises(QualityGateError):
        run_training(data_path=isolated, min_test_auc=1.01, track=False, specs=TINY_SPECS)
    assert not C.MODEL_PATH.exists(), "a model that failed the gate must not be saved"
