"""Separate module so it cannot disturb the shared client in test_api.py."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.schemas import EXAMPLE_PATIENT  # noqa: E402


def test_health_reports_503_when_model_missing(tmp_path):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    from api.main import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=EXAMPLE_PATIENT).status_code == 503
    mp.undo()
