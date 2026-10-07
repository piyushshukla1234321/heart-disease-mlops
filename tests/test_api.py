import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.schemas import EXAMPLE_PATIENT  # noqa: E402


@pytest.fixture(scope="module")
def client(trained_model_path):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(trained_model_path))
    from api.main import app
    with TestClient(app) as c:  # 'with' runs the lifespan -> model is loaded
        yield c
    mp.undo()


def test_live_and_health(client):
    assert client.get("/live").status_code == 200
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_valid_payload(client):
    r = client.post("/predict", json=EXAMPLE_PATIENT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "high"}
    expected_conf = (body["probability_disease"] if body["prediction"] == 1
                     else 1 - body["probability_disease"])
    assert body["confidence"] == pytest.approx(expected_conf, abs=1e-3)
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_request_id_is_propagated(client):
    r = client.post("/predict", json=EXAMPLE_PATIENT, headers={"X-Request-ID": "abc123"})
    assert r.json()["request_id"] == "abc123"


@pytest.mark.parametrize("field,value", [("cp", 1), ("cp", 4), ("slope", 1), ("slope", 3),
                                         ("thal", 3), ("thal", 6), ("restecg", 0), ("ca", 3)])
def test_predict_accepts_every_valid_uci_code(client, field, value):
    """Guards against using the Kaggle coding (cp 0-3, slope 0-2, thal 0-3) by mistake."""
    assert client.post("/predict", json={**EXAMPLE_PATIENT, field: value}).status_code == 200


@pytest.mark.parametrize("field,value", [("age", -5), ("cp", 9), ("chol", 5000),
                                         ("thal", 4), ("sex", "male"),
                                         ("cp", 0), ("slope", 0), ("thal", 1)])
def test_predict_rejects_invalid_values(client, field, value):
    r = client.post("/predict", json={**EXAMPLE_PATIENT, field: value})
    assert r.status_code == 422


def test_predict_rejects_missing_and_extra_fields(client):
    missing = {k: v for k, v in EXAMPLE_PATIENT.items() if k != "chol"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**EXAMPLE_PATIENT, "bmi": 22}).status_code == 422


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"instances": [EXAMPLE_PATIENT] * 3})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=EXAMPLE_PATIENT)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
    assert 'endpoint="/predict"' in text


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200 and r.json()["model_name"] == "test_lr"
