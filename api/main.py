"""FastAPI service for the heart-disease risk model.

Local run (repo root):  uvicorn api.main:app --reload --port 8000
Docs:                   http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import Match

from api.monitoring import (CONFIDENCE, DISEASE_PROBABILITY, IN_PROGRESS, MODEL_INFO,
                            MODEL_LOADED, PREDICTIONS, REQUEST_COUNT, REQUEST_LATENCY,
                            configure_logging)
from api.schemas import (BatchRequest, BatchResponse, PatientFeatures, PredictionResponse)
from cardiorisk import config as C

log = logging.getLogger("heart_api")
LABELS = {0: "no_disease", 1: "disease"}


def risk_band(p: float) -> str:
    # Illustrative bands for the demo - not clinically validated thresholds.
    if p < 0.35:
        return "low"
    if p < 0.65:
        return "moderate"
    return "high"


def load_model(app: FastAPI) -> None:
    model_path = Path(os.getenv("MODEL_PATH", str(C.MODEL_PATH)))
    meta_path = model_path.with_name("model_metadata.json")
    app.state.model, app.state.metadata, app.state.load_error = None, {}, None
    try:
        app.state.model = joblib.load(model_path)
        if meta_path.exists():
            app.state.metadata = json.loads(meta_path.read_text())
        trained_with = app.state.metadata.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            log.warning("sklearn version mismatch",
                        extra={"fields": {"trained": trained_with,
                                          "runtime": sklearn.__version__}})
        MODEL_LOADED.set(1)
        MODEL_INFO.labels(app.state.metadata.get("model_name", "unknown"),
                          app.state.metadata.get("model_version", "unknown")).set(1)
        log.info("model loaded", extra={"fields": {"path": str(model_path),
                                                   "model": app.state.metadata.get("model_name")}})
    except Exception as exc:  # keep the process up; /health reports 503
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        MODEL_LOADED.set(0)
        log.error("model failed to load", extra={"fields": {"path": str(model_path),
                                                            "error": app.state.load_error}})


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.getenv("LOG_LEVEL", "INFO"), os.getenv("LOG_FILE"))
    load_model(app)
    yield


app = FastAPI(title="Heart Disease Risk API", version=C.MODEL_VERSION, lifespan=lifespan,
              description="Predicts presence of heart disease from 13 clinical features "
                          "(UCI Cleveland). Educational project - not a medical device.")


def _route_template(request: Request) -> str:
    """'/predict' rather than the raw URL, so metric label cardinality stays bounded."""
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", "unknown")
    return "unmatched"


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    endpoint = _route_template(request)
    start = time.perf_counter()
    status = 500
    IN_PROGRESS.inc()
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed = time.perf_counter() - start
        IN_PROGRESS.dec()
        REQUEST_COUNT.labels(request.method, endpoint, str(status)).inc()
        REQUEST_LATENCY.labels(endpoint).observe(elapsed)
        if endpoint != "/metrics":  # Prometheus scrapes every 5 s; don't flood the log
            log.info("http_request", extra={"fields": {
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})
    response.headers["X-Request-ID"] = request_id
    return response


def _require_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503,
                            detail=f"Model not loaded: {request.app.state.load_error}")
    return model


def _score(request: Request, patients: list[PatientFeatures]) -> list[PredictionResponse]:
    model = _require_model(request)
    meta = request.app.state.metadata
    threshold = float(meta.get("decision_threshold", C.DECISION_THRESHOLD))
    version = meta.get("model_version", "unknown")
    rows = [p.model_dump() for p in patients]
    frame = pd.DataFrame(rows, columns=C.FEATURES).astype(float)
    proba = model.predict_proba(frame)[:, list(model.classes_).index(1)]

    out = []
    for p in proba:
        p = float(p)
        pred = int(p >= threshold)
        conf = p if pred == 1 else 1.0 - p
        PREDICTIONS.labels(LABELS[pred]).inc()
        DISEASE_PROBABILITY.observe(p)
        CONFIDENCE.observe(conf)
        out.append(PredictionResponse(
            prediction=pred, label=LABELS[pred], probability_disease=round(p, 4),
            confidence=round(conf, 4), risk_band=risk_band(p), model_version=version,
            request_id=request.state.request_id))
    log.info("prediction", extra={"fields": {
        "request_id": request.state.request_id, "n": len(out), "inputs": rows,
        "outputs": [{"prediction": o.prediction, "p": o.probability_disease} for o in out]}})
    return out


# ------------------------------------------------------------ endpoints
@app.get("/", tags=["meta"])
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "predict": "/predict"}


@app.get("/live", tags=["meta"])
def live():
    """Liveness: the process is up (used by the Kubernetes livenessProbe)."""
    return {"status": "alive"}


@app.get("/health", tags=["meta"])
def health(request: Request):
    """Readiness: a model is loaded and can serve (Kubernetes readinessProbe)."""
    if getattr(request.app.state, "model", None) is None:
        return JSONResponse(status_code=503, content={
            "status": "unavailable", "model_loaded": False,
            "error": request.app.state.load_error})
    return {"status": "ok", "model_loaded": True,
            "model_version": request.app.state.metadata.get("model_version", "unknown")}


@app.get("/model-info", tags=["meta"])
def model_info(request: Request):
    _require_model(request)
    return request.app.state.metadata


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(patient: PatientFeatures, request: Request) -> PredictionResponse:
    return _score(request, [patient])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    return BatchResponse(predictions=_score(request, batch.instances))


@app.get("/metrics", tags=["meta"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
