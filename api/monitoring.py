"""Prometheus metrics and structured (JSON-lines) logging for the API."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram

# ------------------------------------------------------------- metrics
REQUEST_COUNT = Counter("api_requests_total", "HTTP requests",
                        ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "HTTP request latency",
                            ["endpoint"],
                            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5))
IN_PROGRESS = Gauge("api_requests_in_progress", "Requests currently being handled")
PREDICTIONS = Counter("model_predictions_total", "Predictions served, by outcome", ["label"])
DISEASE_PROBABILITY = Histogram("model_disease_probability", "Predicted P(disease)",
                                buckets=tuple(round(0.1 * i, 1) for i in range(1, 11)))
CONFIDENCE = Histogram("model_prediction_confidence", "Probability of the predicted class",
                       buckets=(0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
MODEL_LOADED = Gauge("model_loaded", "1 if a model is loaded and serving")
MODEL_INFO = Gauge("model_info", "Currently loaded model (value is always 1)",
                   ["model_name", "model_version"])


# ------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    """One JSON object per line - easy to grep, and what log stacks (Loki, ELK) expect."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_HANDLER_TAG = "_cardiorisk_handler"


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    root = logging.getLogger()
    for h in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(h)  # idempotent: safe to call on every startup
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        setattr(h, _HANDLER_TAG, True)
        root.addHandler(h)
    root.setLevel(level.upper())
