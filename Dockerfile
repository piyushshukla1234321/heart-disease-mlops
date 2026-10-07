# Serving image for the heart-disease risk API.
# Build (repo root, after training so models/ exists):
#   docker build -t heart-risk-api:1.0.0 .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

LABEL org.opencontainers.image.title="heart-risk-api" \
      org.opencontainers.image.description="FastAPI service for UCI heart-disease risk"

WORKDIR /app

# Dependencies first: this layer is cached until requirements-api.txt changes.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser

COPY cardiorisk/ ./cardiorisk/
COPY api/ ./api/
# The build fails loudly here if training has not produced the model yet.
COPY models/model.joblib models/model_metadata.json ./models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# --no-access-log: the app writes its own structured JSON request log.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
