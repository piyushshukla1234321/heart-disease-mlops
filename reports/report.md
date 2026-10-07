# Heart Disease Risk — MLOps Assignment 1 Report

> **Student:** Piyush Shukla
> **Course:** AIMLCZG523 Machine Learning Operations
> **Assignment:** Assignment 1 — End-to-End MLOps Pipeline
> **Dataset:** UCI Heart Disease — Cleveland
> **Runtime:** Python 3.13
> **Repository:** `heart-disease-mlops`

## 1. Executive summary

This project implements an end-to-end MLOps pipeline for binary heart-disease risk classification using the UCI Cleveland heart-disease dataset. The pipeline covers data acquisition and cleaning, exploratory data analysis, preprocessing, model comparison and hyperparameter tuning, MLflow experiment tracking, reproducible packaging, automated testing and CI/CD, Docker containerization, Kubernetes deployment, and production-style monitoring with Prometheus and Grafana.

The final model selected by the training pipeline is **Logistic Regression**. Model selection is based strictly on the mean 5-fold cross-validation ROC-AUC, not on the held-out test set. Logistic Regression achieved a mean CV ROC-AUC of **0.9071** and a held-out test ROC-AUC of **0.9589**. The test-set accuracy was **0.8852**, precision **0.8182**, recall **0.9643**, and F1 **0.8852**. SVM-RBF and Random Forest were also evaluated. SVM-RBF had a slightly higher test ROC-AUC of 0.9600, but Logistic Regression remained the selected model because its CV ROC-AUC was higher (0.9071 versus 0.9044).

The selected preprocessing pipeline and classifier are stored together in `models/model.joblib`, with reproducibility metadata in `models/model_metadata.json`. The model is served through FastAPI, packaged into a non-root Docker image, deployed as two Kubernetes replicas behind a LoadBalancer service, and monitored through Prometheus and Grafana. During validation, 200 prediction requests were generated against the Docker Compose API, producing 186 successful responses and 14 validation-error responses; Prometheus successfully collected these metrics and Grafana visualized request rate, latency, errors, predictions, confidence, and model-health indicators.

## 2. Architecture and repository

The implemented architecture is:

```text
UCI Heart Disease (Cleveland)
          |
          v
  cardiorisk.data.py
   download + clean
          |
          +--> data/raw/processed.cleveland.data
          |
          v
  data/processed/heart_clean.csv
          |
          +--> cardiorisk.eda.py --> reports/figures/*.png
          |
          v
  cardiorisk.train.py
      |       |       |
      |       |       +--> SVM (RBF)
      |       +----------> Random Forest
      +------------------> Logistic Regression
          |
          v
     5-fold CV + tuning
          |
          v
        MLflow
   params / metrics /
   plots / models
          |
          v
    selected pipeline
          |
      +---+------------------+
      |                      |
      v                      v
 model.joblib        model_metadata.json
      |                      |
      +----------+-----------+
                 |
                 v
          FastAPI /predict
                 |
                 v
              Docker
                 |
                 v
       Kubernetes (2 pods)
                 |
          LoadBalancer :8080
                 |
       +---------+---------+
       |                   |
       v                   v
   Prometheus           JSON logs
       |                   |
       v                   v
    Grafana          Docker/Kubernetes logs

GitHub Actions: lint -> unit tests -> data/EDA/training -> Docker smoke test
```

The repository is organized around reusable Python modules in `cardiorisk/`, the serving layer in `api/`, tests in `tests/`, deployment manifests in `k8s/`, monitoring configuration in `monitoring/`, and evidence/report material in `reports/` and `screenshots/`. The notebooks are thin wrappers over the reusable package.

## 3. Task 1 — Data acquisition and EDA

### 3.1 Dataset acquisition and cleaning

The project uses the Cleveland subset of the UCI Heart Disease dataset. The acquisition/cleaning implementation is in `cardiorisk/data.py`. The raw source is stored under `data/raw/processed.cleveland.data`, while the cleaned dataset is generated at `data/processed/heart_clean.csv`.

The final cleaned dataset contains **303 patient records**, with **13 predictor features** and a binary target. No rows were removed during cleaning. The raw dataset contained missing values in `ca` and `thal`: four missing `ca` values and two missing `thal` values. Therefore, six records contain at least one missing feature cell.

The cleaning step deliberately does not globally fill these missing cells. Missing-value handling is learned inside the scikit-learn preprocessing pipeline during model training. This keeps preprocessing inside the fitted pipeline and prevents information from the held-out test set from leaking into the training process.

### 3.2 Class balance

| Class | Count | Proportion |
|---|---:|---:|
| 0 — no disease | 164 | 54.1% |
| 1 — disease | 139 | 45.9% |
| **Total** | **303** | **100%** |

The classes are reasonably balanced but not identical, so the training pipeline uses stratification for the train/test split and 5-fold cross-validation.

**Evidence figure:** `reports/figures/eda_class_balance.png`

### 3.3 Exploratory findings

The EDA module generates distributions, missing-value summaries, disease rates by categorical feature, correlation heatmaps, and outlier boxplots. The strongest absolute Spearman relationships with the target were approximately:

- `ca`: 0.489
- `exang`: 0.432
- `thalach`: -0.423
- `oldpeak`: 0.413
- `sex`: 0.277

These correlations are used for exploration and interpretation only; feature selection is not based on target correlations.

**Evidence figures:**

- `reports/figures/eda_numeric_distributions.png`
- `reports/figures/eda_missing_values.png`
- `reports/figures/eda_disease_rate_by_category.png`
- `reports/figures/eda_correlation_heatmap.png`
- `reports/figures/eda_outlier_boxplots.png`

## 4. Task 2 — Feature engineering and model development

### 4.1 Preprocessing pipeline

Feature groups are:

- **Numeric:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`
- **Binary:** `sex`, `fbs`, `exang`
- **Categorical:** `cp`, `restecg`, `slope`, `thal`

The fitted scikit-learn `ColumnTransformer` performs median imputation followed by `StandardScaler` for numeric features, most-frequent imputation for binary features, and most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categorical features.

The preprocessing and classifier are stored in the same fitted pipeline, so inference uses exactly the preprocessing learned during training.

### 4.2 Model development and selection

The dataset is split into **242 training rows and 61 test rows** using a stratified 80/20 split. Model selection uses **5-fold stratified cross-validation** and mean CV ROC-AUC as the primary criterion.

Three model families were evaluated:

1. Logistic Regression
2. Random Forest
3. SVM with RBF kernel

The hyperparameter search contains **43 candidates** in total: 16 Logistic Regression, 18 Random Forest, and 9 SVM-RBF. With five folds per candidate, this represents **215 cross-validation fits**.

### 4.3 Model comparison

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | **CV ROC-AUC** | Test Accuracy | Test Precision | Test Recall | Test F1 | **Test ROC-AUC** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.8552** | 0.8608 | **0.8281** | **0.8404** | **0.9071** | 0.8852 | 0.8182 | **0.9643** | **0.8852** | 0.9589 |
| SVM-RBF | 0.8347 | **0.8613** | 0.7739 | 0.8107 | 0.9044 | 0.8852 | **0.8387** | 0.9286 | 0.8814 | **0.9600** |
| Random Forest | 0.8305 | 0.8391 | 0.7826 | 0.8063 | 0.9003 | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9589 |

The selected model is **Logistic Regression**, because it has the highest mean cross-validation ROC-AUC (**0.9071**). The slightly higher SVM test ROC-AUC (**0.9600**) was not used for model selection, avoiding selection on the held-out test set.

Selected Logistic Regression parameters:

```text
C = 1.0
class_weight = balanced
penalty = l2
solver = liblinear
max_iter = 5000
random_state = 42
```

### 4.4 Model interpretation and evidence

The project generates ROC curves, confusion matrices, permutation-importance plots, and a cross-model ROC comparison. These artifacts are stored under `reports/figures/`.

Key evidence figures:

- `all_models_roc.png`
- `model_cv_comparison.png`
- `logistic_regression_roc_curve.png`
- `logistic_regression_confusion_matrix.png`
- `logistic_regression_permutation_importance.png`
- equivalent ROC/confusion/permutation plots for Random Forest and SVM-RBF.

## 5. Task 3 — Experiment tracking with MLflow

The project uses MLflow with a local SQLite backend and the experiment name `heart-disease-risk`.

The final experiment contains **47 runs**:

- 1 parent training-session run;
- 3 model-family runs;
- 43 nested hyperparameter-candidate runs.

Each candidate records hyperparameters and cross-validation metrics. Candidate artifacts include fold-level results and generated plots. The selected model is registered in the MLflow Model Registry under:

```text
heart-disease-risk-classifier
```

The selected model was registered as version 1 during the verified training run. Saved metadata records the MLflow parent and model run identifiers.

The experiment therefore provides traceability from candidate configuration through model-family comparison to the selected packaged model.

**Evidence to include:** MLflow experiment overview, candidate run hierarchy, model metrics/parameters, artifacts, and registered model/version.

## 6. Task 4 — Packaging and reproducibility

The final artifact is:

```text
models/model.joblib
```

It contains both preprocessing and the Logistic Regression classifier, avoiding training/serving preprocessing skew.

The companion `models/model_metadata.json` records model name/version, parameters, CV/test metrics, feature list, decision threshold, scikit-learn/Python versions, dataset SHA-256, train/test sizes, and MLflow identifiers.

Verified environment:

```text
Python:       3.13.16
scikit-learn: 1.6.1
MLflow:       2.20.3
PyArrow:      19.0.1
```

Dataset SHA-256:

```text
72eeb46492503e72d315a408b8a9fcc9afa0d09a1fabe47c9cc79c2ea1c82fd1
```

The repository includes the current trained artifact for immediate deployment, while the complete data and training pipeline can regenerate the model from scratch.

## 7. Task 5 — CI/CD and testing

The GitHub Actions workflow in `.github/workflows/ci.yml` is structured as:

```text
lint → test → train → docker
```

The lint stage runs Flake8. The test stage runs pytest with coverage and produces test artifacts. The training stage acquires/cleans data, generates EDA outputs, trains the candidate models, applies the ROC-AUC quality gate, and uploads model/training outputs. The Docker stage builds the image, starts the container, runs the API smoke test, captures container logs, and publishes evidence artifacts.

The local test suite has been verified with:

```text
49 passed
```

The local lint check completed successfully with no Flake8 findings for `cardiorisk`, `api`, `tests`, and `scripts`.

The training quality gate is **test ROC-AUC >= 0.80**. If the selected model falls below this threshold, training fails rather than producing a deployment artifact.

**Evidence still to attach:** successful GitHub Actions run and artifacts, plus an intentionally failed validation run if one is captured. This report does not claim those screenshots until they are actually collected.

## 8. Task 6 — Containerization and API serving

The FastAPI service provides:

```text
GET  /health
GET  /live
GET  /model-info
GET  /metrics
POST /predict
POST /predict/batch
GET  /docs
```

The smoke test verified:

```text
/health                         → ready
POST /predict                   → 200
POST /predict with invalid age → 422
/metrics                       → Prometheus counters exposed
ALL SMOKE TESTS PASSED
```

For the verified sample patient, the API returned:

```json
{
  "prediction": 0,
  "label": "no_disease",
  "probability_disease": 0.0384,
  "confidence": 0.9616,
  "risk_band": "low",
  "model_version": "1.0.0+95a5e322"
}
```

The Docker image is based on `python:3.13-slim`, runs as a non-root user, includes a healthcheck, and installs the API-serving dependency set from `requirements-api.txt`.

Verified image:

```text
heart-risk-api:1.0.0
```

## 9. Task 7 — Kubernetes deployment

The application was deployed to the Docker Desktop Kubernetes cluster in the `heart-risk` namespace.

Two replicas were running:

```text
heart-risk-api-5b5df44885-8vb7z   1/1 Running
heart-risk-api-5b5df44885-wlqnk    1/1 Running
```

Service:

```text
TYPE:        LoadBalancer
CLUSTER-IP:  10.96.17.145
EXTERNAL-IP: 172.18.0.5
PORT:        8080:31489/TCP
```

The service had two endpoints corresponding to the two pods. Readiness and liveness probes are configured in the deployment manifest.

A temporary curl pod successfully verified:

```text
GET /health → {"status":"ok","model_loaded":true,"model_version":"1.0.0+95a5e322"}
```

The same sample prediction was successfully returned through the Kubernetes service.

The Docker Desktop cluster uses a kind-based control-plane node. The locally built image was imported into the Kubernetes containerd `k8s.io` image store before deployment, allowing the cluster to use `heart-risk-api:1.0.0` without pulling it from a public registry.

**Evidence still to attach:** Kubernetes deployment/pods/service output, successful endpoint responses, and a pod-recovery/self-healing demonstration if captured.

## 10. Task 8 — Monitoring and logging

The monitoring stack consists of FastAPI metrics, Prometheus, and Grafana. Prometheus uses a 5-second scrape interval and successfully scrapes both the Docker Compose API and Kubernetes LoadBalancer endpoint.

Verified targets:

```text
heart-risk-api       → health: up
heart-risk-api-k8s   → health: up
```

The API exposes:

```text
api_requests_total
api_request_latency_seconds
api_requests_in_progress
```

### Traffic validation

The traffic generator sent **200** requests to the Docker Compose API:

| Status | Count |
|---|---:|
| HTTP 200 | 186 |
| HTTP 422 | 14 |
| **Total** | **200** |

Prometheus returned the same Docker Compose counters:

```text
/predict status=200 → 186
/predict status=422 → 14
```

The Kubernetes target also reported a `/predict` status-200 counter, confirming independent Kubernetes metric collection.

The `/predict` latency histogram contained 200 observations. The observed latency sum was approximately 4.614 seconds, giving an average of approximately **23.1 ms** across those requests.

### Grafana dashboard

The provisioned dashboard displays:

- model-loaded state;
- total predictions;
- share predicted as disease;
- recent 4xx/5xx error rate;
- request rate by endpoint and status;
- `/predict` p50/p95 latency;
- predictions per minute by outcome;
- mean predicted probability and confidence.

The verified dashboard displayed **187 total predictions**, reflecting Docker Compose successful predictions plus the Kubernetes prediction observed through the shared Prometheus view.

### Structured logs

The API emits structured JSON logs. Verified events include `http_request` and `prediction`, with request ID, timestamp, method/path, status, latency, prediction output, and serving information.

A request event includes fields such as:

```text
request_id
method
path
status
latency_ms
client
```

A production healthcare deployment should review raw feature logging against data-minimization and privacy requirements before handling protected health information. The assignment implementation demonstrates the logging mechanism.

## 11. Conclusion

This project implements the complete MLOps lifecycle required for the heart-disease classification service: reproducible data acquisition and EDA, preprocessing, multi-model comparison and cross-validation, MLflow tracking, packaged model artifacts, automated tests and CI/CD workflow, Docker serving, Kubernetes deployment, and operational monitoring.

Logistic Regression was selected using mean 5-fold CV ROC-AUC and achieved **0.9071 CV ROC-AUC** and **0.9589 test ROC-AUC**. The model is packaged together with its preprocessing pipeline and metadata, reducing the risk of training/serving skew.

The FastAPI service was validated locally, in Docker, and through the Kubernetes service. Prometheus successfully scraped both Docker Compose and Kubernetes endpoints, while Grafana provided operational dashboards for request rate, errors, latency, predictions, confidence, and model health. The implementation demonstrates not only model development but also reproducibility, deployment, observability, and quality controls expected in an end-to-end MLOps workflow.

## Appendix A — Reproduction commands

Run from the repository root.

### Environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Task 1

```bash
python -m cardiorisk.data
python -m cardiorisk.eda
```

### Tasks 2–4

```bash
python -m cardiorisk.train
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Testing

```bash
flake8 cardiorisk api tests scripts
pytest -q
```

### Docker/API

```bash
docker build -t heart-risk-api:1.0.0 .
docker run -d --name heart-api -p 8000:8000 heart-risk-api:1.0.0
python scripts/smoke_test.py
```

### Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
docker save heart-risk-api:1.0.0 | docker exec -i desktop-control-plane ctr -n k8s.io images import -
kubectl apply -f k8s/deployment.yaml -n heart-risk
kubectl apply -f k8s/service.yaml -n heart-risk
kubectl get deployment,pods,service -n heart-risk
```

### Monitoring

```bash
docker compose -f monitoring/docker-compose.yml up -d
python scripts/send_traffic.py
```

Grafana: `http://localhost:3000`
Prometheus: `http://localhost:9090`

## Appendix B — Evidence checklist

1. EDA/class balance and representative plots.
2. MLflow experiment hierarchy and registered model.
3. Successful local lint and pytest result.
4. GitHub Actions successful workflow and artifacts.
5. Docker image/container and API smoke-test response.
6. Kubernetes deployment, two ready pods and LoadBalancer service.
7. Kubernetes `/health` and `/predict` responses.
8. Prometheus targets showing both targets `up`.
9. Prometheus query showing `/predict` status 200/422 counters.
10. Grafana dashboard with generated traffic.
11. Structured JSON API logs.

Where an evidence item has not yet been captured, it is intentionally identified as an evidence requirement rather than claimed as a completed screenshot.
