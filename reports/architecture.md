# MLOps Architecture

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
  plots / models /
          |
          v
 selected pipeline
          |
     +----+------------------+
     |                       |
     v                       v
model.joblib          model_metadata.json
     |                       |
     +-----------+-----------+
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
    Grafana          kubectl / Docker logs

GitHub Actions: lint -> unit tests -> data/EDA/training -> Docker smoke test
```

The architecture is intentionally local and reproducible: the same trained pipeline is packaged into the Docker image and then deployed to Docker Desktop Kubernetes. Prometheus scrapes `/metrics`; Grafana visualizes request and model-serving metrics.
