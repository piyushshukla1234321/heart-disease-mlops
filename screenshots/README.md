# Screenshot checklist

Save each screenshot here with the file name shown. These are reused in the report.

| # | File name | What to capture | Assignment task |
|---|-----------|-----------------|-----------------|
| 1 | 01_eda_figures.png | EDA notebook with class-balance, histograms and heatmap visible | 1 |
| 2 | 02_training_console.png | Terminal output of `python -m cardiorisk.train` (comparison table + selected model) | 2 |
| 3 | 03_mlflow_runs.png | MLflow UI: experiment `heart-disease-risk`, runs table with nested runs expanded | 3 |
| 4 | 04_mlflow_compare.png | MLflow UI: "Compare" view of the 3 best-of-family runs (metrics chart) | 3 |
| 5 | 05_mlflow_artifacts.png | MLflow UI: one run's Artifacts tab (plots + model folder) | 3, 4 |
| 6 | 06_mlflow_registry.png | MLflow UI: Models tab showing `heart-disease-risk-classifier` version | 4 |
| 7 | 07_pytest_local.png | Terminal: `pytest -v` all green | 5 |
| 8 | 08_actions_success.png | GitHub Actions: workflow run with all 4 jobs green | 5 |
| 9 | 09_actions_artifacts.png | GitHub Actions: Artifacts section of that run | 5 |
| 10 | 10_actions_failure.png | GitHub Actions: a deliberately broken commit failing at lint or test (proves fail-fast) | 5 |
| 11 | 11_docker_build_run.png | Terminal: `docker build` + `docker run` + `docker ps` | 6 |
| 12 | 12_docker_predict.png | Swagger UI /docs or curl showing /predict response from the container | 6 |
| 13 | 13_k8s_get_all.png | Terminal: `kubectl get all -n heart-risk` (2 pods Running, service with EXTERNAL-IP) | 7 |
| 14 | 14_k8s_predict.png | /predict response via http://localhost:8080 (the K8s LoadBalancer) | 7 |
| 15 | 15_k8s_logs.png | `kubectl logs` showing JSON request logs | 8 |
| 16 | 16_prometheus_targets.png | Prometheus Status -> Targets page with heart-risk-api UP | 8 |
| 17 | 17_grafana_dashboard.png | Grafana dashboard after running send_traffic.py | 8 |
