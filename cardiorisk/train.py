"""Tune, compare, track (MLflow) and package heart-disease classifiers.

Run from the repository root:
    python -m cardiorisk.train
    python -m cardiorisk.train --min-test-auc 0.85 --no-register
    python -m cardiorisk.train --no-tracking     # Task 2 on its own, no MLflow

What happens:
  1. stratified 80/20 train/test split (test set is never used for tuning)
  2. for each model family: GridSearchCV with stratified 5-fold CV on the train set,
     scored on accuracy / precision / recall / F1 / ROC-AUC, refit on ROC-AUC
  3. every grid candidate is logged to MLflow as a nested run; the best per family
     gets test metrics, plots and the model artifact
  4. the winning family is chosen by mean CV ROC-AUC (not by test score)
  5. quality gate: the winner's test ROC-AUC must reach --min-test-auc, else exit 1
  6. the winner is registered in the MLflow Model Registry and saved to
     models/model.joblib + models/model_metadata.json for the API
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from cardiorisk import config as C  # noqa: E402
from cardiorisk.features import build_preprocessor, split_features_target  # noqa: E402

log = logging.getLogger(__name__)

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# Threshold metrics are computed from P(disease) >= DECISION_THRESHOLD - the same rule
# the API uses. sklearn's default scorers call predict() instead, which for an SVM with
# probability=True is a *different* decision rule (Platt scaling), so CV and test
# numbers would not be comparable.
def _labels(proba):
    return (proba >= C.DECISION_THRESHOLD).astype(int)


def _accuracy_at_t(y, proba):
    return accuracy_score(y, _labels(proba))


def _precision_at_t(y, proba):
    return precision_score(y, _labels(proba), zero_division=0)


def _recall_at_t(y, proba):
    return recall_score(y, _labels(proba), zero_division=0)


def _f1_at_t(y, proba):
    return f1_score(y, _labels(proba), zero_division=0)


SCORING = {
    "accuracy": make_scorer(_accuracy_at_t, response_method="predict_proba"),
    "precision": make_scorer(_precision_at_t, response_method="predict_proba"),
    "recall": make_scorer(_recall_at_t, response_method="predict_proba"),
    "f1": make_scorer(_f1_at_t, response_method="predict_proba"),
    "roc_auc": "roc_auc",
}


# ----------------------------------------------------------- model zoo
def model_specs(random_state: int = C.RANDOM_STATE) -> dict:
    """name -> (estimator, hyper-parameter grid). Grid keys use the 'clf__' prefix."""
    return {
        "logistic_regression": (
            LogisticRegression(solver="liblinear", max_iter=5000, random_state=random_state),
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l1", "l2"],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {
                "clf__n_estimators": [150, 300],
                "clf__max_depth": [None, 4, 8],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
        "svm_rbf": (
            SVC(kernel="rbf", probability=True, random_state=random_state),
            {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", 0.01, 0.1],
            },
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("clf", estimator)])


@dataclass
class TuningResult:
    name: str
    search: GridSearchCV

    @property
    def best_estimator(self) -> Pipeline:
        return self.search.best_estimator_

    @property
    def best_params(self) -> dict:
        return {k.replace("clf__", ""): v for k, v in self.search.best_params_.items()}

    def cv_metric(self, metric: str, stat: str = "mean", index: int | None = None) -> float:
        i = self.search.best_index_ if index is None else index
        return float(self.search.cv_results_[f"{stat}_test_{metric}"][i])

    @property
    def n_candidates(self) -> int:
        return len(self.search.cv_results_["params"])


def tune_model(name: str, estimator, grid: dict, X: pd.DataFrame, y: pd.Series,
               cv_folds: int = C.CV_FOLDS, random_state: int = C.RANDOM_STATE,
               n_jobs: int = -1) -> TuningResult:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(build_pipeline(estimator), grid, scoring=SCORING,
                          refit=C.PRIMARY_METRIC, cv=cv, n_jobs=n_jobs, error_score="raise")
    search.fit(X, y)
    return TuningResult(name=name, search=search)


def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    idx = list(model.classes_).index(1)
    return model.predict_proba(X)[:, idx]


def evaluate(model, X: pd.DataFrame, y: pd.Series,
             threshold: float = C.DECISION_THRESHOLD) -> dict:
    proba = positive_proba(model, X)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def select_best(results: dict[str, TuningResult], metric: str = C.PRIMARY_METRIC) -> str:
    """Highest mean CV score wins; ties go to the lower CV std (more stable model)."""
    return max(results, key=lambda n: (results[n].cv_metric(metric),
                                       -results[n].cv_metric(metric, "std")))


def comparison_table(results: dict[str, TuningResult], test_metrics: dict[str, dict]
                     ) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        row = {"model": name}
        for m in METRICS:
            row[f"cv_{m}_mean"] = round(res.cv_metric(m), 4)
            row[f"cv_{m}_std"] = round(res.cv_metric(m, "std"), 4)
        for m, v in test_metrics[name].items():
            row[f"test_{m}"] = round(v, 4)
        row["best_params"] = json.dumps(res.best_params, default=str)
        rows.append(row)
    return (pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------- plots
def plot_confusion(model, X, y, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    pred = (positive_proba(model, X) >= C.DECISION_THRESHOLD).astype(int)
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["No disease", "Disease"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc(models: dict, X, y, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, model in models.items():
        proba = positive_proba(model, X)
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_permutation_importance(model, X, y, title: str,
                                random_state: int = C.RANDOM_STATE) -> plt.Figure:
    # Model-agnostic: shuffle one raw feature at a time and measure the ROC-AUC drop.
    # Works identically for LR, RF and SVM, so the plots are comparable.
    imp = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=20,
                                 random_state=random_state, n_jobs=1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(X.columns)[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#5B7DB1")
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("Mean drop in test ROC-AUC when shuffled")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fold_scores(res: TuningResult, index: int) -> dict:
    """Per-fold CV scores of one grid candidate: {metric: [fold1, ..., foldK]}."""
    cvr = res.search.cv_results_
    return {m: [float(cvr[f"split{k}_test_{m}"][index]) for k in range(res.search.n_splits_)]
            for m in METRICS}


def plot_fold_scores(scores: dict, title: str) -> plt.Figure:
    """Spread of each metric across CV folds - shows how stable a candidate is."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for j, m in enumerate(METRICS):
        vals = scores[m]
        ax.scatter([j] * len(vals), vals, color="#5B7DB1", alpha=0.7, s=25)
        ax.hlines(np.mean(vals), j - 0.25, j + 0.25, color="#C8553D", lw=2)
    ax.set_xticks(range(len(METRICS)), METRICS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score per fold (red = mean)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_cv_comparison(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(table)
    x = np.arange(len(METRICS))
    for i, row in table.iterrows():
        means = [row[f"cv_{m}_mean"] for m in METRICS]
        stds = [row[f"cv_{m}_std"] for m in METRICS]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=row["model"])
    ax.set_xticks(x + width * (len(table) - 1) / 2, METRICS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean 5-fold CV score (error bar = std)")
    ax.set_title("Cross-validated comparison of the tuned models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- helpers
def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_fig(fig: plt.Figure, name: str, mlflow_module) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    mlflow_module.log_figure(fig, f"plots/{name}.png")
    plt.close(fig)


def _setup_mlflow(mlflow_module) -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(C.ROOT / 'mlflow.db').as_posix()}")
    mlflow_module.set_tracking_uri(uri)
    if mlflow_module.get_experiment_by_name(C.MLFLOW_EXPERIMENT) is None:
        mlflow_module.create_experiment(C.MLFLOW_EXPERIMENT,
                                        artifact_location=(C.ROOT / "mlruns").as_uri())
    mlflow_module.set_experiment(C.MLFLOW_EXPERIMENT)
    log.info("MLflow tracking URI: %s  experiment: %s", uri, C.MLFLOW_EXPERIMENT)


class MlflowTracker:
    """Thin wrapper so run_training talks to one object whether or not MLflow is used."""

    enabled = True

    def __init__(self):
        import mlflow  # imported lazily so unit tests of the pure functions stay light
        import mlflow.sklearn  # noqa: F401  (registers the sklearn flavour)
        from mlflow.models import infer_signature

        self._mlflow = mlflow
        self._infer_signature = infer_signature
        _setup_mlflow(mlflow)

    def __getattr__(self, name):
        return getattr(self._mlflow, name)

    def log_model(self, model, example: pd.DataFrame) -> None:
        signature = self._infer_signature(example, model.predict(example))
        self._mlflow.sklearn.log_model(model, "model", signature=signature, input_example=example)


class NullTracker:
    """Same interface, does nothing. Used by `--no-tracking` (Task 2 on its own) and tests."""

    enabled = False

    @contextmanager
    def start_run(self, run_name: str | None = None, nested: bool = False):
        yield SimpleNamespace(info=SimpleNamespace(run_id=f"untracked-{uuid.uuid4().hex}"))

    def log_model(self, model, example) -> None:
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class QualityGateError(RuntimeError):
    pass


# ----------------------------------------------------------- main entry
def run_training(data_path: Path = C.CLEAN_FILE, min_test_auc: float | None = None,
                 register: bool = True, track: bool = True, specs: dict | None = None,
                 n_jobs: int = 2) -> dict:
    """track=False runs pure model development (Task 2) with no MLflow at all.
    specs overrides model_specs() - used by the tests to keep the end-to-end run fast."""
    from cardiorisk.data import load_clean

    if min_test_auc is None:
        min_test_auc = float(os.getenv("MIN_TEST_ROC_AUC", C.DEFAULT_MIN_TEST_ROC_AUC))

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    log.info("Data: %d rows -> %d train / %d test (positive rate %.2f)",
             len(df), len(X_train), len(X_test), y.mean())

    mlflow = MlflowTracker() if track else NullTracker()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    results: dict[str, TuningResult] = {}
    test_metrics: dict[str, dict] = {}
    child_run_ids: dict[str, str] = {}

    with mlflow.start_run(run_name=f"training-session-{stamp}") as parent:
        mlflow.set_tags({"stage": "model-selection", "dataset": "UCI heart disease (Cleveland)"})
        mlflow.log_params({
            "test_size": C.TEST_SIZE, "random_state": C.RANDOM_STATE, "cv_folds": C.CV_FOLDS,
            "primary_metric": C.PRIMARY_METRIC, "n_rows": len(df), "n_train": len(X_train),
            "n_test": len(X_test), "n_features_raw": X.shape[1],
            "data_sha256": file_sha256(data_path), "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(), "min_test_roc_auc": min_test_auc,
        })
        mlflow.log_artifact(str(data_path), artifact_path="data")
        for eda_png in sorted(C.FIGURES_DIR.glob("eda_*.png")):  # produced by cardiorisk.eda
            mlflow.log_artifact(str(eda_png), artifact_path="eda")

        for name, (estimator, grid) in (specs or model_specs()).items():
            with mlflow.start_run(run_name=name, nested=True) as child:
                log.info("Tuning %s over %d candidates ...", name,
                         int(np.prod([len(v) for v in grid.values()])))
                res = tune_model(name, estimator, grid, X_train, y_train, n_jobs=n_jobs)
                results[name] = res
                child_run_ids[name] = child.info.run_id

                # every grid candidate becomes its own nested run
                cvr = res.search.cv_results_
                for i in range(res.n_candidates):
                    with mlflow.start_run(run_name=f"{name}-cand-{i:02d}", nested=True):
                        params = {k.replace("clf__", ""): str(v)
                                  for k, v in cvr["params"][i].items()}
                        mlflow.log_params(params)
                        mlflow.log_metrics(
                            {f"cv_{m}_mean": res.cv_metric(m, "mean", i) for m in METRICS}
                            | {f"cv_{m}_std": res.cv_metric(m, "std", i) for m in METRICS}
                            | {"cv_fit_time_s": float(cvr["mean_fit_time"][i])})
                        mlflow.set_tags({"model_family": name,
                                         "cv_rank": int(cvr[f"rank_test_{C.PRIMARY_METRIC}"][i])})
                        folds = fold_scores(res, i)
                        mlflow.log_dict(folds, "cv_fold_scores.json")
                        fig = plot_fold_scores(folds, f"{name} candidate {i:02d}: {params}")
                        mlflow.log_figure(fig, "plots/cv_fold_scores.png")
                        plt.close(fig)

                best = res.best_estimator
                tm = evaluate(best, X_test, y_test)
                test_metrics[name] = tm
                mlflow.set_tags({"model_family": name, "role": "best-of-family"})
                mlflow.log_params({f"best_{k}": str(v) for k, v in res.best_params.items()})
                mlflow.log_metrics({f"cv_{m}_mean": res.cv_metric(m) for m in METRICS}
                                   | {f"cv_{m}_std": res.cv_metric(m, "std") for m in METRICS}
                                   | {f"test_{m}": v for m, v in tm.items()})
                pd.DataFrame(cvr).drop(columns=["params"]).to_csv(
                    C.REPORTS_DIR / f"cv_results_{name}.csv", index=False)
                mlflow.log_artifact(str(C.REPORTS_DIR / f"cv_results_{name}.csv"), "cv_results")

                _save_fig(plot_confusion(best, X_test, y_test, f"{name} - test confusion matrix"),
                          f"{name}_confusion_matrix", mlflow)
                _save_fig(plot_roc({name: best}, X_test, y_test, f"{name} - test ROC"),
                          f"{name}_roc_curve", mlflow)
                _save_fig(plot_permutation_importance(best, X_test, y_test,
                                                      f"{name} - permutation importance"),
                          f"{name}_permutation_importance", mlflow)

                mlflow.log_model(best, X_train.dropna().head(3))
                log.info("  %-20s cv_auc=%.4f±%.4f  test_auc=%.4f  params=%s", name,
                         res.cv_metric("roc_auc"), res.cv_metric("roc_auc", "std"),
                         tm["roc_auc"], res.best_params)

        # ------------------------------------------------ compare + select
        table = comparison_table(results, test_metrics)
        table.to_csv(C.REPORTS_DIR / "model_comparison.csv", index=False)
        mlflow.log_artifact(str(C.REPORTS_DIR / "model_comparison.csv"))
        _save_fig(plot_cv_comparison(table), "model_cv_comparison", mlflow)
        _save_fig(plot_roc({n: r.best_estimator for n, r in results.items()}, X_test, y_test,
                           "All tuned models - test ROC"), "all_models_roc", mlflow)

        best_name = select_best(results)
        best_res = results[best_name]
        final_test = test_metrics[best_name]
        mlflow.set_tag("selected_model", best_name)
        mlflow.log_metrics({f"final_test_{m}": v for m, v in final_test.items()}
                           | {"final_cv_roc_auc_mean": best_res.cv_metric("roc_auc")})
        log.info("Selected %s (highest mean CV ROC-AUC = %.4f). Test ROC-AUC = %.4f",
                 best_name, best_res.cv_metric("roc_auc"), final_test["roc_auc"])

        # ----------------------------------------------------- quality gate
        passed = final_test["roc_auc"] >= min_test_auc
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")
        if not passed:
            raise QualityGateError(
                f"Quality gate failed: test ROC-AUC {final_test['roc_auc']:.4f} < "
                f"{min_test_auc:.4f}. Model NOT saved or registered.")

        if register and mlflow.enabled:
            try:
                mv = mlflow.register_model(f"runs:/{child_run_ids[best_name]}/model",
                                           C.REGISTERED_MODEL_NAME)
                log.info("Registered %s version %s", C.REGISTERED_MODEL_NAME, mv.version)
            except Exception as exc:  # registry is a nice-to-have, not a blocker
                log.warning("Model registry step skipped: %s", exc)

        # ------------------------------------------- package for serving
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_res.best_estimator, C.MODEL_PATH)
        metadata = {
            "model_name": best_name,
            "model_version": f"{C.MODEL_VERSION}+{child_run_ids[best_name][:8]}",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in best_res.best_params.items()},
            "cv_metrics_mean": {m: round(best_res.cv_metric(m), 4) for m in METRICS},
            "test_metrics": {m: round(v, 4) for m, v in final_test.items()},
            "features": C.FEATURES,
            "decision_threshold": C.DECISION_THRESHOLD,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "data_sha256": file_sha256(data_path),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "mlflow_parent_run_id": parent.info.run_id,
            "tracked_with_mlflow": mlflow.enabled,
            "mlflow_model_run_id": child_run_ids[best_name],
        }
        C.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(C.METADATA_PATH))
        log.info("Saved %s and %s", C.MODEL_PATH, C.METADATA_PATH)

    return {"selected_model": best_name, "comparison": table, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-test-auc",
        type=float,
        default=None,
        help=(
            f"quality gate (default env MIN_TEST_ROC_AUC or "
            f"{C.DEFAULT_MIN_TEST_ROC_AUC})"
        ),
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="skip the MLflow Model Registry step",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Task 2 only: train, compare, gate and save without MLflow",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="number of parallel jobs for cross-validation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        summary = run_training(
            min_test_auc=args.min_test_auc,
            register=not args.no_register,
            track=not args.no_tracking,
            n_jobs=args.n_jobs,
        )
    except QualityGateError as exc:
        log.error(str(exc))
        sys.exit(1)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n" + summary["comparison"].drop(columns=["best_params"]).to_string())
    print(f"\nSelected model: {summary['selected_model']}")


if __name__ == "__main__":
    main()
