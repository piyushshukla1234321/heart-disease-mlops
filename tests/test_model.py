import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cardiorisk.features import split_features_target
from cardiorisk.train import (build_pipeline, comparison_table, evaluate, model_specs,
                              positive_proba, select_best, tune_model)


@pytest.fixture
def split(clean_df):
    X, y = split_features_target(clean_df)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


def test_pipeline_learns_signal(split):
    X_tr, X_te, y_tr, y_te = split
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    metrics = evaluate(model, X_te, y_te)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["roc_auc"] > 0.7  # synthetic data has a real signal


def test_probabilities_are_valid(split):
    X_tr, X_te, y_tr, _ = split
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    p = positive_proba(model, X_te)
    assert p.shape == (len(X_te),)
    assert np.all((p >= 0) & (p <= 1))


def test_training_is_deterministic(split):
    X_tr, X_te, y_tr, _ = split
    est, _ = model_specs()["random_forest"]
    p1 = positive_proba(build_pipeline(est).fit(X_tr, y_tr), X_te)
    est2, _ = model_specs()["random_forest"]
    p2 = positive_proba(build_pipeline(est2).fit(X_tr, y_tr), X_te)
    np.testing.assert_allclose(p1, p2)


def test_pipeline_handles_missing_values_at_predict_time(split):
    X_tr, X_te, y_tr, _ = split
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    X_bad = X_te.copy()
    X_bad.iloc[0, :] = np.nan
    assert np.isfinite(positive_proba(model, X_bad)).all()


def test_tune_and_select_best_uses_cv_score(split):
    X_tr, _, y_tr, _ = split
    results = {
        "weak_lr": tune_model("weak_lr", LogisticRegression(solver="liblinear"),
                              {"clf__C": [1e-4], "clf__penalty": ["l1"]}, X_tr, y_tr,
                              cv_folds=3, n_jobs=1),
        "good_lr": tune_model("good_lr", LogisticRegression(solver="liblinear"),
                              {"clf__C": [1.0]}, X_tr, y_tr, cv_folds=3, n_jobs=1),
    }
    # an L1 penalty this strong zeroes every coefficient -> AUC 0.5
    assert results["weak_lr"].cv_metric("roc_auc") < results["good_lr"].cv_metric("roc_auc")
    assert select_best(results) == "good_lr"


def test_comparison_table_sorted_by_cv_auc(split):
    X_tr, X_te, y_tr, y_te = split
    results = {
        name: tune_model(name, LogisticRegression(solver="liblinear"), {"clf__C": [c]},
                         X_tr, y_tr, cv_folds=3, n_jobs=1)
        for name, c in [("a", 1e-4), ("b", 1.0)]
    }
    tests = {n: evaluate(r.best_estimator, X_te, y_te) for n, r in results.items()}
    table = comparison_table(results, tests)
    assert table["model"].iloc[0] == select_best(results)
    assert table["cv_roc_auc_mean"].is_monotonic_decreasing


def test_model_specs_grids_target_the_classifier_step():
    for name, (_, grid) in model_specs().items():
        assert grid, name
        assert all(k.startswith("clf__") for k in grid), name


def test_cv_threshold_metrics_match_probability_rule(split):
    """CV accuracy must be computed from P>=0.5 (the API's rule), not predict().
    Recomputing fold-by-fold by hand must reproduce GridSearchCV's number."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC

    from cardiorisk.train import _accuracy_at_t
    X_tr, _, y_tr, _ = split
    est = SVC(probability=True, random_state=0, C=0.1, gamma=0.01)
    res = tune_model("svm", est, {"clf__C": [0.1]}, X_tr, y_tr, cv_folds=3,
                     random_state=0, n_jobs=1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    manual = []
    for tr, te in cv.split(X_tr, y_tr):
        m = build_pipeline(SVC(probability=True, random_state=0, C=0.1, gamma=0.01))
        m.fit(X_tr.iloc[tr], y_tr.iloc[tr])
        manual.append(_accuracy_at_t(y_tr.iloc[te], positive_proba(m, X_tr.iloc[te])))
    assert res.cv_metric("accuracy") == pytest.approx(float(np.mean(manual)), abs=1e-9)


def test_fold_scores_match_cv_means(split):
    from cardiorisk.train import fold_scores, plot_fold_scores
    X_tr, _, y_tr, _ = split
    res = tune_model("lr", LogisticRegression(solver="liblinear"), {"clf__C": [0.1, 1.0]},
                     X_tr, y_tr, cv_folds=3, n_jobs=1)
    for i in range(res.n_candidates):
        folds = fold_scores(res, i)
        assert all(len(v) == 3 for v in folds.values())
        assert np.mean(folds["roc_auc"]) == pytest.approx(res.cv_metric("roc_auc", index=i))
    plot_fold_scores(fold_scores(res, 0), "t").clf()
