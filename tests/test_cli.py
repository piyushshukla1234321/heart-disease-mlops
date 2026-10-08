"""CLI regression tests for cardiorisk.train.main()."""

import sys

import pandas as pd

import cardiorisk.train as train


def test_main_forwards_n_jobs(monkeypatch, capsys):
    """--n-jobs must reach run_training instead of being silently ignored."""
    captured = {}

    def fake_run_training(**kwargs):
        captured.update(kwargs)
        return {
            "selected_model": "logistic_regression",
            "comparison": pd.DataFrame(
                [{"model": "logistic_regression", "best_params": {"C": 1.0}, "cv_roc_auc": 0.9}]
            ),
        }

    monkeypatch.setattr(train, "run_training", fake_run_training)
    monkeypatch.setattr(
        sys, "argv", ["cardiorisk.train", "--n-jobs", "3", "--no-tracking", "--no-register"]
    )

    train.main()

    assert captured["n_jobs"] == 3
    assert captured["track"] is False
    assert captured["register"] is False
    assert "Selected model: logistic_regression" in capsys.readouterr().out
