import dataclasses

import numpy as np
import pandas as pd

from swdss.models.registry import DATASETS
from swdss.models.train import CANDIDATE_MODELS, _fit_best, _regime_labels_for_index


def test_fit_best_returns_cv_metrics_alongside_the_original_holdout_metrics():
    """_fit_best must keep returning the same holdout r2/mae/rmse shape
    every existing caller (train_dataset, train_kp_interval_model) already
    unpacks, while adding walk-forward CV as a genuinely new, additional
    output — not a replacement.
    """
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"x": np.arange(n, dtype=float)})
    y = pd.Series(2.0 * X["x"] + 1.0 + rng.normal(scale=0.5, size=n))

    best_name, best_metrics, best_cv, final_model, all_candidates = _fit_best(X, y)

    regime_keys = {"quiet_mae", "quiet_n", "active_mae", "active_n", "storm_mae", "storm_n"}
    assert best_name in CANDIDATE_MODELS
    assert {"r2", "mae", "rmse"} <= set(best_metrics)
    assert set(best_metrics) - {"r2", "mae", "rmse"} <= regime_keys
    assert set(best_cv) == {"n_folds", "folds", "r2_mean", "r2_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std"}
    # all_candidates always includes every base CANDIDATE_MODELS entry, plus
    # an "Ensemble" blend when 2+ candidates had positive walk-forward weight.
    assert set(CANDIDATE_MODELS) <= set(all_candidates)
    assert set(all_candidates) - set(CANDIDATE_MODELS) <= {"Ensemble"}
    for candidate_result in all_candidates.values():
        assert {"holdout", "cv"} <= set(candidate_result)
        assert set(candidate_result) - {"holdout", "cv"} <= {"weights"}

    # The refit-on-full-data contract is unchanged: the returned model must
    # already be fit and able to predict on the full X immediately.
    preds = final_model.predict(X)
    assert len(preds) == n


def test_fit_best_selects_by_cv_mean_r2_not_holdout_r2():
    """A model that wins the single holdout split by chance but is
    unstable across walk-forward folds should NOT necessarily be selected
    — selection must key off cv.r2_mean, matching _fit_best's own
    docstring contract.
    """
    rng = np.random.default_rng(3)
    n = 600
    X = pd.DataFrame({"x": np.arange(n, dtype=float)})
    y = pd.Series(2.0 * X["x"] + rng.normal(scale=0.5, size=n))

    _, _, best_cv, _, all_candidates = _fit_best(X, y)
    best_cv_r2 = max(c["cv"]["r2_mean"] for c in all_candidates.values())
    assert best_cv["r2_mean"] == best_cv_r2


def test_regime_labels_fall_back_to_quiet_when_analytics_features_csv_is_missing(monkeypatch):
    """analytics_features.csv is a generated, gitignored training artifact
    — it won't exist in a fresh checkout or CI. Regime tagging must degrade
    to "Quiet" for every row (the same fallback already applied to per-row
    missing kp/dst/ae), not raise FileNotFoundError and take every caller
    of _fit_best down with it, which is what actually broke CI here.
    """
    missing_path_config = dataclasses.replace(
        DATASETS["analytics"], training_csv="/nonexistent/analytics_features.csv"
    )
    monkeypatch.setitem(DATASETS, "analytics", missing_path_config)

    index = pd.date_range("2026-01-01", periods=5, freq="h")
    labels = _regime_labels_for_index(index)

    assert list(labels) == ["Quiet"] * 5
