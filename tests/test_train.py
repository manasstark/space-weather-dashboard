import numpy as np
import pandas as pd

from swdss.models.train import CANDIDATE_MODELS, _fit_best


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

    assert best_name in CANDIDATE_MODELS
    assert set(best_metrics) == {"r2", "mae", "rmse"}
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
