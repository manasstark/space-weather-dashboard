import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from swdss.models.validation import evaluate_walk_forward, walk_forward_fold_bounds


def test_fold_bounds_cover_the_tail_without_overlap():
    bounds = walk_forward_fold_bounds(1000, n_folds=5, min_train_fraction=0.5)
    assert len(bounds) == 5
    # Folds are consecutive and non-overlapping.
    for (start, end), (next_start, _) in zip(bounds, bounds[1:]):
        assert end == next_start
    # The last fold reaches exactly to the end of the data.
    assert bounds[-1][1] == 1000
    # The first fold starts at the configured training fraction.
    assert bounds[0][0] == 500


def test_fold_bounds_reject_too_many_folds_for_the_data():
    with pytest.raises(ValueError):
        walk_forward_fold_bounds(10, n_folds=6, min_train_fraction=0.5)


def test_fold_bounds_reject_invalid_min_train_fraction():
    with pytest.raises(ValueError):
        walk_forward_fold_bounds(1000, n_folds=5, min_train_fraction=1.5)


def test_no_fold_ever_trains_on_its_own_or_a_later_folds_future():
    """The defining property of walk-forward validation: every fold's
    training set (implicitly rows [0, test_start)) ends strictly before
    that fold's own test window begins, and each fold's training window
    only grows (expanding window), never shrinks or jumps backward.
    """
    bounds = walk_forward_fold_bounds(500, n_folds=4, min_train_fraction=0.4)
    prev_train_end = -1
    for test_start, test_end in bounds:
        assert test_start > prev_train_end
        assert test_end > test_start
        prev_train_end = test_start


def test_evaluate_walk_forward_recovers_a_near_perfect_linear_signal():
    rng = np.random.default_rng(42)
    n = 1000
    X = pd.DataFrame({"x": np.arange(n, dtype=float)})
    y = pd.Series(3.0 * X["x"] + 5.0 + rng.normal(scale=0.01, size=n))

    result = evaluate_walk_forward(lambda: LinearRegression(), X, y, n_folds=5)

    assert result["n_folds"] == 5
    assert len(result["folds"]) == 5
    assert result["r2_mean"] > 0.99
    # A clean, stationary linear signal should look stable across folds.
    assert result["r2_std"] < 0.01
    for fold in result["folds"]:
        assert fold["n_train"] > 0
        assert fold["n_test"] > 0


def test_evaluate_walk_forward_flags_instability_from_a_regime_shift():
    """A model trained only on the first (stable) regime should perform
    inconsistently once the data-generating process changes partway
    through — walk-forward CV's std should catch this where a single
    80/20 split could easily miss it depending on where the split falls.
    """
    rng = np.random.default_rng(7)
    n = 1000
    x = np.arange(n, dtype=float)
    y = np.where(x < n * 0.6, 2.0 * x, -4.0 * x + 1800.0) + rng.normal(scale=1.0, size=n)
    X = pd.DataFrame({"x": x})
    y = pd.Series(y)

    result = evaluate_walk_forward(lambda: LinearRegression(), X, y, n_folds=5, min_train_fraction=0.5)

    fold_r2s = [f["r2"] for f in result["folds"]]
    # Folds before the regime shift's test window fits a clean line; folds
    # whose test window straddles or follows the shift should degrade —
    # i.e. fold performance should NOT be uniformly high across all folds.
    assert min(fold_r2s) < max(fold_r2s) - 0.05


def test_evaluate_walk_forward_accepts_a_scaling_pipeline_without_leakage():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame({"x": rng.normal(loc=100, scale=50, size=n)})
    y = pd.Series(2.0 * X["x"] + rng.normal(scale=0.1, size=n))

    result = evaluate_walk_forward(
        lambda: make_pipeline(StandardScaler(), LinearRegression()), X, y, n_folds=4
    )
    assert result["n_folds"] == 4
    assert result["r2_mean"] > 0.95


def test_evaluate_walk_forward_reports_test_period_for_pandas_index():
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    X = pd.DataFrame({"x": np.arange(500, dtype=float)}, index=idx)
    y = pd.Series(X["x"] * 2.0, index=idx)

    result = evaluate_walk_forward(lambda: LinearRegression(), X, y, n_folds=5)
    for fold in result["folds"]:
        assert "test_period" in fold
        assert len(fold["test_period"]) == 2
