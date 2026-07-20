"""Walk-forward (rolling-origin) cross-validation for time-ordered regression.

Every model in this project — production, all three Research Laboratories,
and every Optimization Study experiment — was previously benchmarked with a
single chronological 80/20 train/test split (still available via
swdss.models.train.evaluate_split and its per-lab equivalents). A single
split answers "how did the model do on one slice of recent history" but
gives no way to tell whether that number is a stable estimate or an
artifact of whatever happened to fall in that one slice — for a domain
where the events that actually matter (geomagnetic storms) are rare, that
distinction is exactly the kind of gap a space weather researcher would
flag first.

Walk-forward CV instead trains on an expanding window and evaluates on
several consecutive held-out folds further out in time, each fold's
training set strictly preceding its own test window (no leakage — fold
2's training set is a superset of fold 1's, standard rolling-origin
practice), and reports the mean and standard deviation of R²/MAE/RMSE
across folds. This is purely an additional, opt-in evaluation layer: it
does not replace the existing single-split metrics anywhere they're
already stored (see each caller's docstring), and the deployed model
itself is still refit on the full dataset exactly as before.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_N_FOLDS = 5
DEFAULT_MIN_TRAIN_FRACTION = 0.5


def walk_forward_fold_bounds(
    n_samples: int, n_folds: int = DEFAULT_N_FOLDS, min_train_fraction: float = DEFAULT_MIN_TRAIN_FRACTION
) -> list[tuple[int, int]]:
    """Returns `n_folds` consecutive, non-overlapping (test_start, test_end)
    index bounds carved out of the last (1 - min_train_fraction) share of
    `n_samples` rows. Fold i's implied training set is every row strictly
    before test_start_i — an expanding window, since fold i+1's training
    set is a strict superset of fold i's — so no fold ever trains on data
    from its own or a later fold's future.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if not (0.0 < min_train_fraction < 1.0):
        raise ValueError("min_train_fraction must be between 0 and 1")

    first_test_start = int(n_samples * min_train_fraction)
    remaining = n_samples - first_test_start
    if remaining < n_folds:
        raise ValueError(
            f"Not enough rows ({n_samples}) for {n_folds} walk-forward folds "
            f"at min_train_fraction={min_train_fraction} — only {remaining} rows remain for testing."
        )

    fold_size = remaining // n_folds
    bounds = []
    for i in range(n_folds):
        test_start = first_test_start + i * fold_size
        test_end = n_samples if i == n_folds - 1 else test_start + fold_size
        bounds.append((test_start, test_end))
    return bounds


def evaluate_walk_forward(
    model_factory,
    X,
    y,
    n_folds: int = DEFAULT_N_FOLDS,
    min_train_fraction: float = DEFAULT_MIN_TRAIN_FRACTION,
) -> dict:
    """Walk-forward cross-validation of a regression model.

    `model_factory` must be a zero-argument callable returning a fresh,
    unfitted scikit-learn-compatible estimator (a plain estimator, or a
    Pipeline — e.g. `make_pipeline(StandardScaler(), model)` for
    scale-sensitive algorithms, so any scaler is fit fresh on each fold's
    own training rows rather than leaking test-fold statistics in from a
    scaler fit once over the whole dataset).

    `X`/`y` must already be time-ordered (oldest row first) and contain no
    NaNs. Accepts either pandas or numpy input.

    Returns {"n_folds", "folds": [{"n_train", "n_test", "test_period"
    (if X has a usable index), "r2", "mae", "rmse"}, ...], "r2_mean",
    "r2_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std"}.
    """
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    index = X.index if isinstance(X, pd.DataFrame) else None

    bounds = walk_forward_fold_bounds(len(X_arr), n_folds=n_folds, min_train_fraction=min_train_fraction)

    folds = []
    for test_start, test_end in bounds:
        X_train, y_train = X_arr[:test_start], y_arr[:test_start]
        X_test, y_test = X_arr[test_start:test_end], y_arr[test_start:test_end]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold = {
            "n_train": int(test_start),
            "n_test": int(test_end - test_start),
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        }
        if index is not None:
            fold["test_period"] = [str(index[test_start]), str(index[test_end - 1])]
        folds.append(fold)

    r2s = np.array([f["r2"] for f in folds])
    maes = np.array([f["mae"] for f in folds])
    rmses = np.array([f["rmse"] for f in folds])

    return {
        "n_folds": len(folds),
        "folds": folds,
        "r2_mean": float(r2s.mean()),
        "r2_std": float(r2s.std(ddof=0)),
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=0)),
    }
