"""Standalone worker for training Keras LSTM/GRU models — MUST run in its
own fresh Python process, invoked as:

    python -m swdss.models.imf_research_keras_worker <input.npz> <meta.json> <output.json>

This file must NEVER be imported into the main dashboard process. TF's
model.fit was empirically found to hang indefinitely (reproduced with
plain synthetic data, unaffected by import order, OMP_NUM_THREADS=1,
KMP_DUPLICATE_LIB_OK=TRUE, or explicit tf.config.threading limits) when
scikit-learn, XGBoost, LightGBM, and CatBoost are all loaded in the same
process as TensorFlow — the exact set imf_research.py needs for its
tabular models. Rather than depend on a fragile, only-partially-
understood workaround, Keras training runs here instead: a subprocess
that imports TensorFlow and nothing else ML-related, sidestepping the
conflict entirely regardless of its root cause.
"""

import json
import sys


def main() -> None:
    input_path, meta_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    import numpy as np

    data = np.load(input_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    with open(meta_path) as f:
        meta = json.load(f)

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    from tensorflow import keras

    layer_cls = keras.layers.LSTM if meta["model_type"] == "LSTM" else keras.layers.GRU
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(meta["seq_len"], meta["n_features"])),
            layer_cls(meta["units"], dropout=meta["dropout"]),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=meta["epochs"],
        batch_size=meta["batch_size"],
        verbose=0,
    )
    preds = model.predict(X_test, verbose=0).reshape(-1)
    model.save(meta["model_path"])

    result = {
        "preds": [float(v) for v in preds],
        "loss": [float(v) for v in history.history.get("loss", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
    }
    with open(output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
