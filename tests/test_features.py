import pandas as pd
import pytest

from swdss.models import features


def test_add_lag_features_shifts_by_correct_amount():
    df = pd.DataFrame({"speed": [1.0, 2.0, 3.0, 4.0, 5.0]})
    created = features.add_lag_features(df, ["speed"], lags=[1, 2])
    assert created == ["speed_lag1h", "speed_lag2h"]
    assert df["speed_lag1h"].isna().sum() == 1
    assert list(df["speed_lag1h"].iloc[1:]) == [1.0, 2.0, 3.0, 4.0]
    assert list(df["speed_lag2h"].iloc[2:]) == [1.0, 2.0, 3.0]


def test_add_rolling_features_computes_mean_and_std():
    df = pd.DataFrame({"speed": [1.0, 2.0, 3.0, 4.0]})
    created = features.add_rolling_features(df, ["speed"], window=2)
    assert created == ["speed_2h", "speed_2h_std"]
    assert df["speed_2h"].iloc[1] == pytest.approx(1.5)
    assert df["speed_2h"].iloc[3] == pytest.approx(3.5)


def test_add_change_features_is_first_difference():
    df = pd.DataFrame({"speed": [1.0, 3.0, 6.0]})
    created = features.add_change_features(df, ["speed"])
    assert created == ["speed_change"]
    assert df["speed_change"].iloc[1] == pytest.approx(2.0)
    assert df["speed_change"].iloc[2] == pytest.approx(3.0)


def test_add_derived_physics_features_delegates_to_physics_core():
    df = pd.DataFrame({"speed": [400.0], "bz_gsm": [-5.0], "density": [5.0]})
    created = features.add_derived_physics_features(df)
    assert created == ["vbz", "ey", "dynamic_pressure"]
    assert {"vbz", "ey", "dynamic_pressure"}.issubset(df.columns)


def test_build_feature_frame_preserves_base_columns_and_order():
    df = pd.DataFrame({"speed": [1.0, 2.0, 3.0]})
    frame, feature_columns = features.build_feature_frame(df, ["speed"])

    assert feature_columns[0] == "speed"
    assert all(col.startswith("speed") for col in feature_columns)
    # base + 5 lags + 2 rolling + 1 change = 9 feature columns total.
    assert len(feature_columns) == 1 + len(features.LAGS) + 2 + 1
    assert list(frame.columns) == feature_columns


def test_build_feature_frame_does_not_mutate_input():
    df = pd.DataFrame({"speed": [1.0, 2.0, 3.0]})
    original_columns = list(df.columns)
    features.build_feature_frame(df, ["speed"])
    assert list(df.columns) == original_columns
