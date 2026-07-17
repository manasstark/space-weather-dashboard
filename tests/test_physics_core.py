import pandas as pd
import pytest

from swdss.physics import core


def test_vbz_scalar_clips_northward_bz_to_zero():
    assert core.vbz_scalar(speed=400.0, bz=5.0) == 0.0


def test_vbz_scalar_negative_for_southward_bz():
    assert core.vbz_scalar(speed=400.0, bz=-10.0) == pytest.approx(-4000.0)


def test_vbz_series_matches_scalar_elementwise():
    speed = pd.Series([400.0, 500.0, 600.0])
    bz = pd.Series([5.0, -5.0, -10.0])
    series_result = core.vbz_series(speed, bz)
    scalar_result = [core.vbz_scalar(s, b) for s, b in zip(speed, bz)]
    assert list(series_result) == pytest.approx(scalar_result)


def test_ey_scalar_positive_for_southward_bz():
    # Southward (negative) Bz should produce a positive, geoeffective Ey.
    assert core.ey_scalar(speed=400.0, bz=-10.0) == pytest.approx(4.0)


def test_ey_series_matches_scalar_elementwise():
    speed = pd.Series([400.0, 500.0])
    bz = pd.Series([-2.0, 3.0])
    series_result = core.ey_series(speed, bz)
    scalar_result = [core.ey_scalar(s, b) for s, b in zip(speed, bz)]
    assert list(series_result) == pytest.approx(scalar_result)


def test_dynamic_pressure_scalar_known_value():
    # Pdyn = 1.6726e-6 * density * speed^2
    assert core.dynamic_pressure_scalar(density=5.0, speed=400.0) == pytest.approx(
        1.6726e-6 * 5.0 * 400.0**2
    )


def test_dynamic_pressure_series_matches_scalar_elementwise():
    density = pd.Series([1.0, 5.0, 10.0])
    speed = pd.Series([300.0, 400.0, 500.0])
    series_result = core.dynamic_pressure_series(density, speed)
    scalar_result = [core.dynamic_pressure_scalar(d, s) for d, s in zip(density, speed)]
    assert list(series_result) == pytest.approx(scalar_result)


def test_add_derived_physics_features_requires_speed_and_bz_for_vbz_ey():
    df = pd.DataFrame({"speed": [400.0], "density": [5.0]})  # no bz_gsm
    created = core.add_derived_physics_features(df)
    assert created == ["dynamic_pressure"]
    assert "vbz" not in df.columns
    assert "ey" not in df.columns
    assert "dynamic_pressure" in df.columns


def test_add_derived_physics_features_creates_all_three_when_inputs_present():
    df = pd.DataFrame({"speed": [400.0], "bz_gsm": [-5.0], "density": [5.0]})
    created = core.add_derived_physics_features(df)
    assert created == ["vbz", "ey", "dynamic_pressure"]
    assert df.loc[0, "vbz"] == pytest.approx(core.vbz_scalar(400.0, -5.0))
    assert df.loc[0, "ey"] == pytest.approx(core.ey_scalar(400.0, -5.0))
    assert df.loc[0, "dynamic_pressure"] == pytest.approx(core.dynamic_pressure_scalar(5.0, 400.0))


def test_southward_duration_resets_on_northward_turn():
    bz = pd.Series([-1.0, -2.0, -3.0, 1.0, -1.0])
    result = core.southward_duration_series(bz)
    assert list(result) == [1, 2, 3, 0, 1]


def test_strong_southward_duration_uses_stricter_threshold():
    bz = pd.Series([-2.0, -6.0, -7.0, -2.0])
    result = core.strong_southward_duration_series(bz)
    # Only samples below -5.0 (default threshold) count as "strong".
    assert list(result) == [0, 1, 2, 0]


def test_integrated_southward_bz_ignores_northward_samples():
    bz = pd.Series([-2.0, 3.0, -4.0])
    result = core.integrated_southward_bz_series(bz, window=3)
    # Rolling sum of |min(bz, 0)| over the full window.
    assert result.iloc[-1] == pytest.approx(6.0)


def test_integrated_ey_only_sums_positive_part():
    ey = pd.Series([2.0, -3.0, 4.0])
    result = core.integrated_ey_series(ey, window=3)
    assert result.iloc[-1] == pytest.approx(6.0)


def test_integrated_vbz_sums_absolute_value():
    vbz = pd.Series([-1000.0, -2000.0, 0.0])
    result = core.integrated_vbz_series(vbz, window=3)
    assert result.iloc[-1] == pytest.approx(3000.0)
