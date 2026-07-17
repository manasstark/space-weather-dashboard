import pandas as pd
import pytest

from swdss.physics import coupling, geometry


def test_clock_angle_purely_northward_is_zero():
    assert geometry.clock_angle_scalar(by=0.0, bz=10.0) == pytest.approx(0.0)


def test_clock_angle_purely_southward_is_180():
    assert geometry.clock_angle_scalar(by=0.0, bz=-10.0) == pytest.approx(180.0)


def test_clock_angle_wraps_to_0_360_range():
    series = geometry.clock_angle_series(pd.Series([-1.0, 1.0]), pd.Series([1.0, -1.0]))
    assert series.between(0, 360).all()


def test_clock_angle_rate_takes_shortest_signed_path_across_wrap():
    # 359deg -> 1deg is a +2deg rotation, not -358deg.
    clock_angle = pd.Series([359.0, 1.0])
    rate = geometry.clock_angle_rate_series(clock_angle)
    assert rate.iloc[1] == pytest.approx(2.0)


def test_magnetic_shear_is_vector_magnitude_of_change():
    bx = pd.Series([0.0, 3.0])
    by = pd.Series([0.0, 0.0])
    bz = pd.Series([0.0, 4.0])
    result = geometry.magnetic_shear_series(bx, by, bz)
    assert result.iloc[1] == pytest.approx(5.0)  # 3-4-5 triangle


def test_imf_rotation_rate_sums_absolute_rate_over_window():
    clock_angle_rate = pd.Series([2.0, -3.0, 4.0])
    result = geometry.imf_rotation_rate_series(clock_angle_rate, window=3)
    assert result.iloc[-1] == pytest.approx(9.0)


def test_newell_coupling_zero_at_purely_northward_bz():
    # sin(theta_c/2) = 0 when clock angle is 0 (purely northward) -> zero coupling.
    result = coupling.newell_coupling_series(
        pd.Series([400.0]), pd.Series([0.0]), pd.Series([10.0])
    )
    assert result.iloc[0] == pytest.approx(0.0)


def test_newell_coupling_maximal_at_purely_southward_bz():
    # Purely southward (by=0, bz<0) gives clock angle 180deg -> sin(90deg) = 1, the max factor.
    speed, by, bz = 400.0, 0.0, -10.0
    result = coupling.newell_coupling_series(pd.Series([speed]), pd.Series([by]), pd.Series([bz]))
    expected = speed ** (4 / 3) * 10.0 ** (2 / 3)
    assert result.iloc[0] == pytest.approx(expected)


def test_akasofu_epsilon_clips_negative_speed_and_bt():
    result = coupling.akasofu_epsilon_series(pd.Series([-100.0]), pd.Series([-5.0]), pd.Series([180.0]))
    assert result.iloc[0] == pytest.approx(0.0)


def test_akasofu_epsilon_known_value():
    speed, bt, clock_angle = 500.0, 10.0, 180.0
    result = coupling.akasofu_epsilon_series(pd.Series([speed]), pd.Series([bt]), pd.Series([clock_angle]))
    expected = coupling.AKASOFU_WATTS_CONSTANT * speed * bt**2 * 1.0  # sin(90deg)^4 == 1
    assert result.iloc[0] == pytest.approx(expected)


def test_boyle_index_known_value_at_purely_southward_bz():
    speed, bt, clock_angle = 400.0, 10.0, 180.0
    result = coupling.boyle_index_series(pd.Series([speed]), pd.Series([bt]), pd.Series([clock_angle]))
    expected = 1e-4 * speed**2 + 11.7 * bt * 1.0  # sin(90deg)^3 == 1
    assert result.iloc[0] == pytest.approx(expected)


def test_integrated_energy_input_is_rolling_sum():
    epsilon = pd.Series([1.0, 2.0, 3.0])
    result = coupling.integrated_energy_input_series(epsilon, window=3)
    assert result.iloc[-1] == pytest.approx(6.0)
