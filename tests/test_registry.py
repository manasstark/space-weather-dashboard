import pytest

from swdss.models import registry


@pytest.mark.parametrize("dataset", list(registry.DATASETS.keys()))
def test_every_variable_has_a_label_and_unit(dataset):
    config = registry.DATASETS[dataset]
    variables = config.feature_variables or config.variables
    for variable in variables:
        if variable == "predicted_ae":
            continue  # synthesized live, not a raw registry variable
        assert variable in registry.VARIABLE_LABELS, f"{variable} missing a label"
        assert variable in registry.VARIABLE_UNITS, f"{variable} missing a unit"


def test_owning_source_config_resolves_multi_source_dataset():
    config = registry.owning_source_config("analytics", "kp")
    assert config.name == "kp"


def test_owning_source_config_single_source_returns_self():
    config = registry.owning_source_config("solar_wind", "speed")
    assert config.name == "solar_wind"


def test_owning_source_config_unknown_variable_raises():
    with pytest.raises(KeyError):
        registry.owning_source_config("analytics", "not_a_real_variable")


def test_raw_column_for_resolves_through_source_dataset():
    # "analytics" doesn't own "speed" directly — it must resolve through
    # the "solar_wind" source dataset's raw_column_map.
    assert registry.raw_column_for("analytics", "speed") == "solar_wind_speed"


def test_raw_column_for_direct_dataset():
    assert registry.raw_column_for("kp", "kp") == "kp"


def test_model_path_follows_naming_convention(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    path = registry.model_path("solar_wind", "speed", 3)
    assert path.name == "speed_3h.joblib"
    assert path.parent.name == "solar_wind"


def test_kp_interval_model_path_is_not_horizon_based(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    path = registry.kp_interval_model_path("analytics")
    assert path.name == "kp_interval.joblib"


def test_analytics_never_reads_predicted_ae():
    # Version 2 of the AE integration (see README) explicitly requires the
    # Analytics production model to only ever consume observed/historical
    # AE, never the AE tab's own predicted output.
    assert "predicted_ae" not in registry.ANALYTICS_FEATURE_VARIABLES
    assert "ae" in registry.ANALYTICS_FEATURE_VARIABLES


def test_experimental_uses_predicted_ae_not_observed_ae():
    # The cascaded (Version 3) research pipeline is the only one allowed
    # to consume predicted_ae, and must not also read observed "ae".
    assert "predicted_ae" in registry.EXPERIMENTAL_FEATURE_VARIABLES
    assert "ae" not in registry.EXPERIMENTAL_FEATURE_VARIABLES
