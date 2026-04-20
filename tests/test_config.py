"""
Pytest suite for simulation config validation.

Run with:  uv run pytest tests/test_config_validation.py -v
"""

import pytest
from pydantic import ValidationError
from enlight.utils.validation import validate_sim_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_config() -> dict:
    """Minimal valid simulation config matching _template.yaml."""
    return {
        "label": "test_sim",
        "run": {
            "mode": "rolling_horizon",
            "prediction_year": 2040,
            "plant_aggregation": True,
        },
        "rolling_horizon": {"start_week": 1, "end_week": 4},
        "wind_onshore":  {"weather_year": 2020, "capacity_file": "TYNDP_2024_National_Trends", "bid_price": 0.01},
        "wind_offshore": {"weather_year": 2020, "capacity_file": "TYNDP_2024_National_Trends", "bid_price": 0.02},
        "solar_pv":      {"weather_year": 2020, "capacity_file": "TYNDP_2024_National_Trends", "bid_price": 0.03},
        "hydro_ror":     {"weather_year": 2020, "capacity_file": "TYNDP_2024_National_Trends", "bid_price": 0.04},
        "hydro_res":     {"units_file": "hydro_reservoir_units", "energy_weather_year": 2020, "bid_price": "ramboll"},
        "hydro_ps":      {"units_file": "hydro_pumped_storage_units", "initial_soc": 0.5, "roundtrip_efficiency": 0.9, "fuel_projection": "ramboll"},
        "thermal":       {"units_file": "thermal_plant_units", "fuel_projection": "ramboll"},
        "lines":         {"capacity_file": "entsoe"},
        "demand_inflexible": {
            "classical":  {"profile_year": 2020, "amount_file": "TYNDP_2024_National_Trends", "voll": 5000},
            "industrial": {"profile_year": 2020, "amount_file": "TYNDP_2024_National_Trends", "voll": 5001},
            "household":  {"profile_year": 2020, "amount_file": "TYNDP_2024_National_Trends", "voll": 5002},
            "public":     {"profile_year": 2020, "amount_file": "TYNDP_2024_National_Trends", "voll": 5003},
            "ev":         {"profile_year": 2020, "amount_file": "TYNDP_2024_National_Trends", "voll": 5004},
        },
        "demand_flexible": {
            "classical":  {"amount_file": "TYNDP_2024_National_Trends", "capacity_file": "TYNDP_2024_National_Trends", "wtp": 100},
            "industrial": {"amount_file": "TYNDP_2024_National_Trends", "capacity_file": "TYNDP_2024_National_Trends", "wtp": 100},
            "household":  {"amount_file": "TYNDP_2024_National_Trends", "capacity_file": "TYNDP_2024_National_Trends", "wtp": 100},
            "public":     {"amount_file": "TYNDP_2024_National_Trends", "capacity_file": "TYNDP_2024_National_Trends", "wtp": 100},
            "ev":         {"amount_file": "TYNDP_2024_National_Trends", "capacity_file": "TYNDP_2024_National_Trends", "wtp": 100},
        },
        "bess":             {"units_file": "bess_units", "initial_soc": 0.5, "roundtrip_efficiency": 0.85},
        "ptx":              {"units_file": "ptx_units"},
        "district_heating": {"units_file": "district_heating_units"},
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_config_passes(valid_config):
    result = validate_sim_config(valid_config)
    assert result.label == "test_sim"
    assert result.run.mode == "rolling_horizon"
    assert result.rolling_horizon.end_week == 4


def test_valid_yearly_config_passes(valid_config):
    valid_config["run"]["mode"] = "yearly"
    valid_config.pop("rolling_horizon")   # not required for yearly
    result = validate_sim_config(valid_config)
    assert result.run.mode == "yearly"
    assert result.rolling_horizon is None


# ---------------------------------------------------------------------------
# run.mode
# ---------------------------------------------------------------------------

def test_invalid_run_mode_raises(valid_config):
    valid_config["run"]["mode"] = "monthly"
    with pytest.raises(ValidationError, match="mode"):
        validate_sim_config(valid_config)


# ---------------------------------------------------------------------------
# rolling_horizon cross-field logic
# ---------------------------------------------------------------------------

def test_rolling_horizon_missing_when_required(valid_config):
    valid_config.pop("rolling_horizon")
    with pytest.raises(ValidationError, match="rolling_horizon"):
        validate_sim_config(valid_config)


def test_rolling_horizon_end_before_start(valid_config):
    valid_config["rolling_horizon"] = {"start_week": 10, "end_week": 5}
    with pytest.raises(ValidationError, match="end_week"):
        validate_sim_config(valid_config)


def test_rolling_horizon_week_out_of_range(valid_config):
    valid_config["rolling_horizon"] = {"start_week": 0, "end_week": 4}
    with pytest.raises(ValidationError, match="start_week"):
        validate_sim_config(valid_config)


# ---------------------------------------------------------------------------
# prediction_year
# ---------------------------------------------------------------------------

def test_non_milestone_prediction_year_raises(valid_config):
    valid_config["run"]["prediction_year"] = 2041
    with pytest.raises(ValidationError, match="prediction_year"):
        validate_sim_config(valid_config)


def test_prediction_year_out_of_range(valid_config):
    valid_config["run"]["prediction_year"] = 2100
    with pytest.raises(ValidationError, match="prediction_year"):
        validate_sim_config(valid_config)


# ---------------------------------------------------------------------------
# bid prices / efficiencies
# ---------------------------------------------------------------------------

def test_negative_bid_price_raises(valid_config):
    valid_config["wind_onshore"]["bid_price"] = -0.5
    with pytest.raises(ValidationError, match="bid_price"):
        validate_sim_config(valid_config)


def test_soc_above_one_raises(valid_config):
    valid_config["bess"]["initial_soc"] = 1.5
    with pytest.raises(ValidationError, match="initial_soc"):
        validate_sim_config(valid_config)


def test_roundtrip_above_one_raises(valid_config):
    valid_config["hydro_ps"]["roundtrip_efficiency"] = 1.1
    with pytest.raises(ValidationError, match="roundtrip_efficiency"):
        validate_sim_config(valid_config)


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

def test_missing_label_raises(valid_config):
    valid_config.pop("label")
    with pytest.raises(ValidationError, match="label"):
        validate_sim_config(valid_config)


def test_missing_thermal_section_raises(valid_config):
    valid_config.pop("thermal")
    with pytest.raises(ValidationError, match="thermal"):
        validate_sim_config(valid_config)