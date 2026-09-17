from typing import Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Sub-models — run
# ---------------------------------------------------------------------------

class RollingHorizonConfig(BaseModel):
    start_week:          int = Field(ge=1, le=52)
    end_week:            int = Field(ge=1, le=52)
    keep_weekly_results: bool = False

    @model_validator(mode="after")
    def end_not_before_start(self) -> "RollingHorizonConfig":
        if self.end_week < self.start_week:
            raise ValueError(
                f"end_week ({self.end_week}) must be >= start_week ({self.start_week})"
            )
        return self


class RunConfig(BaseModel):
    mode:            Literal["yearly", "rolling_horizon"]
    prediction_year: int = Field(ge=2020, le=2060)
    solver:          Literal["highs", "gurobi"]


# ---------------------------------------------------------------------------
# Sub-models — supply curve
# ---------------------------------------------------------------------------

class WeatherDataConfig(BaseModel):
    source: str
    year:   int = Field(ge=1980, le=2030)


class VREConfig(BaseModel):
    """Config shared by wind_onshore, wind_offshore, solar_pv, hydro_ror."""
    bid_price:     float = Field(ge=0)
    capacity_file: str
    weather_data:  WeatherDataConfig


class HydroResConfig(BaseModel):
    units_file:          str
    plant_aggregation:   bool
    ramp_constraints:    bool
    energy_weather_year: int = Field(ge=1980, le=2030)
    bid_price:           Union[float, str]

    @field_validator("bid_price")
    @classmethod
    def bid_price_valid(cls, v: Union[float, str]) -> Union[float, str]:
        if isinstance(v, str) and v != "demo":
            raise ValueError(f"bid_price must be a number or 'demo', got '{v}'")
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError(f"bid_price must be >= 0, got {v}")
        return v


class HydroPsConfig(BaseModel):
    units_file:           str
    plant_aggregation:    bool
    initial_soc:          float = Field(ge=0, le=1)
    roundtrip_efficiency: float = Field(gt=0, le=1)
    fuel_projection:      str


class BessConfig(BaseModel):
    units_file:           str
    initial_soc:          float = Field(ge=0, le=1)
    roundtrip_efficiency: float = Field(gt=0, le=1)


class MarginalCostConfig(BaseModel):
    """Datasets under data/fuel_price_projections/ used to compute thermal marginal costs."""
    fuel_prices:        str
    co2_quota_prices:   str
    fuel_price_profile: Optional[str] = None  # TODO: monthly fuel price shape, not implemented yet
    taxes:              str


class ThermalConfig(BaseModel):
    units_file:        str
    plant_aggregation: bool
    ramp_constraints:  bool
    marginal_cost:     MarginalCostConfig


class SupplyCurveConfig(BaseModel):
    wind_onshore:  VREConfig
    wind_offshore: VREConfig
    solar_pv:      VREConfig
    hydro_ror:     VREConfig
    hydro_res:     HydroResConfig
    hydro_ps:      HydroPsConfig
    thermal:       ThermalConfig
    bess:          BessConfig


# ---------------------------------------------------------------------------
# Sub-models — demand curve
# ---------------------------------------------------------------------------

class InflexibleDemandConfig(BaseModel):
    profile_year: int = Field(ge=1980, le=2030)
    amount_file:  str
    voll:         float = Field(ge=0)


class FlexibleDemandConfig(BaseModel):
    flex_amount:   str
    flex_capacity: str
    wtp:           float = Field(ge=0)


class UnitsConfig(BaseModel):
    units_file: str


class DemandCurveConfig(BaseModel):
    classical_inflex:  InflexibleDemandConfig
    industrial_inflex: InflexibleDemandConfig
    household_inflex:  InflexibleDemandConfig
    public_inflex:     InflexibleDemandConfig
    ev_inflex:         InflexibleDemandConfig

    classical_flex:  FlexibleDemandConfig
    industrial_flex: FlexibleDemandConfig
    household_flex:  FlexibleDemandConfig
    public_flex:     FlexibleDemandConfig
    ev_flex:         FlexibleDemandConfig

    ptx:              UnitsConfig
    district_heating: UnitsConfig


class LinesConfig(BaseModel):
    capacity_file: str


# ---------------------------------------------------------------------------
# Top-level simulation config
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label:           str
    run:             RunConfig
    rolling_horizon: Optional[RollingHorizonConfig] = None
    supply_curve:    SupplyCurveConfig
    demand_curve:    DemandCurveConfig
    lines:           LinesConfig
    bidding_zones:   list[str]

    @model_validator(mode="after")
    def rolling_horizon_required_for_mode(self) -> "SimulationConfig":
        if self.run.mode == "rolling_horizon" and self.rolling_horizon is None:
            raise ValueError(
                "rolling_horizon block is required when run.mode is 'rolling_horizon'"
            )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_simulation_config(sim_cfg: DictConfig) -> SimulationConfig:
    """
    Parse and validate a simulation DictConfig against the full schema.

    Converts the Hydra DictConfig to a plain dict, then runs it through
    Pydantic. Raises pydantic.ValidationError with structured field-level
    messages on failure.

    Args:
        sim_cfg: The simulations sub-config from the Hydra DictConfig.

    Returns:
        A fully-validated SimulationConfig instance.
    """
    raw = OmegaConf.to_container(sim_cfg, resolve=True)
    return SimulationConfig.model_validate(raw)
