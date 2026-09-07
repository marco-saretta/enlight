from typing import Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class RollingHorizonConfig(BaseModel):
    start_week: int = Field(ge=1, le=52)
    end_week:   int = Field(ge=1, le=52)

    @model_validator(mode="after")
    def end_not_before_start(self) -> "RollingHorizonConfig":
        if self.end_week < self.start_week:
            raise ValueError(
                f"end_week ({self.end_week}) must be >= start_week ({self.start_week})"
            )
        return self


class RunConfig(BaseModel):
    mode:              Literal["yearly", "rolling_horizon"]
    prediction_year:   int = Field(ge=2020, le=2060)
    plant_aggregation: bool


class VREConfig(BaseModel):
    """Config shared by wind_onshore, wind_offshore, solar_pv, hydro_ror."""
    weather_year:  int = Field(ge=1980, le=2030)
    capacity_file: str
    bid_price:     float = Field(ge=0)


class HydroResConfig(BaseModel):
    units_file:          str
    energy_weather_year: int = Field(ge=1980, le=2030)
    bid_price:           Union[float, str]

    @field_validator("bid_price")
    @classmethod
    def bid_price_valid(cls, v: Union[float, str]) -> Union[float, str]:
        if isinstance(v, str) and v != "ramboll":
            raise ValueError(f"bid_price must be a number or 'ramboll', got '{v}'")
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError(f"bid_price must be >= 0, got {v}")
        return v


class StorageConfig(BaseModel):
    """Config shared by hydro_ps and bess."""
    units_file:           str
    initial_soc:          float = Field(ge=0, le=1)
    roundtrip_efficiency: float = Field(gt=0, le=1)
    fuel_projection:      Optional[str] = None


class ThermalConfig(BaseModel):
    units_file:      str
    fuel_projection: str


class LinesConfig(BaseModel):
    capacity_file: str


class DemandCategoryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_year:  Optional[int]   = None
    amount_file:   str
    capacity_file: Optional[str]   = None
    voll:          Optional[float] = None
    wtp:           Optional[float] = None


class DemandConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    classical:  DemandCategoryConfig
    industrial: DemandCategoryConfig
    household:  DemandCategoryConfig
    public:     DemandCategoryConfig
    ev:         DemandCategoryConfig


class UnitsConfig(BaseModel):
    units_file: str


# ---------------------------------------------------------------------------
# Top-level simulation config
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label:             str
    run:               RunConfig
    rolling_horizon:   Optional[RollingHorizonConfig] = None
    wind_onshore:      VREConfig
    wind_offshore:     VREConfig
    solar_pv:          VREConfig
    hydro_ror:         VREConfig
    hydro_res:         HydroResConfig
    hydro_ps:          StorageConfig
    thermal:           ThermalConfig
    lines:             LinesConfig
    demand_inflexible: DemandConfig
    demand_flexible:   DemandConfig
    bess:              StorageConfig
    ptx:               UnitsConfig
    district_heating:  UnitsConfig

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