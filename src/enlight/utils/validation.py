from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from omegaconf import DictConfig


class RollingHorizonConfig(BaseModel):
    start_week: int = Field(ge=1, le=52)
    end_week:   int = Field(ge=1, le=52)

    @validator("end_week")
    def end_after_start(cls, v, values):
        if "start_week" in values and v < values["start_week"]:
            raise ValueError(
                f"end_week ({v}) must be >= start_week ({values['start_week']})"
            )
        return v


class RunConfig(BaseModel):
    mode:              Literal["yearly", "rolling_horizon"]
    prediction_year:   int = Field(ge=2020, le=2060)
    plant_aggregation: bool


class WindConfig(BaseModel):
    weather_year:  int
    capacity_file: str
    bid_price:     float = Field(ge=0)


class SimulationConfig(BaseModel):
    class Config:
        extra = "ignore"   # tolerate keys not yet covered by the schema

    label:           str
    run:             RunConfig
    rolling_horizon: Optional[RollingHorizonConfig] = None
    wind_onshore:    WindConfig
    wind_offshore:   WindConfig
    solar_pv:        WindConfig

    @validator("rolling_horizon", always=True)
    def rolling_horizon_required_when_mode_matches(cls, v, values):
        run = values.get("run")
        if run and run.mode == "rolling_horizon" and v is None:
            raise ValueError(
                "rolling_horizon block is required when run.mode is 'rolling_horizon'"
            )
        return v


def validate_simulation_config(cfg: DictConfig) -> None:
    """
    Validate a simulation DictConfig before the pipeline starts.
    Raises ValueError with a clear message on the first problem found.
    """
    label = cfg.label

    def err(msg: str) -> ValueError:
        return ValueError(f"[{label}] {msg}")

    # Run block
    mode = cfg.run.mode
    if mode not in ("yearly", "rolling_horizon"):
        raise err(f"run.mode must be 'yearly' or 'rolling_horizon', got '{mode}'")

    year = cfg.run.prediction_year
    if not (2020 <= year <= 2060):
        raise err(f"run.prediction_year must be between 2020 and 2060, got {year}")

    # Rolling horizon block
    if mode == "rolling_horizon":
        if not hasattr(cfg, "rolling_horizon"):
            raise err("rolling_horizon block is required when run.mode is 'rolling_horizon'")
        rh = cfg.rolling_horizon
        if not (1 <= rh.start_week <= 52):
            raise err(f"rolling_horizon.start_week must be 1-52, got {rh.start_week}")
        if not (1 <= rh.end_week <= 52):
            raise err(f"rolling_horizon.end_week must be 1-52, got {rh.end_week}")
        if rh.end_week < rh.start_week:
            raise err(
                f"rolling_horizon.end_week ({rh.end_week}) must be "
                f">= start_week ({rh.start_week})"
            )

    # Bid prices (VRE) — skip string values like "ramboll"
    for tech in ("wind_onshore", "wind_offshore", "solar_pv", "hydro_ror"):
        price = getattr(cfg, tech).bid_price
        if isinstance(price, (int, float)) and price < 0:
            raise err(f"{tech}.bid_price must be >= 0, got {price}")

    # Storage efficiencies
    for unit in ("hydro_ps", "bess"):
        eff = getattr(cfg, unit).roundtrip_efficiency
        if not (0 < eff <= 1):
            raise err(f"{unit}.roundtrip_efficiency must be in (0, 1], got {eff}")

    # Storage initial SOC
    for unit in ("hydro_ps", "bess"):
        soc = getattr(cfg, unit).initial_soc
        if not (0 <= soc <= 1):
            raise err(f"{unit}.initial_soc must be in [0, 1], got {soc}")
