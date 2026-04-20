from pydantic import BaseModel, Field, validator
from typing import Literal, Optional

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
    mode:             Literal["yearly", "rolling_horizon"]
    year:             int = Field(ge=2020, le=2060)
    plant_aggregation: bool
    rolling_horizon:  Optional[RollingHorizonConfig] = None

    @validator("rolling_horizon", always=True)
    def rolling_horizon_required_if_mode(cls, v, values):
        if values.get("mode") == "rolling_horizon" and v is None:
            raise ValueError(
                "rolling_horizon settings are required when mode is 'rolling_horizon'"
            )
        return v

class WindConfig(BaseModel):
    weather_year:  int
    capacity_file: str
    bid_price:     float = Field(ge=0)

class SimulationConfig(BaseModel):
    label: str
    run:   RunConfig
    wind_onshore:  WindConfig
    wind_offshore: WindConfig
    solar_pv:      WindConfig   # same shape
    # ... add others as needed