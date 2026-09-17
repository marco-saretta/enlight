from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

import enlight.utils as utils

log = utils.get_logger(__name__)

HOURS_PER_WEEK = 168
N_WEEKS = 52  # week 52 also takes the last 24 h of the year


@dataclass
class VREData:
    """Variable renewable source, aggregated per bidding zone."""

    production: xr.DataArray  # available production [MW], dims (T, Z)
    bid_price: float          # offer price [EUR/MWh]


@dataclass
class InflexibleDemandData:
    """Price-inelastic demand, aggregated per bidding zone."""

    demand: xr.DataArray  # demand [MW], dims (T, Z)
    voll: float           # value of lost load [EUR/MWh]


@dataclass
class HydroReservoirData:
    """Reservoir hydro units and the energy each zone's reservoirs can release per week."""

    zone: xr.DataArray           # bidding zone of each unit, dims (G,)
    capacity: xr.DataArray       # [MW], dims (G,)
    marginal_cost: xr.DataArray  # [EUR/MWh], dims (G,)
    weekly_energy: xr.DataArray  # [MWh], dims (W, Z); only the loaded week in rolling_horizon mode


@dataclass
class ThermalData:
    """Thermal units (one row per unit, or per zone and technology when aggregated)."""

    zone: xr.DataArray           # bidding zone of each unit, dims (G,)
    capacity: xr.DataArray       # [MW], dims (G,)
    marginal_cost: xr.DataArray  # [EUR/MWh], dims (G,)


@dataclass
class LinesData:
    """Interconnectors between active bidding zones; positive flow = from_zone -> to_zone."""

    from_zone: xr.DataArray         # dims (L,)
    to_zone: xr.DataArray           # dims (L,)
    capacity_from_to: xr.DataArray  # [MW], dims (T, L)
    capacity_to_from: xr.DataArray  # [MW], dims (T, L)


@dataclass
class DataLoader:
    """
    Load the preprocessed simulation data (simulations/<label>/data/) into
    named-dimension xarray objects, one attribute per technology.

    Dimensions: T = hour, Z = bidding zone, G = thermal unit, L = transmission line.
    In rolling_horizon mode only the hours of `week` are loaded.

    Every technology has a loader method below so the class reads as a full
    map of the pipeline; the renewables, thermal, classical_inflex demand
    and lines are implemented so far.
    """

    cfg: DictConfig
    week: int | None = None

    def __post_init__(self) -> None:
        label = self.cfg.simulations.label
        self.data_path = Path(self.cfg.paths.processed) / label / "data"
        self.times: pd.Index | None = None  # hours shared by every hourly series, set on first load

        # Supply curve — variable renewables
        self._load_wind_onshore()
        self._load_wind_offshore()
        self._load_solar_pv()
        self._load_hydro_ror()

        # Supply curve — unit-based dispatchable plants
        self._load_hydro_res()
        self._load_hydro_ps()
        self._load_thermal()
        self._load_bess()

        # Demand curve
        self._load_demand_inflexible()
        self._load_demand_flexible()
        self._load_ptx()
        self._load_district_heating()

        # Transmission
        self._load_lines()

        # Week of every loaded hour (hours are numbered 1-8760), for weekly constraints
        self.week_of_hour = xr.DataArray(
            np.minimum((self.times - 1) // HOURS_PER_WEEK + 1, N_WEEKS), coords={"T": self.times}, name="W"
        )

        log.info(
            "%d hours, %d zones, %d thermal units, %d hydro reservoir units, %d lines",
            len(self.times), len(self.cfg.simulations.bidding_zones), self.thermal.zone.size,
            self.hydro_res.zone.size, self.lines.from_zone.size,
        )

    # -------------------------------------------------------------------
    # Supply curve — variable renewables
    # -------------------------------------------------------------------
    def _load_wind_onshore(self) -> None:
        """wind_onshore_production.csv + bid_price -> self.wind_onshore."""
        self.wind_onshore = self._load_vre("wind_onshore")

    def _load_wind_offshore(self) -> None:
        """wind_offshore_production.csv + bid_price -> self.wind_offshore."""
        self.wind_offshore = self._load_vre("wind_offshore")

    def _load_solar_pv(self) -> None:
        """solar_pv_production.csv + bid_price -> self.solar_pv."""
        self.solar_pv = self._load_vre("solar_pv")

    def _load_hydro_ror(self) -> None:
        """hydro_ror_production.csv + bid_price -> self.hydro_ror."""
        self.hydro_ror = self._load_vre("hydro_ror")

    # -------------------------------------------------------------------
    # Supply curve — unit-based dispatchable plants
    # Plan: one static table per technology (one row per unit: zone,
    # capacity, cost), kept 1-D over units. The model maps units to zones
    # with groupby on the zone column — no unit-to-zone incidence matrix.
    # -------------------------------------------------------------------
    def _load_hydro_res(self) -> None:
        """hydro_res_units.csv + hydro_res_energy.csv -> self.hydro_res."""
        units = pd.read_csv(self.data_path / "hydro_res_units.csv", index_col=0)
        units.index.name = "G"

        energy = pd.read_csv(self.data_path / "hydro_res_energy.csv", index_col=0)
        energy.index.name, energy.columns.name = "W", "Z"
        if self.cfg.simulations.run.mode == "rolling_horizon":
            energy = energy.loc[[self.week]]

        self.hydro_res = HydroReservoirData(
            zone=xr.DataArray(units["zone_el"]),
            capacity=xr.DataArray(units["capacity_el"]),
            marginal_cost=xr.DataArray(units["prodcost"]),
            weekly_energy=xr.DataArray(energy),
        )

    def _load_hydro_ps(self) -> None:
        """TODO: hydro_pumped_storage_units.csv."""
        pass

    def _load_thermal(self) -> None:
        """thermal_units.csv -> self.thermal."""
        units = pd.read_csv(self.data_path / "thermal_units.csv", index_col=0)
        units.index.name = "G"
        self.thermal = ThermalData(
            zone=xr.DataArray(units["zone"]),
            capacity=xr.DataArray(units["capacity"]),
            marginal_cost=xr.DataArray(units["marginal_cost"]),
        )

    def _load_bess(self) -> None:
        """TODO: bess_units.csv."""
        pass

    # -------------------------------------------------------------------
    # Demand curve
    # -------------------------------------------------------------------
    def _load_demand_inflexible(self) -> None:
        """demand_<category>.csv + voll -> self.demand_inflexible[category]."""
        # TODO: ev_inflex is already preprocessed; industrial/household/public have no data yet.
        categories = ["classical_inflex"]
        self.demand_inflexible = {
            category: InflexibleDemandData(
                demand=self._load_timeseries(f"demand_{category}.csv", column_dim="Z"),
                voll=self.cfg.simulations.demand_curve[category].voll,
            )
            for category in categories
        }

    def _load_demand_flexible(self) -> None:
        """TODO: per _flex category; weekly amounts via groupby on the week of each hour."""
        pass

    def _load_ptx(self) -> None:
        """TODO: ptx_units.csv."""
        pass

    def _load_district_heating(self) -> None:
        """TODO: district_heating_units.csv."""
        pass

    # -------------------------------------------------------------------
    # Transmission
    # -------------------------------------------------------------------
    def _load_lines(self) -> None:
        """lines.csv + lines_capacity_{from_to,to_from}.csv -> self.lines."""
        lines = pd.read_csv(self.data_path / "lines.csv", index_col=0)
        lines.index.name = "L"
        self.lines = LinesData(
            from_zone=xr.DataArray(lines["from_zone"]),
            to_zone=xr.DataArray(lines["to_zone"]),
            capacity_from_to=self._load_timeseries("lines_capacity_from_to.csv", column_dim="L"),
            capacity_to_from=self._load_timeseries("lines_capacity_to_from.csv", column_dim="L"),
        )

    # -------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------
    def _load_vre(self, tech: str) -> VREData:
        """Available production and bid price for one variable renewable technology."""
        return VREData(
            production=self._load_timeseries(f"{tech}_production.csv", column_dim="Z"),
            bid_price=self.cfg.simulations.supply_curve[tech].bid_price,
        )

    def _load_timeseries(self, filename: str, column_dim: str) -> xr.DataArray:
        """Hourly CSV -> DataArray with dims (T, column_dim)."""
        df = pd.read_csv(self.data_path / filename, index_col=0)
        df.index.name = "T"
        df.columns.name = column_dim
        if self.cfg.simulations.run.mode == "rolling_horizon":
            df = self._slice_week(df)
        utils.validate_df_positive_numeric(df, filename)

        # Technologies without their own hourly file (e.g. thermal) take their
        # hours from here, so every hourly file must cover the same hours.
        if self.times is None:
            self.times = df.index
        elif not df.index.equals(self.times):
            raise ValueError(f"{filename} does not cover the same hours as the other hourly files")
        return xr.DataArray(df)

    def _slice_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows of `week` (1-based); the last week also takes the leftover hours (8760 = 52 x 168 + 24)."""
        n_weeks = len(df) // HOURS_PER_WEEK
        if self.week is None or not 1 <= self.week <= n_weeks:
            raise ValueError(f"week must be in 1..{n_weeks} in rolling_horizon mode, got {self.week}")
        start = (self.week - 1) * HOURS_PER_WEEK
        end = len(df) if self.week == n_weeks else start + HOURS_PER_WEEK
        return df.iloc[start:end]
