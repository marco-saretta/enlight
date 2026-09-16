from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

import enlight.utils as utils

log = utils.get_logger(__name__)


@dataclass
class DataPreprocessor:
    """
    Raw data (data/) -> per-scenario, model-ready CSVs (simulations/<label>/data/).

    One method per supply_curve/demand_curve entry in the config, each reading
    its own raw files, filtering to the active bidding zones and prediction
    year, and writing a single clean (hour x zone) CSV. DataLoader (the next
    pipeline stage) only ever reads from simulations/<label>/data/, never from
    data/ directly.

    Every technology/demand category has a method below so the class reads as
    a full map of the pipeline, but only wind_onshore, wind_offshore, solar_pv,
    and inflexible demand are implemented so far — everything else is a
    documented no-op until its data is ready.
    """

    config: DictConfig

    def __post_init__(self) -> None:
        self.sim_cfg = self.config.simulations
        self.label = self.sim_cfg.label
        self.bidding_zones = list(self.sim_cfg.bidding_zones)
        self.prediction_year = self.sim_cfg.run.prediction_year

        self.data_path = Path(self.config.paths.data)
        self.output_path = Path(self.config.paths.processed) / self.label / "data"
        self.output_path.mkdir(parents=True, exist_ok=True)

        log.info("-------------- DATA PREPROCESSOR: %s --------------", self.label)

        # Supply curve — variable renewables
        self._process_wind_onshore()
        self._process_wind_offshore()
        self._process_solar_pv()
        self._process_hydro_ror()

        # Supply curve — unit-based dispatchable plants
        self._process_hydro_res()
        self._process_hydro_ps()
        self._process_thermal()
        self._process_bess()

        # Demand curve
        self._process_demand_inflexible()
        self._process_demand_flexible()
        self._process_ptx()
        self._process_district_heating()

        # Transmission
        self._process_lines()

    # -------------------------------------------------------------------
    # Supply curve — variable renewables (bid_price, capacity_file, weather_data)
    # -------------------------------------------------------------------
    def _process_wind_onshore(self) -> None:
        """weather_data profile x capacity_file -> wind_onshore_production.csv [MW, T x Z]."""
        self._process_vre_source("wind_onshore")

    def _process_wind_offshore(self) -> None:
        """weather_data profile x capacity_file -> wind_offshore_production.csv [MW, T x Z]."""
        self._process_vre_source("wind_offshore")

    def _process_solar_pv(self) -> None:
        """weather_data profile x capacity_file -> solar_pv_production.csv [MW, T x Z]."""
        self._process_vre_source("solar_pv")

    def _process_hydro_ror(self) -> None:
        """weather_data profile x capacity_file -> hydro_ror_production.csv [MW, T x Z]."""
        self._process_vre_source("hydro_ror")

    # -------------------------------------------------------------------
    # Supply curve — unit-based dispatchable plants
    # -------------------------------------------------------------------
    def _process_hydro_res(self) -> None:
        """TODO: units_file + energy_weather_year -> hydro_reservoir_units.csv + hydro_reservoir_energy.csv."""
        pass

    def _process_hydro_ps(self) -> None:
        """TODO: units_file -> hydro_pumped_storage_units.csv."""
        pass

    def _process_thermal(self) -> None:
        """TODO: units_file + marginal_cost -> conventional_thermal_units.csv."""
        pass

    def _process_bess(self) -> None:
        """TODO: units_file -> bess_units.csv."""
        pass

    # -------------------------------------------------------------------
    # Demand curve
    # -------------------------------------------------------------------
    def _process_demand_inflexible(self) -> None:
        """profile_year x amount_file -> demand_inflexible_<category>.csv [MW, T x Z], per _inflex category."""
        # Maps a demand_curve category to the data/ folder holding its raw files.
        # Only categories with real data on disk are processed today.
        category_to_folder = {
            "classical_inflex": "demand_inflexible_classic",
            "ev_inflex": "demand_inflexible_ev",
            # "industrial_inflex": "demand_inflexible_industrial",  # no data yet
            # "household_inflex":  "demand_inflexible_household",   # no data yet
            # "public_inflex":     "demand_inflexible_public",      # no data yet
        }
        for category, folder in category_to_folder.items():
            self._process_inflexible_demand_category(category, folder)

    def _process_demand_flexible(self) -> None:
        """TODO: amount_file + capacity_file -> demand_flexible_<category>.csv, per _flex category."""
        pass

    def _process_ptx(self) -> None:
        """TODO: units_file -> ptx_units.csv."""
        pass

    def _process_district_heating(self) -> None:
        """TODO: units_file -> district_heating_units.csv."""
        pass

    # -------------------------------------------------------------------
    # Transmission
    # -------------------------------------------------------------------
    def _process_lines(self) -> None:
        """TODO: capacity_file -> lines_a_b.csv + lines_b_a.csv, filtered to active zones."""
        pass

    # -------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------
    def _process_vre_source(self, tech: str) -> None:
        """
        Build hourly production [MW] for one variable-renewable technology:
        per-unit weather profile (0-1) x installed capacity [MW], filtered to
        the active bidding zones. Writes <tech>_production.csv.
        """
        tech_cfg = self.sim_cfg.supply_curve[tech]
        source = tech_cfg.weather_data.source
        year = tech_cfg.weather_data.year

        profile_path = (
            self.data_path / tech / "weather_data" / source / f"{tech}_{source}_wy_{year}.csv"
        )
        profile = pd.read_csv(profile_path, index_col=0).drop(columns="Week", errors="ignore")
        profile = profile[self.bidding_zones]

        capacity_path = self.data_path / tech / "capacity_projections" / f"{tech_cfg.capacity_file}.csv"
        capacity = pd.read_csv(capacity_path, index_col=0)
        capacity_row = capacity.loc[self.prediction_year, self.bidding_zones]

        production = profile.mul(capacity_row, axis="columns")
        utils.validate_df_positive_numeric(production, f"{tech}_production")

        utils.save_data(production, f"{tech}_production.csv", output_dir=self.output_path)
        log.info("  %s: %s", tech, production.shape)

    def _process_inflexible_demand_category(self, category: str, folder: str) -> None:
        """
        Build hourly inflexible demand [MW] for one demand category: per-unit
        demand profile (0-1) x annual energy projection [MWh], filtered to the
        active bidding zones. Writes demand_inflexible_<category>.csv.
        """
        cat_cfg = self.sim_cfg.demand_curve[category]
        category_path = self.data_path / folder

        profile_files = list((category_path / "profile_years").glob(f"*_py_{cat_cfg.profile_year}.csv"))
        if not profile_files:
            raise FileNotFoundError(
                f"No profile file for '{category}' matching *_py_{cat_cfg.profile_year}.csv "
                f"in {category_path / 'profile_years'}"
            )
        profile = pd.read_csv(profile_files[0], index_col=0).drop(columns="Week", errors="ignore")
        profile = profile[self.bidding_zones]

        projection_path = category_path / "demand_projection" / f"{cat_cfg.amount_file}.csv"
        projection = pd.read_csv(projection_path, index_col=0)
        projection_row = projection.loc[self.prediction_year, self.bidding_zones]

        demand = profile.mul(projection_row, axis="columns")
        utils.validate_df_positive_numeric(demand, f"demand_{category}")

        utils.save_data(demand, f"demand_{category}.csv", output_dir=self.output_path)
        log.info("  %s: %s", category, demand.shape)
