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
    a full map of the pipeline; the renewables, thermal, inflexible demand and
    lines are implemented so far — everything else is a documented no-op
    until its data is ready.
    """

    cfg: DictConfig

    def __post_init__(self) -> None:
        label = self.cfg.simulations.label
        self.bidding_zones = list(self.cfg.simulations.bidding_zones)
        self.prediction_year = self.cfg.simulations.run.prediction_year

        self.data_path = Path(self.cfg.paths.data)
        self.output_path = Path(self.cfg.paths.processed) / label / "data"
        self.output_path.mkdir(parents=True, exist_ok=True)

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
        """
        units_file + technology data + emission factors + fuel/CO2 price projections
        -> thermal_units.csv: one row per unit with zone, capacity [MW], every
        marginal cost input and component, and marginal_cost [EUR/MWh-el],
        fixed for the whole prediction year.
        """
        thermal_cfg = self.cfg.simulations.supply_curve.thermal

        units = pd.read_csv(self.data_path / "thermal_plants" / "units" / f"{thermal_cfg.units_file}.csv", index_col=0)
        units = units[units["zone"].isin(self.bidding_zones) & (units["electric_capacity"] > 0)]

        thermal = units[["zone", "technology", "fuel", "fuel_type", "electric_capacity"]]
        thermal = thermal.rename(columns={"electric_capacity": "capacity"})
        thermal = thermal.join(self._thermal_marginal_costs(units, thermal_cfg.marginal_cost))

        missing = thermal["marginal_cost"].isna()
        if missing.any():
            dropped = thermal[missing].groupby(["technology", "fuel_type"])["capacity"].agg(["size", "sum"])
            log.warning(
                "thermal: dropping %d units (%.1f GW) with no technology data, emission factor or fuel price: %s",
                missing.sum(), thermal.loc[missing, "capacity"].sum() / 1e3,
                "; ".join(f"{t} / {f} ({int(n)} units, {c / 1e3:.1f} GW)" for (t, f), (n, c) in dropped.iterrows()),
            )
            thermal = thermal[~missing]

        if thermal_cfg.plant_aggregation:
            thermal = self._aggregate_thermal(thermal)

        thermal.index.name = "unit"
        utils.save_data(thermal, "thermal_units.csv", output_dir=self.output_path)
        log.info("thermal: %d units, %.1f GW", len(thermal), thermal["capacity"].sum() / 1e3)

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
        """
        NTC files -> lines.csv (one row per line: from_zone, to_zone) and
        lines_capacity_from_to.csv / lines_capacity_to_from.csv [MW, T x L],
        keeping only lines whose two ends are both active bidding zones.
        """
        lines_path = self.data_path / "lines" / self.cfg.simulations.lines.capacity_file

        # Both raw files use a two-row header; columns are in the same order,
        # and lines_b_a holds the capacity in the reverse direction.
        from_to = pd.read_csv(lines_path / "lines_a_b.csv", header=[0, 1], index_col=0)
        to_from = pd.read_csv(lines_path / "lines_b_a.csv", header=[0, 1], index_col=0)

        pairs = list(from_to.columns)
        keep = [a in self.bidding_zones and b in self.bidding_zones for a, b in pairs]
        labels = [f"{a}-{b}" for (a, b), k in zip(pairs, keep) if k]

        lines = pd.DataFrame(
            [(a, b) for (a, b), k in zip(pairs, keep) if k],
            index=pd.Index(labels, name="line"),
            columns=["from_zone", "to_zone"],
        )

        capacities = {"from_to": from_to, "to_from": to_from}
        for direction, df in capacities.items():
            capacity = df.loc[:, keep]
            capacity.columns = labels
            capacity.index.name = "Time"
            utils.validate_df_positive_numeric(capacity, f"lines_capacity_{direction}")
            utils.save_data(capacity, f"lines_capacity_{direction}.csv", output_dir=self.output_path)

        utils.save_data(lines, "lines.csv", output_dir=self.output_path)
        log.info("lines: %d between active zones", len(lines))

    # -------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------
    def _process_vre_source(self, tech: str) -> None:
        """
        Build hourly production [MW] for one variable-renewable technology:
        per-unit weather profile (0-1) x installed capacity [MW], filtered to
        the active bidding zones. Writes <tech>_production.csv.
        """
        tech_cfg = self.cfg.simulations.supply_curve[tech]
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
        self._log_energy(tech, "available", production)

    def _process_inflexible_demand_category(self, category: str, folder: str) -> None:
        """
        Build hourly inflexible demand [MW] for one demand category: per-unit
        demand profile (0-1) x annual energy projection [MWh], filtered to the
        active bidding zones. Writes demand_inflexible_<category>.csv.
        """
        cat_cfg = self.cfg.simulations.demand_curve[category]
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
        self._log_energy(category, "demand", demand)

    def _thermal_marginal_costs(self, units: pd.DataFrame, cost_cfg) -> pd.DataFrame:
        """
        Marginal cost per unit for the prediction year, laid out like pypsa-eur's
        cost table: inputs, then the components they produce, then their sum.

            inputs:     efficiency [MWh-el/MWh-th], fuel_price [EUR/MWh-th],
                        co2_intensity [t/MWh-th], co2_capture_rate [-], co2_price [EUR/t]
            components: fuel_cost = fuel_price / efficiency                                    [EUR/MWh-el]
                        co2_cost  = co2_price * co2_intensity * (1 - co2_capture_rate) / efficiency
                        vom_cost  = variable O&M
            total:      marginal_cost = fuel_cost + co2_cost + vom_cost
        """
        if cost_cfg.get("fuel_price_profile"):
            # TODO: scale fuel prices hour by hour with a monthly profile (e.g. TTF
            # monthly averages), like pypsa-eur's conventional.dynamic_fuel_price.
            # marginal_cost then becomes (T, unit) instead of one value per unit.
            raise NotImplementedError("thermal.marginal_cost.fuel_price_profile is not implemented yet")

        tech = self._read_reference(self.data_path / "technology_data" / "technology_data.csv")
        emissions = self._read_reference(self.data_path / "emissions" / "emissions.csv")
        fuel_prices = self._read_price_projection(cost_cfg.fuel_prices)

        costs = pd.DataFrame(index=units.index)
        costs["efficiency"] = units["technology"].map(tech["Electric efficiency CHP"])
        costs["fuel_price"] = units["fuel_type"].map(fuel_prices)
        costs["co2_intensity"] = units["fuel"].map(emissions["co2_emission_pu"]) * 1e-3  # kg -> t
        costs["co2_capture_rate"] = units["technology"].map(tech["CO2 capture rate (amount of emission)"])
        costs["co2_price"] = self._read_price_projection(cost_cfg.co2_quota_prices)["CO2 quota"]

        costs["fuel_cost"] = costs["fuel_price"] / costs["efficiency"]
        costs["co2_cost"] = (
            costs["co2_price"] * costs["co2_intensity"] * (1 - costs["co2_capture_rate"]) / costs["efficiency"]
        )
        costs["vom_cost"] = units["technology"].map(tech["Var. O&M (el)"])

        # skipna=False: a unit missing any input gets NaN, not a partial cost
        costs["marginal_cost"] = costs[["fuel_cost", "co2_cost", "vom_cost"]].sum(axis=1, skipna=False)
        return costs

    @staticmethod
    def _aggregate_thermal(thermal: pd.DataFrame) -> pd.DataFrame:
        """
        One plant per zone and technology, named <zone>_<technology>_plant:
        capacities add up; inputs and costs are capacity-weighted averages.

        Units of one technology share its efficiency, VOM and fuel, so the
        aggregate is exact unless their fuel prices differ (e.g. Central vs
        Decentral gas priced differently in the prediction year).
        """
        numeric = thermal.columns.drop(["zone", "technology", "fuel", "fuel_type", "capacity"])
        weighted = thermal[numeric].mul(thermal["capacity"], axis=0).join(thermal[["zone", "technology", "capacity"]])

        plants = weighted.groupby(["zone", "technology"], as_index=False).sum()
        plants[numeric] = plants[numeric].div(plants["capacity"], axis=0)

        labels = thermal.groupby(["zone", "technology"]).agg(
            fuel=("fuel", "first"),
            fuel_type=("fuel_type", lambda s: " / ".join(sorted(set(s)))),
        )
        plants = plants.join(labels, on=["zone", "technology"])

        slug = plants["technology"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        plants.index = plants["zone"] + "_" + slug + "_plant"
        return plants[thermal.columns]

    def _read_price_projection(self, dataset: str) -> pd.Series:
        """Prices of every fuel (and CO2 quota) for the prediction year from one projection dataset."""
        projections = pd.read_csv(self.data_path / "fuel_price_projections" / dataset / "fuel_price_projections.csv", index_col=0)
        return projections.loc[self.prediction_year]

    def _log_energy(self, name: str, kind: str, hourly: pd.DataFrame) -> None:
        """Log a series' yearly energy, warning when it is zero in every zone."""
        twh = hourly.to_numpy().sum() / 1e6
        if twh == 0:
            log.warning("%s: 0 TWh %s in %d; check its profile and projection for that year", name, kind, self.prediction_year)
        else:
            log.info("%s: %.1f TWh %s", name, twh, kind)

    @staticmethod
    def _read_reference(path: Path) -> pd.DataFrame:
        """Read a reference table whose second header row holds units, keeping only the names."""
        df = pd.read_csv(path, header=[0, 1], index_col=0, na_values=["---", "NA", "n/a"])
        df.columns = df.columns.get_level_values(0)
        return df
