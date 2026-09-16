from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

import enlight.utils as utils

log = utils.get_logger(__name__)


@dataclass
class DataLoader:
    """
    Load energy system input data (renewables, demand, storage, lines, etc.)
    for a given scenario and optionally a specific week.
    """
    config: DictConfig
    week: int | None = None

    def __post_init__(self) -> None:
        """Post-init: prepare paths, load files, and build mappings."""
        self.sim_cfg = self.config.simulations
        self.label = self.sim_cfg.label
        self.mode = self.sim_cfg.run.mode
        self.solver_name = self.sim_cfg.run.solver
        self.prediction_year = self.sim_cfg.run.prediction_year
        self.bidding_zones = list(self.sim_cfg.bidding_zones)

        log.info("-------------- DATA LOADER: %s --------------", self.label)

        self._init_data_paths()
        self._init_time_index()

        # Renewables & demand
        self.load_generation_data()
        self.load_inflexible_demand_data()
        self.load_flexible_demand_data()

        # Time mapping
        self.map_hours_to_weeks()

        # Transmission lines
        self.load_lines_data()
        self.map_transmission_lines()

        # Hydro reservoir
        self.load_hydro_reservoir_data()
        self.load_hydro_res_units_marginal_cost()
        self.map_hydro_res_units_to_zones()

        # Pumped hydro
        self.load_hydro_pumped_data()
        self.load_hydro_ps_units_marginal_cost()
        self.map_hydro_ps_units_to_zones()

        # Conventional thermal
        self.load_conventional_units_data()
        self.load_conventional_units_marginal_cost()
        self.map_conventional_units_to_zones()

        # BESS
        self.load_bess_units()
        self.load_bess_units_marginal_cost()
        # self.map_bess_units_to_zones()

        # PtX
        self.load_ptx_data()
        self.load_ptx_bid_prices()
        self.map_ptx_units_to_zones()

        # DH
        self.load_dh_data()
        self.load_dh_bid_prices()
        self.map_dh_units_to_zones()

    # Path and time utilities
    def _init_data_paths(self) -> None:
        """Initialize scenario base and data directories."""
        self.base_path = Path(self.config.paths.processed) / self.label
        self.data_path = self.base_path / "data"

    def _init_time_index(self) -> None:
        """Build hourly time index based on scenario run mode."""
        if self.mode == "yearly":
            self.T = 8760
        elif self.mode == "rolling_horizon":
            self.T = 168
        else:
            raise ValueError(f"Unknown run.mode: {self.mode}")

        self.time_index = np.arange(1, self.T + 1)
        self.times = list(self.time_index)

    def _load_csv(
        self,
        filename: str,
        index_col: int | None = 0,
        header: int | list[int] | None = 0,
    ) -> pd.DataFrame:
        """Load CSV file from data path, raise if missing."""
        path = self.data_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.read_csv(path, index_col=index_col, header=header)

    def _filter_by_week(
        self,
        df: pd.DataFrame,
        week: int,
        hours_per_week: int = 168,
        week_col: str = 'Week'
    ) -> pd.DataFrame:
        """
        Slice a continuous hourly dataframe into week-sized blocks.
        The final week absorbs any leftover hours.

        Example for length 8760:
            Weeks 1–51: 168 rows each
            Week 52: 168 + 24 = 192 rows (final remainder)
        """
        # TODO: Delete these two following lines, as the rest of the function
        # already creates the week index and then slices based on that.
        # In the input data of the preprocessing, the week column should be
        # removed as well.
        if week_col in df.columns:
            return df[df[week_col] == self.week].drop(columns=week_col)

        if week < 1:
            raise ValueError("week must be >= 1")

        n = len(df)
        n_full_weeks = n // hours_per_week
        remainder = n % hours_per_week

        # Last week number (includes the remainder)
        last_week = n_full_weeks if remainder == 0 else n_full_weeks

        if week > last_week:
            raise IndexError(
                f"Week {week} is out of range. "
                f"Maximum allowable week is {last_week}."
            )

        # Standard start index
        start = (week - 1) * hours_per_week

        # For the last week: extend to the end of df
        if week == last_week:
            end = n
        else:
            end = start + hours_per_week

        return df.iloc[start:end].copy()

    # -------------------------------------------------------------------------
    # Generation data
    # -------------------------------------------------------------------------
    def load_generation_data(self):
        """Load renewable generation time-series and their bid prices from config."""
        self.wind_onshore_production = self._load_csv('wind_onshore_production.csv')
        self.wind_offshore_production = self._load_csv('wind_offshore_production.csv')
        self.solar_pv_production = self._load_csv('solar_pv_production.csv')
        self.hydro_ror_production = self._load_csv('hydro_ror_production.csv')

        self.wind_onshore_bid_price = self.sim_cfg.supply_curve.wind_onshore.bid_price
        self.wind_offshore_bid_price = self.sim_cfg.supply_curve.wind_offshore.bid_price
        self.solar_pv_bid_price = self.sim_cfg.supply_curve.solar_pv.bid_price
        self.hydro_ror_bid_price = self.sim_cfg.supply_curve.hydro_ror.bid_price

        if self.mode == "rolling_horizon":
            self.wind_onshore_production = self._filter_by_week(
                self.wind_onshore_production, week=self.week
            )
            self.wind_offshore_production = self._filter_by_week(
                self.wind_offshore_production, week=self.week
            )
            self.solar_pv_production = self._filter_by_week(
                self.solar_pv_production, week=self.week
            )
            self.hydro_ror_production = self._filter_by_week(
                self.hydro_ror_production, week=self.week
            )

        # Validate data
        utils.validate_df_positive_numeric(
            self.wind_offshore_production, "wind_offshore_production"
        )
        utils.validate_df_positive_numeric(
            self.solar_pv_production, "solar_pv_production"
        )
        utils.validate_df_positive_numeric(
            self.hydro_ror_production, "hydro_ror_production"
        )

    def load_inflexible_demand_data(self):
        """
        Load inflexible demand data and validate.

        Only classical_inflex and ev_inflex are loaded here —
        DataPreprocessor doesn't produce industrial_inflex/household_inflex/
        public_inflex yet, even though they're defined in demand_curve.
        """
        self.demand_classical_inflex = self._load_csv('demand_classical_inflex.csv')
        self.demand_ev_inflex = self._load_csv('demand_ev_inflex.csv')

        self.voll_classical_inflex = self.sim_cfg.demand_curve.classical_inflex.voll
        self.voll_ev_inflex = self.sim_cfg.demand_curve.ev_inflex.voll

        if self.mode == "rolling_horizon":
            self.demand_classical_inflex = self._filter_by_week(self.demand_classical_inflex, week=self.week)
            self.demand_ev_inflex = self._filter_by_week(self.demand_ev_inflex, week=self.week)

        utils.validate_df_positive_numeric(self.demand_classical_inflex, "demand_classical_inflex")
        utils.validate_df_positive_numeric(self.demand_ev_inflex, "demand_ev_inflex")

    def load_flexible_demand_data(self):
        """
        Load flexible demand data.
        """
        # TODO: not yet implemented — awaits DataPreprocessor._process_demand_flexible,
        # which is still a stub. The old subdir-based file layout
        # (demand_flexible_classic/amount/*.csv) predates the supply_curve/
        # demand_curve config and no longer applies; once real files exist,
        # this should read demand_curve.classical_flex/etc. the same way
        # load_inflexible_demand_data does.
        self.flexible_demands_dfs = {}

    def map_hours_to_weeks(self):
        '''
        Make a one-hot-encoded dataframe that maps an hour to its week.
        This is important when enforcing the weekly amount constraint
        for the flexible demands as well as for the weekly energy
        availability of hydro reservoir energy.
        '''
        self.W = int(np.floor(self.T / 168))  # 52 weeks for a whole year
        num_extra_hours = self.T % (self.W * 168)  # 24 for a whole year

        weeks1 = np.repeat(np.arange(1,self.W+1), 168, axis=0)  # shape: (8736,)
        weeks2 = num_extra_hours * [int(weeks1[-1])]  # shape: (24,)
        weeks3 = np.concatenate((weeks1, weeks2))  # shape: (8760,)
        self.T_W_df = pd.get_dummies(weeks3).astype(int)  # shape: (8760, 52)

        # Convert to xarray DataArray for easier indexing in the model
        self.T_W_xr = xr.DataArray(
            self.T_W_df.values,
            coords={
                "T": self.times,   # match with e.g. hydro_res_units_offer
                "W": np.arange(1, self.W+1)
            },
            dims=["T", "W"]
        )

    def load_lines_data(self):
        """Load transmission line capacity or flow data for both directions."""
        # the index_col is set to the transmission line number
        self.lines_a_b_df = self._load_csv("lines_a_b.csv", index_col=0, header=[0,1])  # type: ignore
        self.lines_b_a_df = self._load_csv("lines_b_a.csv", index_col=0, header=[0,1])  # type: ignore

        # The columns are now MultiIndex with (from_zone, to_zone)
        # Extract from_zone and to_zone from the MultiIndex columns
        from_zones = [col[0] for col in self.lines_a_b_df.columns]
        to_zones = [col[1] for col in self.lines_a_b_df.columns]

        # Create line_labels as 'FROM-TO' strings
        self.line_labels = [f"{from_z}-{to_z}" for from_z, to_z in zip(from_zones, to_zones)]

        # Create lines as tuples for the incidence matrix
        self.lines = list(zip(from_zones, to_zones))

        # The capacity arrays are already in the right shape: (T, L)
        # where T = 8760 hours and L = number of lines
        self.lines_a_to_b_cap = self.lines_a_b_df.to_numpy()  # shape: (T, L)
        self.lines_b_to_a_cap = self.lines_b_a_df.to_numpy()  # shape: (T, L)


    def map_transmission_lines(self):
        """Map transmission lines and create incidence matrix."""

        # Extract from_zone and to_zone from the MultiIndex columns
        # The columns are tuples like (from_zone, to_zone)
        # self.lines and self.line_labels were already created in load_lines_data()

        # Build L_Z_df: incidence matrix for power balance
        # L_Z_df[l, z] = +1 if zone z is fromZone of line l
        #              = -1 if zone z is toZone of line l
        #              = 0 otherwise
        # Shape: (L, Z)
        self.L_Z_df = pd.DataFrame(0, index=self.lines, columns=self.bidding_zones)

        for from_zone, to_zone in self.lines:
            if from_zone in self.bidding_zones:
                self.L_Z_df.at[(from_zone, to_zone), from_zone] = 1
            if to_zone in self.bidding_zones:
                self.L_Z_df.at[(from_zone, to_zone), to_zone] = -1

        # Convert to xarray DataArray for easier indexing in the model
        self.L_Z_xr = xr.DataArray(
            self.L_Z_df.values,
            coords={
                "L": self.line_labels,   # Use the string labels like 'AT-CH'
                "Z": self.bidding_zones
            },
            dims=["L", "Z"]
        )

    def load_hydro_reservoir_data(self):
        """Load unit-specific data for hydro reservoir generators."""
        hydro_res_cfg = self.sim_cfg.supply_curve.hydro_res

        # Reservoir hydro units
        self.hydro_res_units = self._load_csv('hydro_reservoir_units.csv')
        self.hydro_res_units_id = list(self.hydro_res_units.index)  # Shape: (G_hydro_res,)

        if hydro_res_cfg.plant_aggregation:

            # Aggregate hydro reservoir units by size (and zone)
            self.agg_hres = utils.agg_by_zone_tech(
                df=self.hydro_res_units,
                tech="Technology"
                )

            # Set a new index with combined size and zone for the aggregated hydro reservoirs
            self.agg_hres = utils.set_agg_idx(self.agg_hres)
            self.hydro_res_units_id = list(self.agg_hres.index)

            # We repeat the capacities for each SIZE (by zone as well)
            self.hydro_res_units_el_cap = np.outer(
                np.ones(self.T),
                self.agg_hres.capacity_el.to_numpy()
            )
        else:
            # Fill the capacities for each hydro unit for all time steps
            self.hydro_res_units_el_cap = np.outer(np.ones(self.T), self.hydro_res_units.capacity_el.to_numpy())


        # Reservoir hydro energy
        hydro_res_energy = self._load_csv('hydro_reservoir_energy.csv')

        if self.mode == "rolling_horizon":
            self.hydro_res_energy = hydro_res_energy.loc[self.week]
        else:
            self.hydro_res_energy = hydro_res_energy

    def load_hydro_res_units_marginal_cost(self):
        """Broadcast each hydro reservoir unit's production cost across all hours."""
        if self.sim_cfg.supply_curve.hydro_res.plant_aggregation:
        # Convert the production cost pandas Series to a DataFrame with time index
        # self.hydro_res_units_marginal_cost_series = self.hydro_res_units.prodcost
        # Use the capacity-weighted average production cost per fuel type (and zone)
            self.hydro_res_units_marginal_cost_series = self.agg_hres.prodcost_weighted

        else:
            self.hydro_res_units_marginal_cost_series = self.hydro_res_units.prodcost

        self.hydro_res_units_marginal_cost_series.index.name = "G_hydro_res"

        self.hydro_res_units_marginal_cost_df = pd.DataFrame(
            data=np.broadcast_to(self.hydro_res_units_marginal_cost_series.to_numpy(),
                                (len(self.times),
                                len(self.hydro_res_units_marginal_cost_series))),
            index=self.times,
            columns=self.hydro_res_units_marginal_cost_series.index
            )

    def map_hydro_res_units_to_zones(self):
        """
        Build binary hydro reservoir-to-zone assignment matrix (G_hydro_res x Z).
        G_hydro_res_Z[g_hydro_res, z] = 1 if generator g_hydro_res belongs to zone z, else 0.
        """
        # Create dummy variables (one-hot encode) from generator zone assignment
        if self.sim_cfg.supply_curve.hydro_res.plant_aggregation:
            self.G_hydro_res_Z_df = pd.get_dummies(self.agg_hres['zone_el']).astype(int)
        else:
            self.G_hydro_res_Z_df = pd.get_dummies(self.hydro_res_units['zone_el']).astype(int)

        # Ensure all zones are represented as columns, even if some have no hydro reservoirs
        self.G_hydro_res_Z_df = self.G_hydro_res_Z_df.reindex(columns=self.bidding_zones, fill_value=0)

        # Wrap into xarray with matching dimensions
        self.G_hydro_res_Z_xr = xr.DataArray(
            self.G_hydro_res_Z_df.values,
            coords={
                "G_hydro_res": self.hydro_res_units_id,   # Generator labels (must match dims in hydro_res_bid)
                "Z": self.bidding_zones     # Zone labels
            },
            dims=["G_hydro_res", "Z"]            # Dimension names for alignment in dot product
        )

    def load_hydro_pumped_data(self):
        """Load unit-specific data for pumped hydro storage units."""
        hydro_ps_cfg = self.sim_cfg.supply_curve.hydro_ps
        self.hydro_ps_initial_SOC = hydro_ps_cfg.initial_soc
        self.hydro_ps_charging_efficiency = float(np.sqrt(hydro_ps_cfg.roundtrip_efficiency))
        self.hydro_ps_discharging_efficiency = float(np.sqrt(hydro_ps_cfg.roundtrip_efficiency))

        self.hydro_ps_units = self._load_csv('hydro_pumped_units.csv')

        if hydro_ps_cfg.plant_aggregation:
            # Aggregate pumped hydro storage units by zone (only one type of tech/size)
            self.agg_phs = utils.agg_storage_by_zone(df=self.hydro_ps_units)

            # Fill in 0s for any zones that do not has PHS
            self.agg_phs = self.agg_phs.reindex(self.bidding_zones, fill_value=0)

            # We need to repeat the charge/discharge and storage capacities for each hydro unit for all time steps:
            self.hydro_ps_units_el_cap = np.outer(
                np.ones(self.T),
                self.agg_phs.capacity_el.to_numpy()
            )
            self.hydro_ps_units_storage_cap = np.outer(
                np.ones(self.T),
                self.agg_phs.capacity_stor.to_numpy()
            )

            # Create an xarray DataArray with the same dimensions and coordinates
            # as the model variables will have to avoid "UserWarning". Set the values
            # in all other time steps than T=0 to zero.
            hydro_ps_initial_SOC_x_storage_cap = self.hydro_ps_initial_SOC * self.hydro_ps_units_storage_cap
            hydro_ps_initial_SOC_x_storage_cap[1:, :] = 0  # Initial SOC only applies in the first hour (T=0)
            self.hydro_ps_initial_SOC_x_storage_cap_xr = xr.DataArray(
                data=hydro_ps_initial_SOC_x_storage_cap,
                dims=["T", "Z"],
                coords=(self.times, self.bidding_zones)
            )

        else:
            self.hydro_ps_units_id = list(self.hydro_ps_units.index)  # Shape: (G_hydro_ps,)
            # We need to repeat the charge/discharge and storage capacities for each hydro unit for all time steps:
            self.hydro_ps_units_el_cap = np.outer(np.ones(self.T), self.hydro_ps_units.capacity_el.to_numpy())
            self.hydro_ps_units_storage_cap = np.outer(np.ones(self.T), self.hydro_ps_units.Storage_Capacity.to_numpy())

            # Create an xarray DataArray with the same dimensions and coordinates
            #   as the model variables will have to avoid "UserWarning". Set the values
            #   in all other time steps than T=0 to zero.
            hydro_ps_initial_SOC_x_storage_cap = self.hydro_ps_initial_SOC * self.hydro_ps_units_storage_cap
            hydro_ps_initial_SOC_x_storage_cap[1:, :] = 0  # Initial SOC only applies in the first hour (T=0)
            self.hydro_ps_initial_SOC_x_storage_cap_xr = xr.DataArray(
                data=hydro_ps_initial_SOC_x_storage_cap,
                dims=["T", "G_hydro_ps"],
                coords=(self.times, self.hydro_ps_units_id))

    def load_hydro_ps_units_marginal_cost(self):
        """
        Broadcast pumped-hydro bid/offer prices across all hours.

        Unlike reservoir hydro and conventional units, pumped hydro requires
        both a bid (pumping) and an offer (generating) price, so the result
        is a dict of two DataFrames, not one.
        """
        if self.sim_cfg.supply_curve.hydro_ps.plant_aggregation:
            bid_and_offer_new_col_names = ["Bid_price", "Offer_price"]
            self.hydro_ps_units_marginal_cost_dfs = {}

            for k in bid_and_offer_new_col_names:
                self.hydro_ps_units_marginal_cost_dfs[k] = pd.DataFrame(
                    data=np.broadcast_to(
                        self.agg_phs[k.lower()+"_weighted"].to_numpy(),
                        (len(self.times),
                        len(self.bidding_zones))
                    ),
                    index=self.times,
                    columns=self.bidding_zones
                )
        else:

            bid_and_offer_col_names = ["Pumped_cons", "Pumped_prod"]
            bid_and_offer_new_col_names = ["Bid_price", "Offer_price"]
            self.hydro_ps_units_marginal_cost_seriess = {}
            self.hydro_ps_units_marginal_cost_dfs = {}

            # Create series and dataframes for the bid and offer prices
                # e.g. hydro_ps_units_marginal_cost_dfs["Bid_price"] = Pumped_cons from the csv
            for k1, k2 in zip(bid_and_offer_col_names, bid_and_offer_new_col_names):
                self.hydro_ps_units_marginal_cost_seriess[k2] = self.hydro_ps_units.loc[:,k1]
                self.hydro_ps_units_marginal_cost_seriess[k2].index.name = "G_hydro_ps"
                self.hydro_ps_units_marginal_cost_dfs[k2] = pd.DataFrame(
                    data=np.broadcast_to(self.hydro_ps_units_marginal_cost_seriess[k2].to_numpy(),
                                        (len(self.times),
                                        len(self.hydro_ps_units_marginal_cost_seriess[k2]))),
                        index=self.times,
                        columns=self.hydro_ps_units_id)

    def map_hydro_ps_units_to_zones(self):
        """
        Build binary hydro_ps-to-zone assignment matrix (G x Z).
        G_hydro_ps_Z[g_hydro_ps, z] = 1 if hydro ps unit G_hydro_ps belongs to zone z, else 0.
        """
        if self.sim_cfg.supply_curve.hydro_ps.plant_aggregation:
            pass
        else:
            # Create dummy variables (one-hot encode) from generator zone assignment
            self.G_hydro_ps_Z_df = pd.get_dummies(self.hydro_ps_units['zone_el']).astype(int)

            # Ensure all zones are represented as columns, even if some have no conventional_units
            self.G_hydro_ps_Z_df = self.G_hydro_ps_Z_df.reindex(columns=self.bidding_zones, fill_value=0)

            # Wrap into xarray with matching dimensions
            self.G_hydro_ps_Z_xr = xr.DataArray(
                self.G_hydro_ps_Z_df.values,
                coords={
                    "G_hydro_ps": self.hydro_ps_units_id,   # Generator labels (must match dims in thermal_gen_bid_vol)
                    "Z": self.bidding_zones     # Zone labels
                },
                dims=["G_hydro_ps", "Z"]            # Dimension names for alignment in dot product
            )

    def load_conventional_units_data(self):
        """Load conventional thermal unit capacities, optionally aggregated by zone + fuel type."""
        self.conventional_units_df = self._load_csv('conventional_thermal_units.csv')

        if self.sim_cfg.supply_curve.thermal.plant_aggregation:

            # Aggregate thermal plants by fuel type (and zone)
            self.agg_g = utils.agg_by_zone_tech(self.conventional_units_df)

            # Set a new index with combined zone and fuel type for the aggregated generators
            self.agg_g = utils.set_agg_idx(self.agg_g)
            self.conventional_units_id = list(self.agg_g.index)

            # We repeat the capacities for each FUEL (by zone and fuel) of generator
            self.conventional_units_el_cap = np.outer(np.ones(self.T), self.agg_g.capacity_el.to_numpy())

        else:
            # We need to repeat the capacities for each generator for all time steps:
            self.conventional_units_id = list(self.conventional_units_df.index)       # Shape: (G,)
            self.conventional_units_el_cap = np.outer(np.ones(self.T), self.conventional_units_df.capacity_el.to_numpy())

    def load_conventional_units_marginal_cost(self):
        """Broadcast each conventional unit's production cost across all hours."""
        if self.sim_cfg.supply_curve.thermal.plant_aggregation:
            self.conventional_units_marginal_cost_series = self.agg_g.prodcost_weighted
        else:
            self.conventional_units_marginal_cost_series = self.conventional_units_df.prodcost

        self.conventional_units_marginal_cost_series.index.name = "G"

        self.conventional_units_marginal_cost_df = pd.DataFrame(
            data=np.broadcast_to(self.conventional_units_marginal_cost_series.to_numpy(),
                            (len(self.times),
                            len(self.conventional_units_marginal_cost_series))),
            index=self.times,
            columns=self.conventional_units_id
            )

    def map_conventional_units_to_zones(self):
        """
        Build binary generator-to-zone assignment matrix (G x Z).
        G_Z[g, z] = 1 if generator g belongs to zone z, else 0.
        """
        # Create dummy variables (one-hot encode) from generator zone assignment
        if self.sim_cfg.supply_curve.thermal.plant_aggregation:
            self.G_Z_df = pd.get_dummies(self.agg_g['zone_el']).astype(int)
        else:
            self.G_Z_df = pd.get_dummies(self.conventional_units_df['zone_el']).astype(int)

        # Ensure all zones are represented as columns, even if some have no conventional_units
        self.G_Z_df = self.G_Z_df.reindex(columns=self.bidding_zones, fill_value=0)

        # Wrap into xarray with matching dimensions
        self.G_Z_xr = xr.DataArray(
            self.G_Z_df.values,
            coords={
                "G": self.conventional_units_id,   # Generator labels (must match dims in thermal_gen_bid_vol)
                "Z": self.bidding_zones     # Zone labels
            },
            dims=["G", "Z"]            # Dimension names for alignment in dot product
        )

    def load_bess_units(self):
        """
        Load battery energy storage system (BESS) unit data.

        BESS has no plant_aggregation switch in the current config (unlike
        hydro_res/hydro_ps/thermal), so this always loads per-unit.
        """
        bess_cfg = self.sim_cfg.supply_curve.bess
        self.bess_initial_SOC = bess_cfg.initial_soc
        self.bess_charging_efficiency = float(np.sqrt(bess_cfg.roundtrip_efficiency))
        self.bess_discharging_efficiency = float(np.sqrt(bess_cfg.roundtrip_efficiency))

        self.bess_units_df = self._load_csv('bess_units.csv')

        self.bess_units_id = list(self.bess_units_df.index)  # Shape: (G_bess,)

        # We need to repeat the charge/discharge and storage capacities for each bess unit for all time steps:
        self.bess_units_el_cap = np.outer(np.ones(self.T), self.bess_units_df.capacity_el.to_numpy())
        self.bess_units_storage_cap = np.outer(np.ones(self.T), self.bess_units_df.storage_capacity.to_numpy())

        # Create an xarray DataArray of the initial SOC in MWh to avoid "UserWarning".
        #   This is identical to the hydro_ps initial SOC handling above.
        bess_initial_SOC_x_storage_cap = self.bess_initial_SOC * self.bess_units_storage_cap
        bess_initial_SOC_x_storage_cap[1:, :] = 0  # Initial SOC only applies in the first hour (T=0)
        self.bess_initial_SOC_x_storage_cap_xr = xr.DataArray(
            data=bess_initial_SOC_x_storage_cap,
            dims=["T", "G_bess"],
            coords=(self.times, self.bess_units_id))

    def load_bess_units_marginal_cost(self):
        """
        Broadcast BESS charge/discharge prices across all hours.

        Like pumped hydro, BESS requires both a bid (charge) and an offer
        (discharge) price, so the result is a dict of two DataFrames.
        """
        bid_and_offer_col_names = ["charge_bid_price", "discharge_bid_price"]
        bid_and_offer_new_col_names = ["Bid_price", "Offer_price"]
        self.bess_units_marginal_cost_seriess = {}
        self.bess_units_marginal_cost_dfs = {}

        # Create series and dataframes for the bid and offer prices
            # e.g. bess_units_marginal_cost_dfs["Bid_price"] = charge_bid_price from the csv
        for k1, k2 in zip(bid_and_offer_col_names, bid_and_offer_new_col_names):
            self.bess_units_marginal_cost_seriess[k2] = self.bess_units_df.loc[:,k1]
            self.bess_units_marginal_cost_seriess[k2].index.name = "G_bess"
            self.bess_units_marginal_cost_dfs[k2] = pd.DataFrame(
                data=np.broadcast_to(self.bess_units_marginal_cost_seriess[k2].to_numpy(),
                                    (len(self.times),
                                    len(self.bess_units_marginal_cost_seriess[k2]))),
                    index=self.times,
                    columns=self.bess_units_id)

    def map_bess_units_to_zones(self):
        """
        Build binary BESS-to-zone assignment matrix (G_bess x Z).
        G_bess_Z[g_bess, z] = 1 if BESS unit g_bess belongs to zone z, else 0.
        """
        # Create dummy variables (one-hot encode) from generator zone assignment
        self.G_bess_Z_df = pd.get_dummies(self.bess_units_df['zone_el']).astype(int)

        # Ensure all zones are represented as columns, even if some have no conventional_units
        self.G_bess_Z_df = self.G_bess_Z_df.reindex(columns=self.bidding_zones, fill_value=0)

        # Wrap into xarray with matching dimensions
        self.G_bess_Z_xr = xr.DataArray(
            self.G_bess_Z_df.values,
            coords={
                "G_bess": self.bess_units_id,   # Generator labels (must match dims in thermal_gen_bid_vol)
                "Z": self.bidding_zones     # Zone labels
            },
            dims=["G_bess", "Z"]            # Dimension names for alignment in dot product
        )

    def load_ptx_data(self):
        """
        Load PtX (power-to-X) unit capacities.

        PtX has no plant_aggregation switch in the current config, so this
        always loads per-unit.
        """
        self.ptx_units_df = self._load_csv('ptx_units.csv')

        self.ptx_units_id = list(self.ptx_units_df.index)  # shape (L_PtX)
        # Repeat capacities for each time steps
        self.ptx_units_el_cap = np.outer(np.ones(self.T),
                                         self.ptx_units_df["Electric capacity"].to_numpy())

    def load_ptx_bid_prices(self):
        """
        Makes a (T, L_PtX) DataFrame of the PtX unit bid prices.
        These bid prices are simply the LCoX of each plant.
        """
        self.ptx_units_bid_prices_series = self.ptx_units_df["Demand price"]
        self.ptx_units_bid_prices_series.index.name = "L_PtX"

        self.ptx_units_bid_prices_df = pd.DataFrame(
            data=np.broadcast_to(self.ptx_units_bid_prices_series.to_numpy(),
                                 (len(self.times),
                                 len(self.ptx_units_bid_prices_series))),
            index=self.times,
            columns=self.ptx_units_id
        )

    def map_ptx_units_to_zones(self):
        """
        Build binary PtX unit-to-zone assignment matrix (L_PtX x Z).
        L_PtX_Z[l_PtX, z] = 1 if PtX unit l_PtX belongs to zone z, else 0.
        """
        # Create dummy variables (one-hot encode) from PtX unit zone assignment
        self.L_PtX_Z_df = pd.get_dummies(self.ptx_units_df['zone_el']).astype(int)

        # Ensure all zones are represented as columns, even if some have no PtX units
        self.L_PtX_Z_df = self.L_PtX_Z_df.reindex(columns=self.bidding_zones, fill_value=0)

        # Wrap into xarray with matching dimensions
        self.L_PtX_Z_xr = xr.DataArray(
            self.L_PtX_Z_df.values,
            coords={
                "L_PtX": self.ptx_units_id,   # PtX unit labels
                "Z": self.bidding_zones     # Zone labels
            },
            dims=["L_PtX", "Z"]            # Dimension names for alignment in dot product
        )

    def load_dh_data(self):
        """
        Load district-heating unit capacities.

        District heating has no plant_aggregation switch in the current
        config, so this always loads per-unit.
        """
        self.dh_units_df = self._load_csv('dh_units.csv')

        self.dh_units_id = list(self.dh_units_df.index)  # shape (L_DH)
        # Repeat capacities for each time steps
        self.dh_units_el_cap = np.outer(np.ones(self.T),
                                        self.dh_units_df["Thermal capacity"].to_numpy())

    def load_dh_bid_prices(self):
        """Broadcast each district-heating unit's demand price across all hours."""
        self.dh_units_bid_prices_series = self.dh_units_df.demand_price
        self.dh_units_bid_prices_series.index.name = "L_DH"

        self.dh_units_bid_prices_df = pd.DataFrame(
            data=np.broadcast_to(self.dh_units_bid_prices_series.to_numpy(),
                                 (len(self.times),
                                 len(self.dh_units_bid_prices_series))),
            index=self.times,
            columns=self.dh_units_id
        )

    def map_dh_units_to_zones(self):
        """
        Build binary DH unit-to-zone assignment matrix (L_DH x Z).
        L_DH_Z[l_DH, z] = 1 if DH unit l_DH belongs to zone z, else 0.
        """
        # Create dummy variables (one-hot encode) from DH unit zone assignment
        self.L_DH_Z_df = pd.get_dummies(self.dh_units_df['zone_el']).astype(int)

        # Ensure all zones are represented as columns, even if some have no DH units
        self.L_DH_Z_df = self.L_DH_Z_df.reindex(columns=self.bidding_zones, fill_value=0)

        # Wrap into xarray with matching dimensions
        self.L_DH_Z_xr = xr.DataArray(
            self.L_DH_Z_df.values,
            coords={
                "L_DH": self.dh_units_id,   # DH unit labels
                "Z": self.bidding_zones     # Zone labels
            },
            dims=["L_DH", "Z"]            # Dimension names for alignment in dot product
        )
