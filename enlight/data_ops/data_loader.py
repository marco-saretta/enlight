from pathlib import Path
from dataclasses import dataclass
from logging import Logger
import pandas as pd
import xarray as xr
import numpy as np
import yaml
import enlight.utils as utils

@dataclass
class DataLoader:
    """
    Loads energy system input data for a given week from structured CSV files
    and a YAML auxiliary scenario metadata file.
    
    Example usage:
    open interactive window in VSCode,
    >>> cd ../../
    run the script data_loader.py in the interactive window,
    >>> data = DataLoader(week=1, input_path='simulations/scenario_1/data')
    """
    week: int
    input_path: Path
    logger: Logger

    def __post_init__(self):
        """
        Post-initialization to load and validate all required datasets.
        """
        # Initialize logger
        self.logger.info(
            f"INITIALIZING DATA LOADER FOR WEEK {self.week} FROM {self.input_path}"
        )
        
        self.input_path = Path(self.input_path).resolve()
        
        # Load YAML metadata (auxiliary scenario data)
        self.load_yaml_aux_data('scenario_1_aux_data.yaml')
        
        # Load CSV datasets
        self.load_generation_data()
        self.load_inflexible_demand_data()
        self.load_lines_data()
        self.load_flexible_demand_data()  # -> uses self.T from load_lines_data()
        self.map_transmission_lines()
        self.load_hydro_reservoir_data()
        self.load_hydro_res_units_marginal_cost()
        self.map_hydro_res_units_to_zones()
        self.load_hydro_pumped_data()
        self.load_hydro_ps_units_marginal_cost()
        self.map_hydro_ps_units_to_zones()
        self.load_conventional_units_data()
        self.load_conventional_units_marginal_cost()
        self.map_conventional_units_to_zones()
        # Optionally uncomment these as needed
        # self.load_heating()
        # self.load_ptx()
        self.load_bess_units()
        self.load_bess_units_marginal_cost()
        self.map_bess_units_to_zones()


    def _load_csv(self, filename: str, index_col=0) -> pd.DataFrame:
        """Helper function to load a CSV file. Raises FileNotFoundError if missing."""
        file_path = self.input_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        return pd.read_csv(file_path, index_col=index_col)


    def _filter_by_week(self, df: pd.DataFrame, week_col: str = 'Week') -> pd.DataFrame:
        """Filter a DataFrame by the specified week and drop the 'Week' column."""
        return df[df[week_col] == self.week].drop(columns=week_col)
    
    def load_yaml_aux_data(self, filename: str):
        """
        Load auxiliary metadata for the scenario from a YAML file.
        Exposes the content as a dictionary: self.yaml_data
        and attaches key values as class attributes (flattened).
        """
        yaml_path = self.input_path / filename
        with open(yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)

        # Flatten and assign attributes for convenience
        self.scenario_name = self.yaml_data.get('scenario_name')
        self.prediction_year = self.yaml_data.get('prediction_year')
        self.bidding_zones = self.yaml_data.get('bidding_zones', [])

        # Nested values
        wind_on = self.yaml_data.get('wind_on', {})
        wind_off = self.yaml_data.get('wind_off', {})
        solar = self.yaml_data.get('solar', {})
        hydro_ror = self.yaml_data.get('hydro_ror', {})
        demand_classic = self.yaml_data.get('demand_inflexible_classic', {})
        demand_ev = self.yaml_data.get('demand_inflexible_ev', {})
        # The flexible demand data is set up to also be run dynamically,
        #    for now it's hardcoded like the other values for simplicity.
        demand_flexible_classic = self.yaml_data.get('demand_flexible_classic', {})
        demand_flexible_ev = self.yaml_data.get('demand_flexible_ev', {})

        self.wind_onshore_bid_price = wind_on.get('bid_prices')
        self.wind_offshore_bid_price = wind_off.get('bid_prices')
        self.solar_pv_bid_price = solar.get('bid_prices')
        self.hydro_ror_bid_price = hydro_ror.get('bid_prices')

        self.wind_on_weather_year = wind_on.get('weather_year')
        self.wind_off_weather_year = wind_off.get('weather_year')
        self.solar_weather_year = solar.get('weather_year')
        self.hydro_ror_weather_year = hydro_ror.get('weather_year')

        self.voll_classic = demand_classic.get('voll')
        self.voll_ev = demand_ev.get('voll')
        self.wtp_classic = demand_flexible_classic.get('wtp')
        self.wtp_ev = demand_flexible_ev.get('wtp')

        # Storage roundtrip efficiencies and initial SOCs
            # BESS
        self.bess_charging_efficiency = float(np.sqrt(self.yaml_data.get('bess_roundtrip')))  # np.float gives errors. It has to just be a Python scalar
        self.bess_discharging_efficiency = float(np.sqrt(self.yaml_data.get('bess_roundtrip')))
        self.bess_initial_SOC = self.yaml_data.get('bess_initial_soc')
            # Pumped hydro storage
        self.hydro_ps_charging_efficiency = float(np.sqrt(self.yaml_data.get('hydro_ps_roundtrip')))  # np.float gives errors. It has to just be a Python scalar
        self.hydro_ps_discharging_efficiency = float(np.sqrt(self.yaml_data.get('hydro_ps_roundtrip')))
        self.hydro_ps_initial_SOC = self.yaml_data.get('hydro_ps_initial_soc')
        

    def load_generation_data(self):
        """Load all renewable generation time-series data filtered by week and validate them."""
        self.wind_onshore_production = self._filter_by_week(self._load_csv('wind_onshore_production.csv'))
        utils.validate_df_positive_numeric(self.wind_onshore_production, "wind_onshore_production")

        self.wind_offshore_production = self._filter_by_week(self._load_csv('wind_offshore_production.csv'))
        utils.validate_df_positive_numeric(self.wind_offshore_production, "wind_offshore_production")

        self.solar_pv_production = self._filter_by_week(self._load_csv('solar_pv_production.csv'))
        utils.validate_df_positive_numeric(self.solar_pv_production, "solar_pv_production")

        self.hydro_ror_production = self._filter_by_week(self._load_csv('hydro_ror_production.csv'))
        utils.validate_df_positive_numeric(self.hydro_ror_production, "hydro_ror_production")

    def load_inflexible_demand_data(self):
        """Load inflexible demand data (classic and EV) filtered by week and validate."""
        self.demand_inflexible_classic = self._filter_by_week(self._load_csv('demand_inflexible_classic.csv'))
        utils.validate_df_positive_numeric(self.demand_inflexible_classic, "demand_inflexible_classic")

        self.times = list(self.demand_inflexible_classic.index)  # this is important: if the time index of the cost dataframes don't match the model variables' then it assumes 0-cost

        self.demand_inflexible_ev = self._filter_by_week(self._load_csv('demand_inflexible_ev.csv'))
        utils.validate_df_positive_numeric(self.demand_inflexible_ev, "demand_inflexible_ev")

    def load_flexible_demand_data(self):
        """Load flexible demand data filtered by week and validate."""
        # subdir_labels = ["demand_flexible_classic", "demand_flexible_industry", "demand_flexible_household", "demand_flexible_public", "demand_flexible_ev"]
        subdir_labels = ["demand_flexible_classic", "demand_flexible_ev"]  # data for the others are not yet available
        parameter_names = ["amount", "capacity"]  # params used to model flexible demand
        
        self.flexible_demands_dfs = {
            flex_load: {param: pd.DataFrame() for param in parameter_names}
            for flex_load in subdir_labels
        }  # Nested dict to hold DataFrames for each flexible load type and the two parameters
        
        for flex_load in subdir_labels:  # e.g. "demand_flexible_classic"
            for param in parameter_names: # "amount" or "capacity"
                flex_dem_df = self._load_csv(f"{flex_load}_{param}.csv")
                
                # Repeat the maximum capacity for all time steps
                if param == "capacity":
                    self.flexible_demands_dfs[flex_load][param] = np.outer(np.ones(self.T),
                                                                           flex_dem_df.loc[self.week].to_numpy())
                else:  # param == "amount" in which case we only need the values for the specific week w/o repetition
                    self.flexible_demands_dfs[flex_load][param] = flex_dem_df.loc[self.week]

        # call in energy_model.py as e.g.:
        #   self.data.flexible_demands_dfs['demand_flexible_classic']['capacity']

    def load_lines_data(self):
        """Load transmission line capacity or flow data for both directions."""
        self.lines_a_b_df = self._load_csv("lines_a_b.csv", index_col=None)  # type: ignore
        self.lines_b_a_df = self._load_csv("lines_b_a.csv", index_col=None)  # type: ignore

        # Create a new dataframe with zone information and weekly capacity data
        week_col = str(self.week)
        self.lines_week_df = self.lines_a_b_df[["from_zone", "to_zone"]].copy()
        self.lines_week_df["capacity_a_to_b"] = self.lines_a_b_df[week_col]
        self.lines_week_df["capacity_b_to_a"] = self.lines_b_a_df[week_col]

        # Repeat static line capacities over all time steps
        # Shape: (T, L)

        # self.time_index = pd.Index(np.arange(168), name="T")
        # The series for the conventional generator costs have to be 1-indexed to match the other time series (RES, demand, etc.)
        self.time_index = pd.Index(np.arange(1,168+1), name="T")
        self.T = len(self.time_index)

        self.lines_a_to_b_cap = np.outer(
            np.ones(self.T),  # Shape: (T,)
            self.lines_week_df["capacity_a_to_b"].to_numpy(),  # Shape: (L,)
        )

        self.lines_b_to_a_cap = np.outer(
            np.ones(self.T), 
            self.lines_week_df["capacity_b_to_a"].to_numpy(),
        )
        
    def map_transmission_lines(self):   
        
        # More efficient way to create lines and labels
        line_zones = self.lines_week_df[['from_zone', 'to_zone']].values
        self.lines = [tuple(row) for row in line_zones]  
        self.line_labels = [f"{a}-{b}" for a, b in line_zones]  # Create labels for each line
        self.lines = [tuple(x) for x in self.lines_week_df[['from_zone', 'to_zone']].to_numpy()]  # Shape: (L,)
        
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
                "L": self.line_labels,   # match with thermal_gen_bid_vol
                "Z": self.bidding_zones
            },
            dims=["L", "Z"]
        )

    def load_hydro_reservoir_data(self):
        """Load unit-specific data for various dispatchable generators."""
        self.hydro_res_units = self._load_csv('hydro_reservoir_units.csv')
        hydro_res_energy = self._load_csv('hydro_reservoir_energy.csv')
        self.hydro_res_energy = hydro_res_energy.loc[self.week]
        self.hydro_res_units_id = list(self.hydro_res_units.index)  # Shape: (G_hydro_res,)

        # We need to repeat the capacities for each hydro unit for all time steps:
        self.hydro_res_units_el_cap = np.outer(np.ones(self.T), self.hydro_res_units.capacity_el.to_numpy())

    def load_hydro_res_units_marginal_cost(self):
        # Convert the production cost pandas Series to a DataFrame with time index
        self.hydro_res_units_marginal_cost_series = self.hydro_res_units.prodcost
        self.hydro_res_units_marginal_cost_series.index.name = "G_hydro_res"

        self.hydro_res_units_marginal_cost_df = pd.DataFrame(
            data=np.broadcast_to(self.hydro_res_units_marginal_cost_series.to_numpy(),
                                (len(self.times),
                                len(self.hydro_res_units_marginal_cost_series))),
            index=self.times,
            columns=self.hydro_res_units_marginal_cost_series.index)

    def map_hydro_res_units_to_zones(self):
        """
        Build binary hydro reservoir-to-zone assignment matrix (G_hydro_res x Z).
        G_hydro_res_Z[g_hydro_res, z] = 1 if generator g_hydro_res belongs to zone z, else 0.
        """
        # Create dummy variables (one-hot encode) from generator zone assignment
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
        """Load unit-specific data for various pumped hydro units."""
        self.hydro_ps_units = self._load_csv('hydro_pumped_units.csv')
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
        # Convert the production cost pandas Series to a DataFrame with time index
            # Unlike reservoir hydro and conventional units, pumped hydro requires
            # both bid and offer prices, so two dataframes for "marginal costs" are
            # provided in a single dictionary
        # Initialize lists and dicts for dynamic handling
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
        self.conventional_units_df = self._load_csv('conventional_thermal_units.csv')
        self.conventional_units_id = list(self.conventional_units_df.index)       # Shape: (G,)

        # We need to repeat the capacities for each generator for all time steps:
        self.conventional_units_el_cap = np.outer(np.ones(self.T), self.conventional_units_df.capacity_el.to_numpy())

    def load_conventional_units_marginal_cost(self):
        # Convert the production cost pandas Series to a DataFrame with time index
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
        
    def load_ptx(self):
        """Load power-to-X unit demand data."""
        self.ptx_demand_df = self._load_csv('ptx_units.csv')

    def load_heating(self):
        """Load district heating demand data."""
        self.heating_demand_df = self._load_csv('district_heating_units.csv')

    def load_bess_units(self):
        """Load battery energy storage system (BESS) unit data."""
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
        # Convert the production cost pandas Series to a DataFrame with time index
            # Like pumped hydro storage, bess requires both bid and offer prices,
            # so two dataframes for "marginal costs" are provided in a single dictionary
        # Initialize lists and dicts for dynamic handling
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