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
    scenario_name: str
    scenario_config: dict
    config_yaml: dict
    logger: Logger
    root_path: Path
    week: int | None = None   
    
    def __post_init__(self):
        """
        Post-initialization to load and validate all required datasets.
        """
        # Initialize logger
        self.logger.info(f"-------------- DATA LOADER: {self.scenario_name} --------------")
        
        
        # Load YAML metadata (auxiliary scenario data)
        self._init_data_paths()
        self.load_yaml_aux_data()
        self._init_time_index()
        
        # Load CSV datasets
        self.load_generation_data()
        self.load_inflexible_demand_data()
        self.load_flexible_demand_data()
        self.map_hours_to_weeks()
        self.load_lines_data()
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
        self.load_bess_units()
        self.load_bess_units_marginal_cost()
        self.map_bess_units_to_zones()
        self.load_ptx_data()
        self.load_ptx_bid_prices()
        self.map_ptx_units_to_zones()
        self.load_dh_data()
        self.load_dh_bid_prices()
        self.map_dh_units_to_zones()

    def _init_data_paths(self) -> None:
        """Initialize all data directory paths according to the updated folder structure."""
        # Demand paths
        
        self.base_path = self.root_path / 'simulations' / f'{self.scenario_name}'
        self.data_path = self.base_path / 'data'
        
    def _init_time_index(self):
        """Initialize the time index for the scenario based on the loaded demand data."""
        
        if self.scenario_config.get('run_mode') == 'yearly':
            self.T = 8760
            
        elif self.scenario_config.get('run_mode') == 'weekly':
            self.T = 168
        
        self.time_index = np.arange(self.T)
        self.times = list(self.time_index)  
        # this is important: if the time index of the cost dataframes don't match the model variables' then it assumes 0-cost

    def _load_csv(self, filename: str, index_col=0) -> pd.DataFrame:
        """Helper function to load a CSV file. Raises FileNotFoundError if missing."""
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        return pd.read_csv(file_path, index_col=index_col)

    def _filter_by_week(self, df: pd.DataFrame, week: int, hours_per_week: int = 168, week_col: str = 'Week') -> pd.DataFrame:
        """
        Slice a continuous hourly dataframe into week-sized blocks.
        The final week absorbs any leftover hours.

        Example for length 8760:
            Weeks 1–51: 168 rows each
            Week 52:    168 + 24 = 192 rows (final remainder)
        """
        
        # TODO: Delete these two following lines, as the rest of the funciton already creats the week index and then slides based on that. In the input data of the preprocessing, it should be removed the week column as well.
        if week_col in df.columns:
            return df[df[week_col] == self.week].drop(columns=week_col)

        if week < 1:
            raise ValueError("week must be >= 1")

        n = len(df)
        n_full_weeks = n // hours_per_week       # e.g. 8760 // 168 = 52
        remainder = n % hours_per_week           # e.g. 24

        # Last week number (includes the remainder)
        last_week = n_full_weeks if remainder == 0 else n_full_weeks

        if week > last_week:
            raise IndexError(
                f"Week {week} is out of range. Maximum allowable week is {last_week}."
            )

        # Standard start index
        start = (week - 1) * hours_per_week

        # For the last week: extend to the end of df
        if week == last_week:
            end = n
        else:
            end = start + hours_per_week

        return df.iloc[start:end].copy()

    def load_yaml_aux_data(self):
        """
        Load auxiliary metadata for the scenario from a YAML file.
        Exposes the content as a dictionary: self.yaml_data
        and attaches key values as class attributes (flattened).
        """
        yaml_path = self.data_path / f'{self.scenario_name}_aux_data.yaml'
        
        with open(yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)

        # Flatten and assign attributes for convenience
        self.scenario_name = self.yaml_data.get('scenario_name')
        self.prediction_year = self.yaml_data.get('prediction_year')
        self.bidding_zones = self.yaml_data.get('bidding_zones', [])
        self.solver_name = self.yaml_data.get('solver_name')

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
        
        self.wind_onshore_production = self._load_csv('wind_onshore_production.csv')
        self.wind_offshore_production = self._load_csv('wind_offshore_production.csv')
        self.solar_pv_production = self._load_csv('solar_pv_production.csv')
        self.hydro_ror_production = self._load_csv('hydro_ror_production.csv')
        
        if self.scenario_config.get('run_mode') == 'weekly':
            self.wind_onshore_production = self._filter_by_week(self.wind_onshore_production, week=self.week) # type: ignore
            self.wind_offshore_production = self._filter_by_week(self.wind_offshore_production, week=self.week) # type: ignore
            self.solar_pv_production = self._filter_by_week(self.solar_pv_production, week=self.week) # type: ignore
            self.hydro_ror_production = self._filter_by_week(self.hydro_ror_production, week=self.week) # type: ignore
                
        #utils.validate_df_positive_numeric(self.wind_onshore_production, "wind_onshore_production")
        utils.validate_df_positive_numeric(self.wind_offshore_production, "wind_offshore_production")
        utils.validate_df_positive_numeric(self.solar_pv_production, "solar_pv_production")
        utils.validate_df_positive_numeric(self.hydro_ror_production, "hydro_ror_production")

    def load_inflexible_demand_data(self):
        """Load inflexible demand data (classic and EV) filtered by week and validate."""
        self.demand_inflexible_classic = self._load_csv('demand_inflexible_classic.csv')
        self.demand_inflexible_ev = self._load_csv('demand_inflexible_ev.csv')
        
        if self.scenario_config.get('run_mode') == 'weekly':
            self.demand_inflexible_classic = self._filter_by_week(self.demand_inflexible_classic, week=self.week) # type: ignore
            self.demand_inflexible_ev = self._filter_by_week(self.demand_inflexible_ev, week=self.week) # type: ignore
            
        utils.validate_df_positive_numeric(self.demand_inflexible_classic, "demand_inflexible_classic")
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
                    arr1 = np.repeat(flex_dem_df.to_numpy(), 168, axis=0) # shape: (8736, Z)
                    arr2 = np.outer(np.ones(self.T % flex_dem_df.shape[0]), flex_dem_df.iloc[-1, :].to_numpy())  # shape: (24, Z)
                    self.flexible_demands_dfs[flex_load][param] = np.concatenate((arr1, arr2), axis=0)  # shape: (8760, Z) # type: ignore
                    # self.flexible_demands_dfs[flex_load][param] = np.outer(np.ones(self.T),
                    #                                                        flex_dem_df.loc[self.week].to_numpy())
                else:  # param == "amount" in which case we only need the values for the specific week w/o repetition
                    flex_dem_df.index.name = "W"  # change index name from "Week" to "W" to correspond to the mapping from hours to weeks.
                    self.flexible_demands_dfs[flex_load][param] = flex_dem_df  #.loc[self.week]

        # call in energy_model.py as e.g.:
        #   self.data.flexible_demands_dfs['demand_flexible_classic']['capacity']
        
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
        self.lines_a_b_df = self._load_csv("lines_a_b.csv", index_col=0)  # type: ignore
        self.lines_b_a_df = self._load_csv("lines_b_a.csv", index_col=0)  # type: ignore

        # Create a new dataframe with zone information and weekly capacity data
        # week_col = str(self.week)
        # self.lines_week_df = self.lines_a_b_df[["from_zone", "to_zone"]].copy()
        # self.lines_week_df["capacity_a_to_b"] = self.lines_a_b_df[week_col]
        # self.lines_week_df["capacity_b_to_a"] = self.lines_b_a_df[week_col]
        
        # Get the names of the weeks
        week_cols = self.lines_a_b_df.columns[~self.lines_a_b_df.columns.isin(['from_zone', 'to_zone'])]
        
        # Repeat the weekly capacity for all hours in that week for lines a to b
        arr_a_to_b_1 = np.repeat(self.lines_a_b_df[week_cols].to_numpy(), 168, axis=1) # shape: (L, 8736)
        arr_a_to_b_2 = np.outer(np.ones(self.T%len(week_cols)), self.lines_a_b_df[week_cols[-1]].to_numpy()).T  # shape: (L, 24)
        lines_a_to_b_cap = np.concatenate((arr_a_to_b_1, arr_a_to_b_2), axis=1)  # shape: (L, 8760)
        self.lines_a_to_b_cap = lines_a_to_b_cap.T  #shape: (8760, L)

        # Repeat the weekly capacity for all hours in that week for lines b to a
        arr_b_to_a_1 = np.repeat(self.lines_b_a_df[week_cols].to_numpy(), 168, axis=1) # shape: (L, 8736)
        arr_b_to_a_2 = np.outer(np.ones(self.T%len(week_cols)), self.lines_b_a_df[week_cols[-1]].to_numpy()).T  # shape: (L, 24)
        lines_b_to_a_cap = np.concatenate((arr_b_to_a_1, arr_b_to_a_2), axis=1)  # shape: (L, 8760)
        self.lines_b_to_a_cap = lines_b_to_a_cap.T  #shape: (8760, L)
        # Repeat static line capacities over all time steps
        # Shape: (T, L)

        # self.time_index = pd.Index(np.arange(168), name="T")
        # The series for the conventional generator costs have to be 1-indexed to match the other time series (RES, demand, etc.)
        # self.time_index = pd.Index(np.arange(1,len(self.times)+1), name="T")
        # self.T = len(self.time_index)

        # self.lines_a_to_b_cap = np.outer(
        #     np.ones(self.T),  # Shape: (T,)
        #     self.lines_week_df["capacity_a_to_b"].to_numpy(),  # Shape: (L,)
        # )

        # self.lines_b_to_a_cap = np.outer(
        #     np.ones(self.T), 
        #     self.lines_week_df["capacity_b_to_a"].to_numpy(),
        # )
        
    def map_transmission_lines(self):   
        
        # More efficient way to create lines and labels
        line_zones = self.lines_a_b_df[["from_zone", "to_zone"]].values
        self.lines = [tuple(row) for row in line_zones]  
        self.line_labels = [f"{a}-{b}" for a, b in line_zones]  # Create labels for each line
        self.lines = [tuple(x) for x in self.lines_a_b_df[['from_zone', 'to_zone']].to_numpy()]  # Shape: (L,)
        
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
        # Reservoir hydro units
        self.hydro_res_units = self._load_csv('hydro_reservoir_units.csv')
        self.hydro_res_units_id = list(self.hydro_res_units.index)  # Shape: (G_hydro_res,)
        
        if self.scenario_config.get('plant_aggregation'):

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
        
        if self.scenario_config.get('run_mode') == 'weekly':
            self.hydro_res_energy = hydro_res_energy.loc[self.week]
        else:
            self.hydro_res_energy = hydro_res_energy  #.loc[self.week]

    def load_hydro_res_units_marginal_cost(self):

        if self.scenario_config.get('plant_aggregation'):
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
        if self.scenario_config.get('plant_aggregation'):
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
        """Load unit-specific data for various pumped hydro units."""
        self.hydro_ps_units = self._load_csv('hydro_pumped_units.csv')


        if self.scenario_config.get('plant_aggregation'):
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
                #dims=["T", "G_hydro_ps"],
                #coords=(self.times, self.hydro_ps_units_id))
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
        # Convert the production cost pandas Series to a DataFrame with time index
            # Unlike reservoir hydro and conventional units, pumped hydro requires
            # both bid and offer prices, so two dataframes for "marginal costs" are
            # provided in a single dictionary
        # Initialize lists and dicts for dynamic handling
        
        if self.scenario_config.get('plant_aggregation'):
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
        if self.scenario_config.get('plant_aggregation'):
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
        self.conventional_units_df = self._load_csv('conventional_thermal_units.csv')

        if self.scenario_config.get('plant_aggregation'):
            
            # Aggregate thermal plants by fuel type (and zone)
            self.agg_g = utils.agg_by_zone_tech(self.conventional_units_df)

            # self.conventional_units_id = list(self.conventional_units_df.index)       # Shape: (G,)
            # Set a new index with combined zone and fuel type for the aggregated generators
            self.agg_g = utils.set_agg_idx(self.agg_g)
            self.conventional_units_id = list(self.agg_g.index)

            # # We need to repeat the capacities for each generator for all time steps:
            #self.conventional_units_el_cap = np.outer(np.ones(self.T), self.conventional_units_df.capacity_el.to_numpy())

            # We repeat the capacities for each FUEL (by zone and fuel) of generator
            self.conventional_units_el_cap = np.outer(np.ones(self.T), self.agg_g.capacity_el.to_numpy())
            
        else:
            # We need to repeat the capacities for each generator for all time steps:
            self.conventional_units_id = list(self.conventional_units_df.index)       # Shape: (G,)
            self.conventional_units_el_cap = np.outer(np.ones(self.T), self.conventional_units_df.capacity_el.to_numpy())

    def load_conventional_units_marginal_cost(self):
        # Convert the production cost pandas Series to a DataFrame with time index
        if self.scenario_config.get('plant_aggregation'):
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
        #self.G_Z_df = pd.get_dummies(self.conventional_units_df['zone_el']).astype(int)
        if self.scenario_config.get('plant_aggregation'):
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
        """Load battery energy storage system (BESS) unit data."""
        self.bess_units_df = self._load_csv('bess_units.csv')
        
        if self.scenario_config.get('plant_aggregation'):
            # Aggregate BESS units by zone (only one type of tech/size)
            self.agg_bess = utils.agg_storage_by_zone(
                df=self.bess_units_df,
                bid_price="charge_bid_price",
                offer_price="discharge_bid_price",
                storage_cap="storage_capacity")

            # Fill in 0s for any zones that do not has PHS
            self.agg_bess = self.agg_bess.reindex(self.bidding_zones, fill_value=0)

            # self.bess_units_id = list(self.bess_units_df.index)  # Shape: (G_bess,)
            # No longer any unit id's when aggregating by zone


            # We need to repeat the charge/discharge and storage capacities for each bess unit for all time steps:
            self.bess_units_el_cap = np.outer(
                np.ones(self.T),
                self.agg_bess.capacity_el.to_numpy()
            )
            self.bess_units_storage_cap = np.outer(
                np.ones(self.T),
                self.agg_bess.capacity_stor.to_numpy()
            )

            # Create an xarray DataArray of the initial SOC in MWh to avoid "UserWarning".
            #   This is identical to the hydro_ps initial SOC handling above.
            bess_initial_SOC_x_storage_cap = self.bess_initial_SOC * self.bess_units_storage_cap
            bess_initial_SOC_x_storage_cap[1:, :] = 0  # Initial SOC only applies in the first hour (T=0)
            self.bess_initial_SOC_x_storage_cap_xr = xr.DataArray(
                data=bess_initial_SOC_x_storage_cap,
                # dims=["T", "G_bess"],
                # coords=(self.times, self.bess_units_id))
                dims=["T", "Z"],
                coords=(self.times, self.bidding_zones)
            )
            
        else:
            
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

        if self.scenario_config.get('plant_aggregation'):
            
            bid_and_offer_new_col_names = ["Bid_price", "Offer_price"]
            
            self.bess_units_marginal_cost_dfs = {}
            
            for k in bid_and_offer_new_col_names:
                self.bess_units_marginal_cost_dfs[k] = pd.DataFrame(
                    data=np.broadcast_to(
                        self.agg_bess[k.lower()+"_weighted"].to_numpy(),
                        (len(self.times),
                        len(self.bidding_zones))
                    ),
                    index=self.times,
                    columns=self.bidding_zones
            )
                
        else:
            
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
        
        
        if self.scenario_config.get('plant_aggregation'):
            pass
        else:
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
        self.ptx_units_df = self._load_csv('ptx_units.csv')
        self.ptx_units_id = list(self.ptx_units_df.index)  # shape (L_DH)

        # Repeat capacities for each time steps
        self.ptx_units_el_cap = np.outer(np.ones(self.T),
                                        self.ptx_units_df["Electric capacity"].to_numpy())
        
    def load_ptx_bid_prices(self):
        """
        Makes a (T, L_PtX) DataFrame of the PtX unit bid prices.
        These bid prices are simply the LCoX of each plant.
        """
        # Convert the bid prices pandas Series to a DataFrame with time index
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
        self.dh_units_df = self._load_csv('dh_units.csv')
        self.dh_units_id = list(self.dh_units_df.index)  # shape (L_DH)

        # Repeat capacities for each time steps
        self.dh_units_el_cap = np.outer(np.ones(self.T),
                                        self.dh_units_df["Thermal capacity"].to_numpy())
        
    def load_dh_bid_prices(self):
        # Convert the bid prices pandas Series to a DataFrame with time index
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
