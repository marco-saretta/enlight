import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import linopy

def validate_df_positive_numeric(df: pd.DataFrame, name: str, check_numeric: bool = True, check_positive: bool = True) -> None:
    """
    Validate that all values in the DataFrame meet numeric and positivity criteria.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        name (str): Name used in error messages.
        check_numeric (bool): Whether to ensure all values are numeric.
        check_positive (bool): Whether to ensure all values are non-negative.

    Raises:
        ValueError: If numeric conversion fails or if negative values are found.
    """
    if check_numeric:
        try:
            df.astype(float)
        except ValueError as e:
            raise ValueError(f"[{name}] DataFrame contains non-numeric values:\n{e}")

    if check_positive:
        if (df < 0).any().any():
            raise ValueError(f"[{name}] DataFrame contains negative values.")
            
def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        raise ValueError(f"File {path} is empty.")
    return df

def setup_logging(log_dir: str = "logs", log_file: str = "enlight.log") -> logging.Logger:
    """
    Configure logging, create log folder if it doesn't exist, and return a logger.

    Args:
        log_dir (str): Folder to store log files (default "logs").
        log_file (str): Log file name (default "app.log").

    Returns:
        logging.Logger: Configured logger.
    """
    # Ensure the logs folder exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Full path for the log file
    file_path = log_path / log_file

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),        # Console output
            logging.FileHandler(file_path)  # File output
        ]
    )

    return logging.getLogger(__name__)
    
def save_data(
    data: pd.DataFrame,
    filename: str,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Save a DataFrame to CSV in the specified output directory.

    Args:
        data (pd.DataFrame): Data to save.
        filename (str): Name of the CSV file.
        output_dir (Optional[Path]): Directory to save the file. Defaults to current working directory.
        logger (Optional[logging.Logger]): Logger for info messages. Defaults to None.

    Returns:
        Path: Full path to the saved file.
    """
    # Default directory is current working directory
    output_dir = Path(output_dir or Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename
    data.to_csv(file_path, index=True)  # Save row labels

    if logger:
        logger.info(f"Saved DataFrame to {file_path}")

    return file_path

def check_vars_list(vre_list: list, dem_list: list, gen_units_list: list, stor_units_list: list, dem_units_list: list, model_vars: linopy.variables.Variables):
    # Verifies that all variables are indeed accounted for when post-calculating social welfare.
    #   Only 'export' and 'SOC'-variables may be overlooked.
    stor_expanded = []
    for s in stor_units_list:
        stor_expanded.extend([f"{s}_offer", f"{s}_bid"])
    vars_accounted_for = vre_list + dem_list + gen_units_list + stor_expanded + dem_units_list + ['lineflow']
    vars_accounted_for = list(map(lambda x: x.replace("_sol",""), vars_accounted_for))
    vars_unaccounted_for = set(model_vars) - set(vars_accounted_for)
    for v in vars_unaccounted_for:
        if not (v.endswith('_SOC') or v == 'export'):
            raise Exception(f"Variable {v} is not accounted for in the calculation of social welfare.\nAll variables unaccounted for are: {vars_unaccounted_for} of which SOC's and export shouldn't be accounted for.")

def save_model_results(self):#, week: int):
    """
    Calculate and save results from the optimized model.
    """
    results_path = Path(self.simulation_path) / "results" #/ f"week_{week}"
    results_path.mkdir(parents=True, exist_ok=True)
    
    self.logger.info('Saving the results')
    try:
        # --- Step 1: Initialize dictionary with all outputs ---
        self.results_dict = {}

        # Helper to extract solution
        def get_solution(var):
            return var.solution.to_dataframe().squeeze().unstack()

        # --- Step 2: Calculate values ---
        self.results_dict.update({
            # Bids
            "demand_inflexible_classic_bid_sol": get_solution(self.demand_inflexible_classic_bid),
            "demand_flexible_classic_bid_sol": get_solution(self.demand_flexible_classic_bid),
            "hydro_ps_units_bid_sol": get_solution(self.hydro_ps_units_bid),
            "bess_units_bid_sol": get_solution(self.bess_units_bid),
            "ptx_units_bid_sol": get_solution(self.ptx_units_bid),
            "dh_units_bid_sol": get_solution(self.dh_units_bid),

            # Offers
            "wind_onshore_offer_sol": get_solution(self.wind_onshore_offer),
            "wind_offshore_offer_sol": get_solution(self.wind_offshore_offer),
            "solar_pv_offer_sol": get_solution(self.solar_pv_offer),
            "hydro_ror_offer_sol": get_solution(self.hydro_ror_offer),
            "conventional_units_offer_sol": get_solution(self.conventional_units_offer),
            "hydro_res_units_offer_sol": get_solution(self.hydro_res_units_offer),
            "hydro_ps_units_offer_sol": get_solution(self.hydro_ps_units_offer),
            "bess_units_offer_sol": get_solution(self.bess_units_offer),

            # Transmission
            "electricity_export_sol": get_solution(self.electricity_export),
            "lineflow_sol": get_solution(self.lineflow),
        })

        # Derived results
        self.results_dict["demand_inflexible_classic_bid_curt"] = (
            self.data.demand_inflexible_classic - self.results_dict["demand_inflexible_classic_bid_sol"]
        )
        self.results_dict["wind_on_bid_vol_curt"] = (
            self.data.wind_onshore_production - self.results_dict["wind_onshore_offer_sol"]
        )
        self.results_dict["electricity_prices"] = self.model.dual["power_balance"].to_pandas()

        # # ###### ADDED a while ago but updated to be faster - MARGINAL GENERATOR ######
        # # # Read out the marginal generator by its unit name
        thermal_el_cap_xr = xr.DataArray(
            self.data.conventional_units_el_cap,
            dims=["T", "G"],
            coords=[self.times, self.data.conventional_units_id]
        )
        thermal_marginal_mask = (self.conventional_units_offer.solution > 0) & (self.conventional_units_offer.solution < thermal_el_cap_xr)
        thermal_marginal_costs = self.data.conventional_units_marginal_cost_df.where(thermal_marginal_mask.to_pandas())

        hydro_res_el_cap_xr = xr.DataArray(
            self.data.hydro_res_units_el_cap,
            dims=["T", "G_hydro_res"],
            coords=[self.times, self.data.hydro_res_units_id]
        )
        hydro_res_marginal_mask = (self.hydro_res_units_offer.solution > 0) & (self.hydro_res_units_offer.solution < hydro_res_el_cap_xr)
        hydro_res_marginal_costs = self.data.hydro_res_units_marginal_cost_df.where(hydro_res_marginal_mask.to_pandas())

        hydro_ps_el_cap_xr = xr.DataArray(
            self.data.hydro_ps_units_el_cap,
            dims=["T", "G_hydro_ps"],
            coords=[self.times, self.data.hydro_ps_units_id]
        )
        hydro_ps_marginal_mask = (self.hydro_ps_units_offer.solution > 0) & (self.hydro_ps_units_offer.solution < hydro_ps_el_cap_xr)
        hydro_ps_marginal_costs = self.data.hydro_ps_units_marginal_cost_dfs["Offer_price"].where(hydro_ps_marginal_mask.to_pandas())

        bess_el_cap_xr = xr.DataArray(
            self.data.bess_units_el_cap,
            dims=["T", "G_bess"],
            coords=[self.times, self.data.bess_units_id]
        )
        bess_marginal_mask = (self.bess_units_offer.solution > 0) & (self.bess_units_offer.solution < bess_el_cap_xr)
        bess_marginal_costs = self.data.bess_units_marginal_cost_dfs["Offer_price"].where(bess_marginal_mask.to_pandas())

        # Build final dataframe
        marginal_generators_df = pd.DataFrame({
            'thermal_generator_id': [
                tuple(thermal_marginal_costs.columns[thermal_marginal_mask.sel(T=t).values])
                for t in self.times
            ],
            'thermal_generator_cost': [
                tuple(thermal_marginal_costs.loc[t].dropna().values)
                for t in self.times
            ],
            'hydro_res_generator_id': [
                tuple(hydro_res_marginal_costs.columns[hydro_res_marginal_mask.sel(T=t).values])
                for t in self.times
            ],
            'hydro_res_generator_cost': [
                tuple(hydro_res_marginal_costs.loc[t].dropna().values)
                for t in self.times
            ],
            'hydro_ps_generator_id': [
                tuple(hydro_ps_marginal_costs.columns[hydro_ps_marginal_mask.sel(T=t).values])
                for t in self.times
            ],
            'hydro_ps_generator_cost': [
                tuple(hydro_ps_marginal_costs.loc[t].dropna().values)
                for t in self.times
            ],
            'bess_generator_id': [
                tuple(bess_marginal_costs.columns[bess_marginal_mask.sel(T=t).values])
                for t in self.times
            ],
            'bess_generator_cost': [
                tuple(bess_marginal_costs.loc[t].dropna().values)
                for t in self.times
            ]
        }, index=self.data.times)

        self.results_dict["marginal_generator"] = marginal_generators_df

        ################################################

        # Save all results ---
        for name, df in self.results_dict.items():
            if df is not None:  # only save if calculated
                df.to_csv(results_path / f"{name}.csv")
        self.logger.info('Files saved')
    except Exception as e:
        print(f"Error saving results: {e}")
        # raise e

    self.logger.info("Processing the results")

    # Process the results to reproduce the social welfare from the optimization model
    try:
        self.results_econ = {"revenues":{} , "costs":{}, 
                         "profits":{}, "profits_sw": {},
                         "profits_tot": {},
                         "consumer surplus" : float,
                         "producer surplus" : float,
                         "producer surplus perceived" : float,
                         "social welfare": float,
                         "social welfare perceived": float}


        # List generators that are given BY ZONE - in this case only: VRE technologies.
        #   And put relevant data in a corresponding dictionary.
        vre_list = ['wind_onshore_offer_sol', 'wind_offshore_offer_sol', 'solar_pv_offer_sol', 'hydro_ror_offer_sol']
        VRE_costs = dict(zip(vre_list, [self.data.wind_onshore_bid_price, self.data.wind_offshore_bid_price, self.data.solar_pv_bid_price, self.data.hydro_ror_bid_price]))

        # List demands and put relevant data in a dict.
        dem_list = ['demand_inflexible_classic_bid_sol', 'demand_flexible_classic_bid_sol']
        dem_bid_price = dict(zip(dem_list, [self.data.voll_classic, self.data.wtp_classic]))

        # List generators that are given BY UNIT.
        gen_units_list = ['conventional_units_offer_sol', 'hydro_res_units_offer_sol']
        gen_units_map_to_zone = dict(zip(gen_units_list, [self.data.G_Z_df, self.data.G_hydro_res_Z_df]))
        gen_units_marginal_cost_dfs = dict(zip(gen_units_list, [self.data.conventional_units_marginal_cost_df, self.data.hydro_res_units_marginal_cost_df]))

        # List storage types (BY UNIT).
        stor_units_list = ['bess_units', 'hydro_ps_units']
        stor_units_map_to_zone = dict(zip(stor_units_list, [self.data.G_bess_Z_df, self.data.G_hydro_ps_Z_df]))
        stor_units_marginal_cost_dfs = dict(zip(stor_units_list, [self.data.bess_units_marginal_cost_dfs, self.data.hydro_ps_units_marginal_cost_dfs]))

        # List demands that are given BY UNIT.
        dem_units_list = ['ptx_units_bid_sol', 'dh_units_bid_sol']
        dem_units_map_to_zone = dict(zip(dem_units_list, [self.data.L_PtX_Z_df, self.data.L_DH_Z_df]))
        dem_units_bid_prices_dfs = dict(zip(dem_units_list, [self.data.ptx_units_bid_prices_df, self.data.dh_units_bid_prices_df]))

        # Verify that all variables are indeed accounted for. Only 'export' and 'SOC'-variables may be overlooked.
        check_vars_list(vre_list=vre_list,
                        dem_list=dem_list,
                        gen_units_list=gen_units_list,
                        stor_units_list=stor_units_list,
                        dem_units_list=dem_units_list,
                        model_vars=self.model.variables)

        # Get VRE revenues and operatings costs
        for vre in vre_list:
            vre_ = vre.replace("_offer_sol", "")  # prettifying keys
            self.results_econ["revenues"][vre_] = (self.results_dict[vre].mul(self.results_dict['electricity_prices'], axis='columns')).sum(axis=0)  # sum across hours
            self.results_econ["costs"][vre_] = (self.results_dict[vre] * VRE_costs[vre]).sum(axis=0)

        # Get demands utilities and power costs
        for dem in dem_list:
            dem_ = dem.replace("_bid_sol","")
            # Revenue for a demand is the utility
            self.results_econ["revenues"][dem_] = (self.results_dict[dem] * dem_bid_price[dem]).sum(axis=0)
            # Costs for a demand is the power costs
            self.results_econ["costs"][dem_] = (self.results_dict[dem].mul(self.results_dict['electricity_prices'], axis='columns')).sum(axis=0)

        # Get conventional and hydro res units revenues and operatings costs
        for gen in gen_units_list:
            gen_ = gen.replace("_offer_sol", "")
            self.results_econ["revenues"][gen_] = (self.results_dict[gen] * (self.results_dict['electricity_prices'].dot(gen_units_map_to_zone[gen].T))).sum(axis=0)
            self.results_econ["costs"][gen_] = (self.results_dict[gen] * gen_units_marginal_cost_dfs[gen]).sum(axis=0)

        # Get storage revenues and costs
        # The "true" revenue of the bess is the revenue of the offer. The "true" cost is the power cost from the bid. However, when adding all profits one needs to take the bid revenue and the offer cost into account as well. This is what's done in "profits_sw".
        for stor_ in stor_units_list:
            # Auxiliary variables
            stor_offer = stor_ + "_offer_sol"
            stor_bid = stor_ + "_bid_sol"
            # Get the offer revenue and bid power costs
            self.results_econ["revenues"][stor_] = (self.results_dict[stor_offer] * (self.results_dict['electricity_prices'].dot(stor_units_map_to_zone[stor_].T))).sum(axis=0)
            self.results_econ["costs"][stor_] = (self.results_dict[stor_bid] * (self.results_dict['electricity_prices'].dot(stor_units_map_to_zone[stor_].T))).sum(axis=0)

            # Calculate the social welfare when considering the bid as a "normal" load and the offer as a "normal" generator.
            profit_offer = (self.results_dict[stor_offer] * (self.results_dict['electricity_prices'].dot(stor_units_map_to_zone[stor_].T)) - self.results_dict[stor_offer] * stor_units_marginal_cost_dfs[stor_]["Offer_price"]).sum(axis=0)
            
            profit_bid = (self.results_dict[stor_bid] * stor_units_marginal_cost_dfs[stor_]['Bid_price'] - self.results_dict[stor_bid] * (self.results_dict['electricity_prices'].dot(stor_units_map_to_zone[stor_].T))).sum(axis=0)
            
            self.results_econ['profits_sw'][stor_] = (profit_offer + profit_bid).sum()

        # Get PtX and DH units utilities and power costs
        for dem in dem_units_list:
            dem_ = dem.replace("_bid_sol", "")
            # Revenue for a demand is the utility
            self.results_econ["revenues"][dem_] = (self.results_dict[dem] * dem_units_bid_prices_dfs[dem]).sum(axis=0)
            # Costs are simply power costs as opposed to operating costs for a generator
            self.results_econ["costs"][dem_] = (self.results_dict[dem] * (self.results_dict['electricity_prices'].dot(dem_units_map_to_zone[dem].T))).sum(axis=0)

        # Transmission System Operater: congestion rent
        # Align index (they are slightly different)
        self.data.L_Z_df.index = self.results_dict['lineflow_sol'].columns
        self.results_econ['profits']['lineflow'] = (-self.results_dict['lineflow_sol'].dot(self.data.L_Z_df) * self.results_dict['electricity_prices']).sum(axis=0)

        # Calculate profits for other participants
        for k in self.results_econ['revenues'].keys():
            self.results_econ['profits'][k] = self.results_econ['revenues'][k] - self.results_econ['costs'][k]
            self.results_econ['profits_tot'][k] = float(np.round(self.results_econ['profits'][k].sum()/1e9,4))

        # Compare social welfare from this function and from the model.
        self.results_econ["consumer surplus"] = sum(map(lambda x: self.results_econ['profits'][x.replace("_bid_sol","")].sum(), dem_list+dem_units_list))
        self.results_econ["producer surplus"] = sum(map(lambda x: self.results_econ['profits'][x.replace("_offer_sol","")].sum(), vre_list+gen_units_list+stor_units_list+['lineflow']))
        self.results_econ["producer surplus perceived"] = sum(map(lambda x: self.results_econ['profits'][x.replace("_offer_sol","")].sum(), vre_list+gen_units_list+['lineflow'])) + sum(map(lambda x: self.results_econ['profits_sw'][x].sum(), stor_units_list))
        self.results_econ["social welfare"] =  self.results_econ["producer surplus"] + self.results_econ["consumer surplus"]
        self.results_econ["social welfare perceived"] =  self.results_econ["producer surplus perceived"] + self.results_econ["consumer surplus"]

    except Exception as e:
        print(f"Error processing results: {e}")
        # raise e
    self.logger.info('Results processed')
    
def make_aggregated_supply_and_demand_curves(demand_curve_unsorted, supply_curve_unsorted, colors=['#990000', '#2F3EEA']):
    '''
    Plots the aggregated supply and demand curves provided (unsorted) as:
    - demand_curve_unsorted: np.ndarray (2, D)
    - supply_curve_unsorted: np.ndarray( 2, S)
    Where D and S is the number of demands and suppliers.
    In both arrays:
        1st row should be QUANTITY: consumption required or unit capacity
        2nd row is PRICE: bid or offer

    The function returns the plot object:
    - fig
    - ax
    '''
    # Get the decreasing order of bids based on their WTPs
    demand_merit_order = np.argsort(demand_curve_unsorted[1, :])[::-1]
    # Get the increasing order of offers based on their marginal costs
    supply_merit_order = np.argsort(supply_curve_unsorted[1, :])

    # Sort the bids in decreasing order based on their WTPs
    demand_curve = demand_curve_unsorted[:, demand_merit_order].copy()
    # Sort the offers in increasing order based on their marginal costs
    supply_curve = supply_curve_unsorted[:, supply_merit_order].copy()

    # Aggregate the quantity of the bids
    demand_curve[0,:] = np.cumsum(demand_curve[0,:])
    # Aggregate the quantity of the offers
    supply_curve[0,:] = np.cumsum(supply_curve[0,:])
    # Insert starting point of the curve. Without these the curves start at the capacity of the first bid/offer which would be counter-intuitive.
    demand_curve = np.insert(demand_curve, 0, np.array((0,demand_curve[1,0])), 1)
    supply_curve = np.insert(supply_curve, 0, np.array((0, supply_curve[1,0])), 1)

    fig, ax = plt.subplots(figsize=(8,6))
    y_max = max(max(supply_curve[1, :]), max(demand_curve[1, :]))
    y_min = min(min(supply_curve[1, :]), min(demand_curve[1, :]))

    # Draw aggregate demand and supply curves
    # 'steps-pre' draws the lines at the x-coordinate given. 'steps-post' would draw the line at the next x-coordinate.
    sns.lineplot(ax=ax,
                 x=demand_curve[0, :],
                 y=demand_curve[1, :],
                 drawstyle='steps-pre',
                 color=colors[0],
                 label='demand')
    sns.lineplot(ax=ax,
                 x=supply_curve[0, :],
                 y=supply_curve[1, :],
                 drawstyle='steps-pre',
                 color=colors[1],
                 label='supply')

    # Show a wider price range
    ax.set_ylim((min(1.05*y_min, 0.95*y_min), 1.05*y_max))

    # Add vertical line at the end of the demand curve to show
    #   the intersection with the supply curve
    ax.plot(
        [demand_curve[0, -1], demand_curve[0, -1]],  # at the equilibrium amount
        [ax.get_ylim()[0], demand_curve[1, -1]],  # from the bottom of the plot to the WTP of the last demand
        color=colors[0],
        linestyle='--'
    )

    # Return the plot object so it can be used for visualizing results on top of the aggregated market curves as well
    return fig, ax

def get_unsorted_aggregated_market_curves_from_dataloader_object(example_hour, dataloader_obj):
    '''
    This function calculates the unsorted aggregated market curves from a DataLoader object.
    Its output is used as input to make_aggregated_supply_and_demand_curves(). It is used
    to visualize both data and results which is why it is included simply as a function and not
    as a method.
    '''
    h = example_hour  # descriptive name as input, but simple name is more functional

    # Collect all of the DEMAND
    demand_curve_raw = np.vstack([
        # 1st column is the bid quantity
        np.hstack([
        # Inflexible demands
            # classic
            dataloader_obj.demand_inflexible_classic.sum(axis=1).iloc[h],
        # Flexible demands
            # classic
            dataloader_obj.flexible_demands_dfs['demand_flexible_classic']['capacity'].sum(axis=1)[h],
            # Storage
                # BESS
            dataloader_obj.bess_units_df.capacity_el.values,
                # PHS
            dataloader_obj.hydro_ps_units.capacity_el.values,
            # PtX
            dataloader_obj.ptx_units_df["Electric capacity"].values,
            # DH
            dataloader_obj.dh_units_df["Thermal capacity"].values
        ])
        ,
        # 2nd column is the bid price (VOLL or WTP)
        np.hstack([
            dataloader_obj.voll_classic,
            dataloader_obj.wtp_classic,
            dataloader_obj.bess_units_marginal_cost_dfs['Bid_price'].iloc[h].values,
            dataloader_obj.hydro_ps_units_marginal_cost_dfs['Bid_price'].iloc[h].values,
            dataloader_obj.ptx_units_df["Demand price"].values,
            dataloader_obj.dh_units_df["demand_price"].values
        ])
    ])


    # Collect all of the SUPPLY
    supply_curve_raw = np.vstack([
        # 1st column is the offer quantity
        np.hstack([
        # VREs
            dataloader_obj.solar_pv_production.sum(axis=1).iloc[h],
            dataloader_obj.wind_onshore_production.sum(axis=1).iloc[h],
            dataloader_obj.wind_offshore_production.sum(axis=1).iloc[h],
            dataloader_obj.hydro_ror_production.sum(axis=1).iloc[h],
        # Dispatchable
            dataloader_obj.conventional_units_df.capacity_el.values,
            dataloader_obj.hydro_res_units.capacity_el.values,
        # Storages
            dataloader_obj.bess_units_df.capacity_el.values,
            dataloader_obj.hydro_ps_units.capacity_el.values
        ])
        ,
        # 2nd column is the offer price (marginal cost)
        np.hstack([
        # VREs
            dataloader_obj.solar_pv_bid_price,
            dataloader_obj.wind_onshore_bid_price,
            dataloader_obj.wind_offshore_bid_price,
            dataloader_obj.hydro_ror_bid_price,
        # Dispatchable
            dataloader_obj.conventional_units_df.prodcost.values,
            dataloader_obj.hydro_res_units.prodcost.values,
        # Storages
            dataloader_obj.bess_units_marginal_cost_dfs['Offer_price'].iloc[h].values,
            dataloader_obj.hydro_ps_units_marginal_cost_dfs['Offer_price'].iloc[h].values
        ])
    ])

    return demand_curve_raw, supply_curve_raw

# def combine_simulations_result(weeks: list, result_path: Path, result: str):
#     '''
#     This function combines dataframes of a single result e.g. electricity prices
#     across multiple simulations (weeks) to arrive at a combined dataframe for all of the
#     simulations, e.g. the electricity prices for the entire year specified in configuration file.
#     '''
#     # Initialize an empty dict to store all of the dataframes after loading
#     df_dict = {}
#     for w in weeks:
#         df_dict[f"week_{w}"] = pd.read_csv(result_path / f"week_{w}/{result}.csv")
#     df_combined = pd.concat(df_dict.values())  # e.g. electricity prices in the entire year
#     return df_combined

def hourly_int_index_to_datetime(df: pd.DataFrame, year0: int):
    '''
    This function takes a dataframe with index T being the hours from 1-8760 in a year
    and returns the dataframe with datetime index instead.
    '''
    start_date = f"01-01-{year0}"
    dates = pd.date_range(start=start_date, periods=8760, freq='h')

    df_dt = df.copy()  # Avoid changing the original df
    df_dt.index = df.index-1  # Hours are from 1-8760 but we need 0-indexing
    df_dt = df_dt.set_index(dates[df_dt.index])  # Set datetime corresponding to the hours as new index
    return df_dt

def load_plot_config(palette):
    # Set palette for easier and consistent plotting
    sns.set_palette(palette)  # seaborn
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)  # matplotlib.pyplot

#### OBSOLETE FUNCTIONS - DO NOT CONSIDER ####

# Function to extract results
def extract_results(model):
    
    # Zonal electricity prices   
    el_price = pd.DataFrame(data= np.array([model.constraints.powerbalance[i].Pi for i in range(model.data.T)]),
                            index = model.data.times, columns = model.data.el_zones)
    
    # Onshore wind production
    windon_prod = pd.DataFrame(model.variables.windon_prod.x.T, 
                               index = model.data.times, columns=model.data.el_zones)
    
    # Offshore wind production
    windof_prod = pd.DataFrame(model.variables.windof_prod.x.T, 
                               index = model.data.times, columns=model.data.el_zones)
    
    # Solar PV production
    solarpv_prod = pd.DataFrame(model.variables.solarpv_prod.x.T, 
                                index = model.data.times, columns=model.data.el_zones)
    
    # Run-of-river production
    runofriver_prod = pd.DataFrame(model.variables.runofriver_prod.x.T, 
                                   index = model.data.times, columns=model.data.el_zones)
    
    # Reservoir hydro production
    reshydro_prod = pd.DataFrame(model.variables.reshydro_prod.x.T, 
                                 index = model.data.times, columns=model.data.reshydro_generators)   
    
    # Pumped hydro production - DISCHARGIN
    pumphydro_dis_prod = pd.DataFrame(model.variables.pumphydro_dis_prod.x.T, 
                                 index = model.data.times, columns=model.data.pumphydro_generators)   
    
    # Pumped hydro consumption - CHARGING
    pumphydro_cha_demand = pd.DataFrame(model.variables.pumphydro_cha_demand.x.T, 
                                 index = model.data.times, columns=model.data.pumphydro_generators)   

    # Pumped hydro reservoir level - SOC
    pumphydro_SOC = pd.DataFrame(model.variables.pumphydro_SOC.x.T, 
                                 index = model.data.times, columns=model.data.pumphydro_generators)   

    # Generators production
    generators_prod = pd.DataFrame(model.variables.generators_prod.x.T, 
                                   index = model.data.times, columns=model.data.generators)

    # Electricity export per zone
    export = pd.DataFrame(model.variables.export.x.T, 
                          index = model.data.times, columns = model.data.el_zones)
    
    # Electricity lineflows between zones
    lineflow = pd.DataFrame(model.variables.lineflow.x.T, 
                            index = model.data.times, columns = model.data.lines)
    
    # Classic demand
    classic_dem = pd.DataFrame(model.variables.classic_dem.x.T, 
                               index = model.data.times, columns=model.data.el_zones)
    
    
    # Electric heating
    electric_heating = pd.DataFrame(model.variables.ElHeat_dem.x.T, 
                               index = model.data.times, columns=model.data.heating_units)
    # EV baseload
    EV_baseload = pd.DataFrame(model.variables.EV_baseload_dem.x.T, 
                               index = model.data.times, columns=model.data.el_zones)
    # EV flexload
    EV_flexible_dem =  pd.DataFrame(model.variables.EV_flexible_dem.x.T, 
                               index = model.data.times, columns=model.data.el_zones)
    
    # Total EV demand
    
    EV_total_demand = EV_baseload + EV_flexible_dem
    # PtX 
    PtX_dem =  pd.DataFrame(model.variables.PtX_dem.x.T, 
                               index = model.data.times, columns=model.data.PtX_units)
    
    # Reservoir hydro production by zone
    reshydro_prod_T = reshydro_prod.T
    reshydro_prod_T['zone_el'] = list(model.data.reshydro_generators_df.zone_el)
    old_cols = reshydro_prod_T.columns.tolist()
    new_cols = old_cols[-2:] + old_cols[:-2]
    reshydro_prod_T = reshydro_prod_T[new_cols]
    reshydro_prod_byzone = pd.pivot_table(data=reshydro_prod_T, 
                                          index=['zone_el'],aggfunc='sum').T
      
    return(el_price, 
           windon_prod, 
           windof_prod, 
           solarpv_prod,
           runofriver_prod, 
           reshydro_prod,
           generators_prod,  
           export,
           lineflow,           
           classic_dem,
           reshydro_prod_byzone,
           pumphydro_dis_prod,
           pumphydro_cha_demand,
           pumphydro_SOC,
           electric_heating,
           EV_baseload,
           EV_flexible_dem,
           EV_total_demand,
           PtX_dem)

# Function to extract annual energy balance
def energy_balance(model, windon_prod_df, windof_prod_df, solarpv_prod_df,
                   runofriver_prod_df, reshydro_prod_byzone_df, export_df,
                   classic_dem_df, generators_prod_df, pumphydro_dis_prod_df,
                   pumphydro_cha_demand_df, electric_heating_df,
                   EV_baseload_df, EV_flexible_dem_df, PtX_dem_df, inputpath):
    
    # Create dataframe
    annual_balance_df = pd.DataFrame(index=model.data.el_zones)
    
    # Add conventional generation columns
    annual_generators_prod_df = pd.DataFrame(data=generators_prod_df.sum(), columns=['annual_prod'])
    annual_generators_prod_df['fuel'] = model.data.generators_df['fuel']
    annual_generators_prod_df['zone_el'] = model.data.generators_df['zone_el']
    annual_generators_df = pd.pivot_table(annual_generators_prod_df, values = 'annual_prod',index='zone_el', columns='fuel', aggfunc=np.sum)
    annual_balance_df = annual_balance_df.merge(annual_generators_df,how='outer',left_index=True, right_index=True)
    annual_balance_df = annual_balance_df.fillna(0)
    
    # Pumphydro_dis_prod_df
    annual_pumphydro_dis_prod_df = pd.DataFrame(data=pumphydro_dis_prod_df.sum(), columns=['annual_pumphydro_dis_prod'])
    annual_pumphydro_dis_prod_df['zone_el'] = model.data.pumphydro_generators_df['zone_el']
    annual_pumphydro_dis_prod_df = pd.pivot_table(annual_pumphydro_dis_prod_df, values='annual_pumphydro_dis_prod', index='zone_el', aggfunc=np.sum)
    annual_balance_df = annual_balance_df.merge(annual_pumphydro_dis_prod_df, how='outer', left_index=True, right_index=True)
    annual_balance_df = annual_balance_df.fillna(0)
    
    # Add renewable generation
    annual_balance_df['Wind onshore'] = windon_prod_df.sum(axis=0).T
    annual_balance_df['Wind offshore'] = windof_prod_df.sum(axis=0).T
    annual_balance_df['Solar PV'] = solarpv_prod_df.sum(axis=0).T
    annual_balance_df['Run-of-river'] = runofriver_prod_df.sum(axis=0).T
    annual_balance_df['Reservoir Hydro'] = reshydro_prod_byzone_df.sum(axis=0).T
    annual_balance_df = annual_balance_df.fillna(0)
    
    # Sum all generation
    annual_balance_df['Total generation'] = annual_balance_df.sum(axis=1)
    
    # Add net exports
    annual_balance_df['Net export'] = export_df.sum(axis=0).T
    
    # Add classic demand
    annual_balance_df['Classic demand'] = classic_dem_df.sum(axis=0).T
    
    # pumphydro_cha_demand_df
    annual_pumphydro_cha_demand_df = pd.DataFrame(data=pumphydro_cha_demand_df.sum(), columns=['annual_pumphydro_cha_demand'])
    annual_pumphydro_cha_demand_df['zone_el'] = model.data.pumphydro_generators_df['zone_el']
    annual_pumphydro_cha_demand_df = pd.pivot_table(annual_pumphydro_cha_demand_df, values='annual_pumphydro_cha_demand', index='zone_el', aggfunc=np.sum)
    annual_balance_df = annual_balance_df.merge(annual_pumphydro_cha_demand_df, how='outer', left_index=True, right_index=True)
    annual_balance_df = annual_balance_df.fillna(0)

    # Add EV baseload and flexload
    annual_balance_df['EV baseload demand'] = EV_baseload_df.sum(axis=0).T
    
    # Add EV flexload and flexload
    annual_balance_df['EV flexload demand'] = EV_flexible_dem_df.sum(axis=0).T
    
    # Add electric heating
    annual_ElHeat_dem_df = pd.DataFrame(data=electric_heating_df.sum(), columns=['annual_elheat'])
    annual_ElHeat_dem_df = annual_ElHeat_dem_df.reset_index()
    annual_ElHeat_dem_df['zone_el'] = annual_ElHeat_dem_df['index'].apply(lambda x: model.data.heating_demand_df.loc[x, 'zone_el'])
    annual_ElHeat_df = pd.pivot_table(annual_ElHeat_dem_df, values='annual_elheat', index='zone_el', aggfunc=np.sum)
    annual_balance_df = annual_balance_df.merge(annual_ElHeat_df, how='outer', left_index=True, right_index=True)
    annual_balance_df = annual_balance_df.fillna(0)
    
    # PtX_dem_df
    annual_PtX_dem_df = pd.DataFrame(data=PtX_dem_df.sum(), columns=['annual_PtX_dem'])
    #annual_PtX_dem_df = annual_PtX_dem_df.reset_index()
    annual_PtX_dem_df['zone_el'] = model.data.PtX_demand_df['zone_el']
    annual_PtX_dem_df = pd.pivot_table(annual_PtX_dem_df, values='annual_PtX_dem', index='zone_el', aggfunc=np.sum)
    annual_balance_df = annual_balance_df.merge(annual_PtX_dem_df,how='outer',left_index=True, right_index=True)
    annual_balance_df = annual_balance_df.fillna(0)
    
    # Calculate the energy balance as generation - exports - demand 
    annual_balance_df['Balance'] = (annual_balance_df['Total generation'] - annual_balance_df['Net export'] 
                                    - annual_balance_df['Classic demand'] -annual_balance_df['EV baseload demand'] -annual_balance_df['EV flexload demand']
                                    -annual_balance_df['annual_PtX_dem'] - annual_balance_df['annual_elheat'] - annual_balance_df['annual_pumphydro_cha_demand'])
    
    # Add renewables curtailment
    annual_balance_df['Wind onshore curtailed'] = pd.read_csv(inputpath + '00. Onshore wind.csv',index_col=0).sum(axis=0) - annual_balance_df['Wind onshore']
    annual_balance_df['Wind offshore curtailed'] = pd.read_csv(inputpath + '01. Offshore wind.csv',index_col=0).sum(axis=0) - annual_balance_df['Wind offshore']
    annual_balance_df['Solar PV curtailed'] = pd.read_csv(inputpath + '02. Solar PV.csv',index_col=0).sum(axis=0) - annual_balance_df['Solar PV']
    annual_balance_df['Run-of-river curtailed'] = pd.read_csv(inputpath + '03. Run-of-river.csv',index_col=0).sum(axis=0) - annual_balance_df['Run-of-river']
    
    # Add demand loadshed
    annual_balance_df['Demand loadshed'] = pd.read_csv(inputpath + '08. Classic demand.csv',index_col=0).sum(axis=0) - annual_balance_df['Classic demand']
    
    # Convert to TWh and round to 1 decimal
    annual_balance_df_TWh = (annual_balance_df.fillna(0) / 1000000).round(1)
    annual_balance_df_GWh = (annual_balance_df.fillna(0) / 1000).round(1)
    
    return(annual_balance_df_TWh, annual_balance_df_GWh)

# Function to extract hourly generation dispatch by market and fuel
def generation_dispatch(model, generators_prod_df):
    
    # Create dataframe from generators individual production aggregating units generation by fuel and market
    generators_prod_df_T = generators_prod_df.T
    #generators_prod_df_T = generators_prod_df_T.reset_index()
    generators_prod_df_T['fuel'] = model.data.generators_df['fuel']
    generators_prod_df_T['zone_el'] = model.data.generators_df['zone_el']
    generators_prod_df_T = generators_prod_df_T.groupby(['zone_el','fuel']).sum()
    generators_prod_byfuel_df = generators_prod_df_T.T
    
    return(generators_prod_byfuel_df)

def PtX_dispatch(model, PtX_dem_df):
    
    # Create dataframe from generators individual production aggregating units generation by fuel and market
    PtX_dem_df_T = PtX_dem_df.T
    #generators_prod_df_T = generators_prod_df_T.reset_index()

    PtX_dem_df_T['zone_el'] = model.data.PtX_demand_df['zone_el']
    PtX_dem_df_T = PtX_dem_df_T.groupby('zone_el').sum()
    PtX_dem_df_byzone_df = PtX_dem_df_T.T
    
    return(PtX_dem_df_byzone_df)



# Function to extract baseload and technology capture prices
def capture_prices(model, el_price_df, windon_prod_df, windof_prod_df, 
                   solarpv_prod_df, generators_prod_byfuel_df):
    
    # Create dataframe for all technology capture prices (zones as index, technologies as columns)
    captureprices_df = pd.DataFrame(index = model.data.el_zones, 
                                    columns = ['Baseload', 'Wind onshore', 'Wind offshore',
                                               'Solar PV', 'Coal', 'Gas', 'Nuclear'])
    
    # Fill baseload and renewable capture price columns
    captureprices_df.loc[:,'Baseload'] = el_price_df.mean()
    captureprices_df.loc[:,'Wind onshore'] = (el_price_df * windon_prod_df).sum() / windon_prod_df.sum()
    captureprices_df.loc[:,'Wind offshore'] = (el_price_df * windof_prod_df).sum() / windof_prod_df.sum()
    captureprices_df.loc[:,'Solar PV'] = (el_price_df * solarpv_prod_df).sum() / solarpv_prod_df.sum()
    
    # Fill non renewable capture prices
    generators_prod_byfuel_df_T = generators_prod_byfuel_df.T
    coal_prod_df = generators_prod_byfuel_df_T[generators_prod_byfuel_df_T.index.isin(['Coal', 'Brown coal'], level=1)].groupby(level=0).sum().T
    captureprices_df.loc[:,'Coal'] = (el_price_df * coal_prod_df).sum() / coal_prod_df.sum()
    
    gas_prod_df = generators_prod_byfuel_df_T[generators_prod_byfuel_df_T.index.isin(['Natural gas'], level=1)].groupby(level=0).sum().T
    captureprices_df.loc[:,'Gas'] = (el_price_df * gas_prod_df).sum() / gas_prod_df.sum()
    
    nuclear_prod_df = generators_prod_byfuel_df_T[generators_prod_byfuel_df_T.index.isin(['Nuclear'], level=1)].groupby(level=0).sum().T
    captureprices_df.loc[:,'Nuclear'] = (el_price_df * nuclear_prod_df).sum() / nuclear_prod_df.sum()
    
    return(captureprices_df)