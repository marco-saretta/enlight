import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import xarray as xr

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

def save_model_results(self):
    """
    Calculate and save results from the optimized model.
    """
    results_path = Path(self.simulation_path) / "results"
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
            "demand_inflexible_classic_sol": get_solution(self.demand_inflexible_classic_bid),
            "wind_onshore_bid_sol": get_solution(self.wind_onshore_bid),
            "wind_offshore_bid_sol": get_solution(self.wind_offshore_bid),
            "solar_pv_bid_sol": get_solution(self.solar_pv_bid),
            "hydro_ror_bid_sol": get_solution(self.hydro_ror_bid),
            "conventional_units_sol": get_solution(self.conventional_units_bid),
            "electricity_export_sol": get_solution(self.electricity_export),
            "lineflow_sol": get_solution(self.lineflow),
        })

        # Derived results
        self.results_dict["demand_inflexible_classic_curt"] = (
            self.data.demand_inflexible_classic - self.results_dict["demand_inflexible_classic_sol"]
        )
        self.results_dict["wind_on_bid_vol_curt"] = (
            self.data.wind_onshore_production - self.results_dict["wind_onshore_bid_sol"]
        )
        self.results_dict["electricity_prices"] = self.model.dual["power_balance"].to_pandas()

        # # ###### NEWLY ADDED - MARGINAL GENERATOR ######
        # # # Read out the marginal generator by its unit name
        el_cap_xr = xr.DataArray(self.data.conventional_units_el_cap,
                                    dims=["T", "G"],
                                    coords=[self.times, self.data.conventional_units_id])

        marginal_thermal_generators = np.empty(len(self.times)+1, dtype=pd.DataFrame)
        for h, h_idx in zip(self.times, self.data.time_index):
            marginal_thermal_generators_prod = self.conventional_units_bid.solution.sel(T=h)[
                (self.conventional_units_bid.solution.sel(T=h) > 0)  # is the generator activated?
                & (self.conventional_units_bid.solution.sel(T=h) < el_cap_xr.sel(T=h))  # is it at full capacity?
                ]
            marginal_thermal_generators[h_idx] = self.data.conventional_units_marginal_cost_df.loc[
                h, marginal_thermal_generators_prod.G
                ]
        
        el_cap_hydro_xr = xr.DataArray(self.data.hydro_res_units_el_cap,
                                    dims=["T", "G_hydro_res"],
                                    coords=[self.times, self.data.hydro_res_units_id])

        # # Read out the marginal hydro reservoir generator by its unit name
        marginal_hydro_generators = np.empty(len(self.data.time_index)+1, dtype=pd.DataFrame)
        for h, h_idx in zip(self.times, self.data.time_index):
            marginal_hydro_generators_prod = self.hydro_res_units_bid.solution.sel(T=h)[
                (self.hydro_res_units_bid.solution.sel(T=h) > 0) # is the generator activated?
                & (self.hydro_res_units_bid.solution.sel(T=h) < el_cap_hydro_xr.sel(T=h)) # is it at full capacity?
                ]
            marginal_hydro_generators[h_idx] = self.data.hydro_res_units_marginal_cost_df.loc[
                h, marginal_hydro_generators_prod.G_hydro_res
                ]

        marginal_generators_df = pd.DataFrame({
            'thermal_generator_id': [tuple(marginal_thermal_generator.index) for marginal_thermal_generator in marginal_thermal_generators[1:]],
            'thermal_generator_cost': [tuple(marginal_thermal_generator.values) for marginal_thermal_generator in marginal_thermal_generators[1:]],
            'hydro_res_generator_id': [tuple(marginal_hydro_generator.index) for marginal_hydro_generator in marginal_hydro_generators[1:]],
            'hydro_res_generator_cost': [tuple(marginal_hydro_generator.values) for marginal_hydro_generator in marginal_hydro_generators[1:]]})
        marginal_generators_df.index = self.data.times

        self.results_dict["marginal_generator"] = marginal_generators_df

        # The code below is used to explain why the marginal hydro reservoir is a given hour
        #   is NOT equivalent to the electricity price in that hour. It is due to the temporal coupling
        #   of the hydro reservoir energy availability constraint.
        hydro_reservoir_energy_utilized = (self.hydro_res_units_bid.solution.sum(axis=0)
                                           / self.data.hydro_res_units_energy_availability)
        tol = 1e-5  # used to avoid floating point precision issues
        self.results_dict["marginal_energy_availability_cost"] = (
            self.data.hydro_res_units.loc[
                hydro_reservoir_energy_utilized[  # units between 0 and 1 have not fully depleted their reservoir energy for that week
                    (hydro_reservoir_energy_utilized > 0)
                    & (hydro_reservoir_energy_utilized < 1-tol)
                ].G_hydro_res.values  # get the unit IDs in order to select the undepleted reservoirs from the hydro reservoir dataframe
            ]
            .groupby("zone_el")  # group by bidding zone since congestion might occur
            .prodcost.min()  # the cheapest undepleted hydro reservoir unit sets the price
        )
        ################################################

        # Save all results ---
        for name, df in self.results_dict.items():
            if df is not None:  # only save if calculated
                df.to_csv(results_path / f"{name}.csv")
        self.logger.info('Files saved')
    except Exception as e:
        print(f"Error saving results: {e}")
        # raise e
            
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