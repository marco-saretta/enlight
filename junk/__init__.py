
# Standard modules
import pandas as pd
import os
from tqdm import tqdm
# External functions
import energy_system_model
from extract_results import *

# Define scenarios and years to be simulated
simulation_scen = [#'WY2015','WY2016','WY2017','WY2018','WY2019',
                   'WY2020',
                   #'WY2021',
                   #'WY2022'
                   ]

simulation_years = [2025,
                    2030,
                    2035,
                    2040,
                    2045,
                    2050
                    ]

# Define paths for input data and output results for all simulations
scenarios = os.listdir('00. Input')
scen_years = {}
inputpaths = []
outputpaths = []
i = 0

                
for s in simulation_years:  # Only include scenarios file for the years that are simulating
    if str(s) in scenarios:
        # Dictionary with scenarios available for each year
        scen_years[str(s)] = os.listdir('00. Input/' + str(s))
        # Create output scenario folder if it does not exist
        os.makedirs('01. Output/' + str(s), exist_ok=True)
        for y in scen_years[str(s)]:
            if y in simulation_scen:
                # Define input and output paths
                inputpaths.append('00. Input/' + str(s) + '/' + y + '/')
                outputpaths.append('01. Output/' + str(s) + '/' + y + '/')
                # Create output year folder if it does not exist
                os.makedirs('01. Output/' + str(s) + '/' + y, exist_ok=True)
# Simulations
simulations = range(len(inputpaths))

# Weeks
weeks = range(1,53)
   

for s in simulations:
    # Weekly model run
    for w in tqdm(weeks):
        
        # Create model instance and execute the model
        model = energy_system_model.model(w, inputpaths[s])
        model._run_model()
        
        # Extract model results
        (el_price, 
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
         PtX_dem ) = extract_results(model)
        
        if w == 1: 
            # Add model results to dataframes
            el_price_df = el_price
            windon_prod_df = windon_prod
            windof_prod_df = windof_prod
            solarpv_prod_df = solarpv_prod
            runofriver_prod_df = runofriver_prod
            reshydro_prod_df = reshydro_prod       
            generators_prod_df = generators_prod
            export_df = export
            lineflow_df = lineflow
            classic_dem_df = classic_dem
            reshydro_prod_byzone_df = reshydro_prod_byzone
            pumphydro_dis_prod_df = pumphydro_dis_prod
            pumphydro_cha_demand_df = pumphydro_cha_demand
            pumphydro_SOC_df = pumphydro_SOC
            electric_heating_df = electric_heating 
            EV_baseload_df = EV_baseload 
            EV_flexible_dem_df = EV_flexible_dem
            EV_total_demand_df = EV_total_demand
            PtX_dem_df = PtX_dem
            
        else:
            # Add model results to dataframes
            el_price_df = pd.concat([el_price_df, el_price])
            generators_prod_df = pd.concat([generators_prod_df, generators_prod])
            windon_prod_df = pd.concat([windon_prod_df, windon_prod])
            windof_prod_df = pd.concat([windof_prod_df, windof_prod])
            solarpv_prod_df = pd.concat([solarpv_prod_df, solarpv_prod])
            runofriver_prod_df = pd.concat([runofriver_prod_df, runofriver_prod])
            reshydro_prod_df = pd.concat([reshydro_prod_df, reshydro_prod])
            export_df = pd.concat([export_df, export])
            lineflow_df = pd.concat([lineflow_df, lineflow])
            classic_dem_df = pd.concat([classic_dem_df, classic_dem])
            reshydro_prod_byzone_df = pd.concat([reshydro_prod_byzone_df, reshydro_prod_byzone])
            pumphydro_dis_prod_df = pd.concat([pumphydro_dis_prod_df, pumphydro_dis_prod])
            pumphydro_cha_demand_df = pd.concat([pumphydro_cha_demand_df, pumphydro_cha_demand])
            pumphydro_SOC_df = pd.concat([pumphydro_SOC_df, pumphydro_SOC])
            electric_heating_df = pd.concat([electric_heating_df, electric_heating]) 
            EV_baseload_df = pd.concat([EV_baseload_df, EV_baseload])  
            EV_flexible_dem_df = pd.concat([EV_flexible_dem_df, EV_flexible_dem]) 
            EV_total_demand_df = pd.concat([EV_total_demand_df, EV_total_demand]) 
            PtX_dem_df = pd.concat([PtX_dem_df, PtX_dem])  
    
    # Electricity price
    el_price_df.to_csv(outputpaths[s] + '00. Electricity prices.csv')
    
    # Onshore wind
    windon_prod_df.to_csv(outputpaths[s] + '01. Onshore wind production.csv')
    
    # Offshore wind
    windof_prod_df.to_csv(outputpaths[s] + '02. Offshore wind production.csv')
    
    # Solar PV
    solarpv_prod_df.to_csv(outputpaths[s] + '03. Solar PV production.csv')
    
    # Run-of-river
    runofriver_prod_df.to_csv(outputpaths[s] + '04. Run-of-river hydro production.csv')
    
    # Reservoir hydro
    reshydro_prod_df.to_csv(outputpaths[s] + '05. Reservoir hydro production.csv')
    
    # Generators
    generators_prod_df.to_csv(outputpaths[s] + '06. Generators production.csv')
    
    # Classic demand
    classic_dem_df.to_csv(outputpaths[s] + '07. Classic electricity demand.csv')
    
    # Electricity export
    export_df.to_csv(outputpaths[s] + '08. Electricity export.csv')
    
    # Line flows
    lineflow_df.to_csv(outputpaths[s] + '09. Lineflows.csv')
    
    # Reservoir hydro production per zone
    reshydro_prod_byzone_df.to_csv(outputpaths[s] + '10. Reservoir hydro production per zone.csv')
    
    # Annual energy balance
    annual_balance_df_TWh, annual_balance_df_GWh = energy_balance(model, windon_prod_df, windof_prod_df, 
                                       solarpv_prod_df,runofriver_prod_df,
                                       reshydro_prod_byzone_df, export_df,
                                       classic_dem_df, generators_prod_df, 
                                       pumphydro_dis_prod_df, pumphydro_cha_demand_df,
                                       electric_heating_df, EV_baseload_df,
                                       EV_flexible_dem_df, PtX_dem_df,
                                       inputpaths[s])
    
    annual_balance_df_TWh.to_csv(outputpaths[s] + '11. Annual energy balance TWh.csv')
    annual_balance_df_GWh.to_csv(outputpaths[s] + '11. Annual energy balance GWh.csv')
    
    # Generators production aggregated by fuel and market
    generators_prod_byfuel_df = generation_dispatch(model, generators_prod_df)
    generators_prod_byfuel_df.to_csv(outputpaths[s] + '12. Generators production by fuel.csv')
    
    # Baseload and capture prices
    captureprices_df = capture_prices(model, el_price_df, windon_prod_df, windof_prod_df, 
                                      solarpv_prod_df, generators_prod_byfuel_df)
    captureprices_df.to_csv(outputpaths[s] + '13. Capture prices.csv')
    # Pumped hydro
    pumphydro_dis_prod_df.to_csv(outputpaths[s] + '14. Pumped hydro production.csv')
    pumphydro_cha_demand_df.to_csv(outputpaths[s] + '15. Pumped hydro consumption.csv')
    pumphydro_SOC_df.to_csv(outputpaths[s] + '16. Pumped hydro SOC.csv')
    
    # Electric heating
    electric_heating_df.to_csv(outputpaths[s] + '17. Electric heating.csv')
    
    # EVs demand
    EV_baseload_df.to_csv(outputpaths[s] + '18. EV baseload.csv')
    EV_flexible_dem_df.to_csv(outputpaths[s] + '19. EV flexload.csv')
    EV_total_demand_df.to_csv(outputpaths[s] + '20. EV total demand.csv')
    # PtX plants
    PtX_dem_df.to_csv(outputpaths[s] + '21. PtX demand.csv')
    
    PtX_dem_df_byzone_df = PtX_dispatch(model, PtX_dem_df)
    PtX_dem_df_byzone_df.to_csv(outputpaths[s] + '22. PtX demand by zone.csv')
    

    
