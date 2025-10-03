"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
#%%
from enlight.runner import EnlightRunner  # Updated import path
from pathlib import Path

if __name__ == "__main__":
    # Create an instance of the EnlightRunner
    runner = EnlightRunner()
    runner.visualize_data(scenario_name='scenario_1',
                          week=25,
                          simulation_path=Path('simulations/scenario_1'))
    #%%
    # Prepare input data for simulations based on the configuration in scenarios_config.yaml
    runner.prepare_data_single_simulation('scenario_1')
    # debugger = runner.data_processor
    
    # Prepare input data for simulations based on the configuration in scenarios_config.yaml
    # runner.prepare_data_all_simulations()

    # Run a single simulation for the specified week and save the results
    runner.run_single_simulation(week=25, simulation_path=Path('simulations/scenario_1'))
    # runner.enlight_model.run_model()
    # Access the model instance for debugging purposes
    d = runner.enlight_model
    # %%
    runner.visualizer.plot_aggregated_curves_with_zonal_prices(
        example_hour=12,
        bids_accepted=d.demand_inflexible_classic_bid.solution,
        zonal_prices=d.results_dict['electricity_prices']
    )

# %%
