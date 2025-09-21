"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
from enlight.runner import EnlightRunner  # Updated import path
from pathlib import Path

if __name__ == "__main__":
    # Create an instance of the EnlightRunner
    runner = EnlightRunner()

    # Prepare input data for simulations based on the configuration in scenarios_config.yaml
    runner.prepare_data_single_simulation('scenario_1')
    # debugger = runner.data_processor
    
    # Prepare input data for simulations based on the configuration in scenarios_config.yaml
    # runner.prepare_data_all_simulations()

    # Run a single simulation for the specified week and save the results
    runner.run_single_simulation(week=1, simulation_path=Path('simulations/scenario_1'))
    # runner.enlight_model.run_model()
    # Access the model instance for debugging purposes
    d = runner.enlight_model

    # The zonal prices may not correspond exactly to production costs of the marginal generator
        # if that generator is a hydro reservoir unit with a binding constraint on energy availability.
        # the dual value of the zonal energy availability constraint reflects this deviation.
    # print(d.hydro_res_energy_availability.dual)