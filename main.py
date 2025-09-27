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
    runner.run_single_simulation(week=25, simulation_path=Path('simulations/scenario_1'))
    # runner.enlight_model.run_model()
    # Access the model instance for debugging purposes
    d = runner.enlight_model


    import pandas as pd
    import numpy as np
    tech_df = pd.read_csv("data/technology_data/technology_data.csv", index_col=0, skiprows=[1])
    print((tech_df['Ramp down'] * 60).min())
    print((tech_df['Ramp up'] * 60).min())
    # Even the least flexible technologies e.g. nuclear can fully ramp up or down within an hour.
    #   Since the DA market is solved in an hourly basis, this makes ramp rates obsolete.
