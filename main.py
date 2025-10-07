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

    # Prepare input data for simulations based on the configuration in scenarios_config.yaml
    # Creates instance of DataProcessor:
    runner.prepare_data_single_simulation('scenario_1')
    # debugger = runner.data_processor

    w=1
    # Creates instance of DataLoader:
    runner.load_data_single_simulation(week=w, simulation_path=Path('simulations/scenario_1'))

    h=133
    # Creates instance of DataVisualizer. Data has to be prepared when running this:
    runner.visualize_data(week=w, example_hour=h)
    
    # Prepare input data for simulations based on the configuration in scenarios_config.yaml
    # runner.prepare_data_all_simulations()

    # Run a single simulation for the specified week and save the results
    runner.run_single_simulation(simulation_path=Path('simulations/scenario_1'))
    # runner.enlight_model.run_model()
    # Access the model instance for debugging purposes
    d = runner.enlight_model

    runner.visualize_results(example_hour=h)

    # Verify social welfare calculations
    print(f"{d.results_econ['social welfare']/1e9:.6f} b.€")
    print(f"{d.results_econ['social welfare perceived']/1e9:.6f} b.€")
    print(f"{d.model.objective.value/1e9:.6f} b.€")
# %%
