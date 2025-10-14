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

    h=133  # needed for .visualize_data() and .visualize_results()

    '''Combination of methods to VISUALIZE INPUT data:'''
    # Creates instance of the DataProcessor:
    runner.prepare_data_single_scenario('scenario_1')
    # Creates instance of the DataLoader:
    runner.load_data_single_simulation(simulation_path=Path('simulations/scenario_1'))
    # Creates instance of the DataVisualizer. Data has to be prepared when running this:
    runner.visualize_data(example_hour=h)

    '''Combination of methods to RUN a SINGLE simulation
    and SHOW RESULTS for that simulation:'''
    # Creates instance of the EnlightModel
    runner.run_single_simulation(simulation_path=Path('simulations/scenario_1'))
    # Creates instance of the ResultsVisualizer.
    runner.visualize_results(example_hour=h)

    # Verify social welfare calculations
    print(f"{runner.enlight_model.results_econ['social welfare']/1e9:.6f} b.€")
    print(f"{runner.enlight_model.results_econ['social welfare perceived']/1e9:.6f} b.€")
    print(f"{runner.enlight_model.model.objective.value/1e9:.6f} b.€")
