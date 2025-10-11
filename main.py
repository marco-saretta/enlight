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

    w=1  # needed for DataLoader
    h=133  # needed for .visualize_data() and .visualize_results()

    '''Combination of methods to VISUALIZE INPUT data:'''
    # # Creates instance of the DataProcessor:
    # runner.prepare_data_single_scenario('scenario_1')
    # # Creates instance of the DataLoader:
    # runner.load_data_single_simulation(week=w, simulation_path=Path('simulations/scenario_1'))
    # # Creates instance of the DataVisualizer. Data has to be prepared when running this:
    # runner.visualize_data(week=w, example_hour=h)

    '''Combination of methods to RUN a SINGLE simulation
    and SHOW RESULTS for that simulation:'''
    # runner.load_data_single_simulation(week=w, simulation_path=Path('simulations/scenario_1'))
    # # Creates instance of the EnlightModel
    # runner.run_single_simulation(simulation_path=Path('simulations/scenario_1'))
    # d=runner.enlight_model
    # # Creates instance of the ResultsVisualizer.
    # runner.visualize_results(example_hour=h)

    # Verify social welfare calculations
    # print(f"{d.results_econ['social welfare']/1e9:.6f} b.€")
    # print(f"{d.results_econ['social welfare perceived']/1e9:.6f} b.€")
    # print(f"{d.model.objective.value/1e9:.6f} b.€")

    '''Combination of methods to RUN ALL simulations:'''
    # runner.load_data_all_simulations(simulation_path=Path('simulations/scenario_1'))
    # runner.run_all_simulations(simulation_path=Path('simulations/scenario_1'))

    '''Combination of methods to VISUALIZE NBS INPUTS:'''
    # 'week' needed to initialize DataVisualizer because that requires DataLoader...
    runner.prepare_data_single_scenario('scenario_1')  # -> simplest way to load the raw data
    runner.load_data_single_simulation(week=w, simulation_path=Path('simulations/scenario_1'))
    runner.visualize_NBS_data(
        z0='DK2',
        prices_path=Path('simulations/scenario_1/results/yearly_electricity_prices.csv'),
        week=w
    )  # week is irrelevant for the plots currently produced


# %%
