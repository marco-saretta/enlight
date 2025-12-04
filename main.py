"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
from enlight.runner import EnlightRunner  # Updated import path
from pathlib import Path

if __name__ == "__main__":
    # Get path of project root
    root_path: Path = Path(__file__).parent.resolve()    
    
    # Create an instance of the EnlightRunner
    r = EnlightRunner(root_path=root_path)
    
    # Creates instance of the DataProcessor:
    r.run_scenario('scenario_1', dry_run = False)
    
    # just a debug
    m1 = r.enlight_model.model
    
    # Creates instance of the DataProcessor:
    r.run_scenario('scenario_2', dry_run = False)
    
    # just a debug
    m2 = r.enlight_model.model
    print(m2.constraints['hydro_res_energy_availability_1'])
    # m.variables
    # m.constraints
    # m.constraints['power_balance']
    
    
    #r.run_scenario('scenario_2')
    