"""
Main function to execute the Enlight energy scenario runner.

This function creates an instance of the EnlightRunner, prepares input data,
and runs a single simulation.
"""
from enlight.runner import EnlightRunner  # Updated import path
from pathlib import Path

if __name__ == "__main__":
    # Get path of project root
    file_path: Path = Path(__file__).resolve()
    root_path: Path = file_path.parent    
    
    # Create an instance of the EnlightRunner
    r = EnlightRunner(root_path=root_path)
    
    # Creates instance of the DataProcessor:
    r.run_scenario('scenario_1')
    
    r.run_scenario('scenario_2')