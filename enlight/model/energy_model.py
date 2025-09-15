from pathlib import Path
import numpy as np
import pandas as pd
import linopy
import xarray as xr

from enlight.data_ops import DataLoader
import enlight.utils as utils

class EnlightModel:
    """
    Electricity market optimization model using Linopy.

    Attributes:
        T (int): Number of time steps.
        Z (int): Number of zones.
        G (int): Number of conventional_units.
        L (int): Number of transmission lines.
    """
    
    def __init__(self, week, simulation_path, logger):
        
        
        # Initialize logger
        self.logger = logger
        self.logger.info(
            "INITIALIZING ENLIGHT MODEL"
        )
        
        self.simulation_path = simulation_path
        self.data = DataLoader(week=week, 
                               input_path=Path(self.simulation_path) / 'data',
                               logger = self.logger)
        
        self.model = linopy.Model()
        
        self._aux_data()
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        
    def _aux_data(self):
        """
        Extract and define core sets and their lengths from input data.
        """
        self.time_index = pd.Index(np.arange(168), name="T")
        self.times = list(self.data.demand_inflexible_classic.index)        # Shape: (T,)
        self.bidding_zones = self.data.bidding_zones               # Shape: (Z,)

        

        self.T = len(self.times)     # Number of time steps
        self.Z = len(self.bidding_zones)  # Number of zones
        self.G = len(self.data.conventional_units_id) 

        
    def _build_variables(self):
        """
        Declare model variables.
        
        Note: Unused variables will be excluded from the model unless referenced
                in the objective or constraints.
        """
        
        # Onshore wind production [MW]
        # Shape: (T, Z)
        self.wind_onshore_bid = self.model.add_variables(
            lower=0,
            upper=self.data.wind_onshore_production.values,   # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='wind_onshore_bid'
        )
        
        # Offshore wind production [MW]
        # Shape: (T, Z)
        self.wind_offshore_bid = self.model.add_variables(
            lower=0,
            upper=self.data.wind_offshore_production.values,   # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='wind_offshore_bid'
        )
        
        # Solar PV production [MW]
        # Shape: (T, Z)
        self.solar_pv_bid = self.model.add_variables(
            lower=0,
            upper=self.data.solar_pv_production.values,   # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='solar_pv_bid'
        )
        
        # Hydro ROR production [MW]
        # Shape: (T, Z)
        self.hydro_ror_bid = self.model.add_variables(
            lower=0,
            upper=self.data.hydro_ror_production.values,   # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='hydro_ror_bid'
        )

        # Classic demand [MW]
        # Shape: (T, Z)
        self.demand_inflexible_classic_bid = self.model.add_variables(
            lower=0,
            upper=self.data.demand_inflexible_classic.values,  # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='demand_inflexible_classic_bid'
        )
        
        ## Thermal generation production variable (shape: T x G)
        ## upper bound = generator capacity repeated for all time steps
        ## np.outer(np.ones(T), capacities) produces a (T, G) matrix
        self.conventional_units_bid = self.model.add_variables(
            lower=0,
            upper=self.data.conventional_units_el_cap,
            coords=[self.times, self.data.conventional_units_id],
            dims=["T", "G"],
            name='conventional_units_bid'
        )

        # Electricity export
        self.electricity_export = self.model.add_variables(
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name = 'export'
        )
        
        self.lineflow = self.model.add_variables(
            lower = -self.data.lines_b_to_a_cap,
            upper = self.data.lines_a_to_b_cap,
            coords = [self.times, self.data.line_labels],
            dims=["T", "L"],
            name='lineflow'
        )

    def _build_constraints(self):
        """
        Placeholder for adding model constraints.
        """
        
        self.power_balance = self.model.add_constraints(
            (self.wind_onshore_bid
             + self.wind_offshore_bid
             + self.solar_pv_bid
             + self.hydro_ror_bid
             + self.conventional_units_bid.dot(self.data.G_Z_xr) # type: ignore
             == 
             self.demand_inflexible_classic_bid
            + self.electricity_export
            ),
            name='power_balance'
            )
        
        self.electricity_exports = self.model.add_constraints(
            (self.lineflow.dot(self.data.L_Z_xr) == self.electricity_export), # type: ignore
            name='electricity_exports'
            )
     
    def _build_objective(self):
        """
        Define the objective function for profit maximization.
        """
        self.model.add_objective(
            expr = (
                - self.demand_inflexible_classic_bid * self.data.voll_classic
                + self.wind_onshore_bid * self.data.wind_onshore_bid_price
                + self.wind_offshore_bid * self.data.wind_offshore_bid_price
                + self.solar_pv_bid * self.data.solar_pv_bid_price
                + self.hydro_ror_bid * self.data.hydro_ror_bid_price
               ).sum()
           
            # Important: variables with different dimensions must be in different parenthesis to be summed correctly
            + (self.conventional_units_bid * (self.data.conventional_units_marginal_cost_df)).sum(),
            sense="min"
        )

        print('obj added')

    def solve_model(self, solver_name='gurobi'):
        """
        Solve the model using the specified solver.
        """
        self.logger.info("Start solving model")
        self.model.solve(solver_name=solver_name)
        self.logger.info('Model solved, good job champ!')

    def save_model_to_lp_file(self):
        """
        Export model to .lp file.
        """
        self.logger.info('Saving the .lp model file')
        Path('results').mkdir(parents=True, exist_ok=True)
        self.model.to_file(Path(self.simulation_path) / 'results' / 'debug_model.lp', io_api='lp', explicit_coordinate_names=True)
        self.logger.info('Saved .lp model file')


    def run_model(self):
        """
        Solve the model using Gurobi.
        """
        self.solve_model(solver_name='gurobi')
        utils.save_model_results(self)
        # self.save_model_to_lp_file()
