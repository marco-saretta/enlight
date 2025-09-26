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
                               logger=self.logger)

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
        self.G_hydro_res = len(self.data.hydro_res_units_id) 


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
      
        # Thermal generation production variable (shape: T x G)
        # upper bound = generator capacity repeated for all time steps
        # np.outer(np.ones(T), capacities) produces a (T, G) matrix
        self.conventional_units_bid = self.model.add_variables(
            lower=0,
            upper=self.data.conventional_units_el_cap,
            coords=[self.times, self.data.conventional_units_id],
            dims=["T", "G"],
            name='conventional_units_bid'
        )

        # Hydro reservoir generator production variable (shape: T x G_hydro_res)
        # upper bound = generator capacity repeated for all time steps
        self.hydro_res_units_bid = self.model.add_variables(
            lower=0,
            upper=self.data.hydro_res_units_el_cap,  # np.array
            coords=[self.times, self.data.hydro_res_units_id],
            dims=["T", "G_hydro_res"],
            name='hydro_res_units_bid'
        )

        # The following three variables are for PUMPED HYDRO STORAGE:
            # Bid / consumption - [MW]
            # Offer / production - [MW]
            # State of charge (SOC) - [MWh]
        # Hydro pumped CONSUMPTION variable (shape: T x G_hydro_ps)
        # upper bound = pumped hydro capacity repeated for all time steps
        self.hydro_ps_units_bid = self.model.add_variables(
            lower=0,
            upper=self.data.hydro_ps_units_el_cap,  # np.array
            coords=[self.times, self.data.hydro_ps_units_id],
            dims=["T", "G_hydro_ps"],
            name='hydro_ps_units_bid'
        )
        self.hydro_ps_units_offer = self.model.add_variables(
            lower=0,
            upper=self.data.hydro_ps_units_el_cap,  # np.array
            coords=[self.times, self.data.hydro_ps_units_id],
            dims=["T", "G_hydro_ps"],
            name='hydro_ps_units_offer'
        )
        self.hydro_ps_units_SOC = self.model.add_variables(
            lower=0,
            upper=self.data.hydro_ps_units_storage_cap,  # np.array
            coords=[self.times, self.data.hydro_ps_units_id],
            dims=["T", "G_hydro_ps"],
            name='hydro_ps_units_SOC'
        )

        # The following three variables are for BESS:
            # Bid / charge - [MW]
            # Offer / discharge - [MW]
            # State of charge (SOC) - [MWh]
        # BESS charge variable (shape: T x G_hydro_ps)
        # upper bound = BESS power capacity repeated for all time steps
        self.bess_units_bid = self.model.add_variables(
            lower=0,
            upper=self.data.bess_units_el_cap,  # np.array
            coords=[self.times, self.data.bess_units_id],
            dims=["T", "G_bess"],
            name='bess_units_bid'
        )
        self.bess_units_offer = self.model.add_variables(
            lower=0,
            upper=self.data.bess_units_el_cap,  # np.array
            coords=[self.times, self.data.bess_units_id],
            dims=["T", "G_bess"],
            name='bess_units_offer'
        )
        self.bess_units_SOC = self.model.add_variables(
            lower=0,
            upper=self.data.bess_units_storage_cap,  # np.array
            coords=[self.times, self.data.bess_units_id],
            dims=["T", "G_bess"],
            name='bess_units_SOC'
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
             + self.conventional_units_bid.dot(self.data.G_Z_xr)  # type: ignore
             + self.hydro_res_units_bid.dot(self.data.G_hydro_res_Z_xr)
             + self.hydro_ps_units_offer.dot(self.data.G_hydro_ps_Z_xr)
             + self.bess_units_offer.dot(self.data.G_bess_Z_xr)
             ==
             self.demand_inflexible_classic_bid
             + self.electricity_export
             + self.hydro_ps_units_bid.dot(self.data.G_hydro_ps_Z_xr)
             + self.bess_units_bid.dot(self.data.G_bess_Z_xr)
             ),
            name='power_balance'
            )

        self.electricity_exports = self.model.add_constraints(
            (self.lineflow.dot(self.data.L_Z_xr) == self.electricity_export),  # type: ignore
            name='electricity_exports'
            )

        self.hydro_res_energy_availability = self.model.add_constraints(
            # If simulating multiple weeks, this constraint HAS to be changed
            # The weekly inflow to hydro reservoirs in each bidding zone is
            #   allocated to all the hydro reservoir units based on their share
            #   of the total hydro reservoir capacity in that zone.
            # APPLY CONSTRAINT WEEKLY
            (self.hydro_res_units_bid.sum(dim='T')
             <= self.data.hydro_res_units_energy_availability),  # Shape: (G_hydro_res,)
            name='hydro_res_energy_availability'
        )

        self.hydro_ps_units_SOC_balance = self.model.add_constraints(  # Shape: (T, G_hydro_ps)
            # In each hour the change in the SOC is equal to the net energy
            #   charged/discharged. In the first hour we add the initial SOC in MWh.
            self.hydro_ps_units_SOC.diff(n=1, dim="T")  # .isel(T=slice(1, None)) is the reason for "UserWarning". It messes with the coordinates.
            - self.data.hydro_ps_initial_SOC_x_storage_cap_xr  # =0 for all T.index > 0
            ==
            self.hydro_ps_units_bid * self.data.hydro_ps_charging_efficiency
            - self.hydro_ps_units_offer / self.data.hydro_ps_discharging_efficiency
            ,
            name='hydro_ps_SOC_balance'
        )

        self.bess_units_SOC_balance = self.model.add_constraints(  # Shape: (T, G_bess)
            # Identical to pumped hydro storage SOC.
            self.bess_units_SOC.diff(n=1, dim="T")
            - self.data.bess_initial_SOC_x_storage_cap_xr
            ==
            self.bess_units_bid * self.data.bess_charging_efficiency
            - self.bess_units_offer / self.data.bess_discharging_efficiency
            ,
            name='bess_SOC_balance'
        )

    def _build_objective(self):
        """
        Define the objective function for minimization of negative social welfare.
        """
        self.model.add_objective(
            expr = (
                # Loads:
                - self.demand_inflexible_classic_bid * self.data.voll_classic
                # Generators:
                + self.wind_onshore_bid * self.data.wind_onshore_bid_price
                + self.wind_offshore_bid * self.data.wind_offshore_bid_price
                + self.solar_pv_bid * self.data.solar_pv_bid_price
                + self.hydro_ror_bid * self.data.hydro_ror_bid_price
               ).sum()
           
            # Important: variables with different dimensions must be in different parenthesis to be summed correctly
            # Loads:
            - (self.hydro_ps_units_bid * (self.data.hydro_ps_units_marginal_cost_dfs["Bid_price"])).sum()
            - (self.bess_units_bid * (self.data.bess_units_marginal_cost_dfs["Bid_price"])).sum()
            # Generators:
            + (self.conventional_units_bid * (self.data.conventional_units_marginal_cost_df)).sum()
            + (self.hydro_res_units_bid * (self.data.hydro_res_units_marginal_cost_df)).sum()
            + (self.hydro_ps_units_offer * (self.data.hydro_ps_units_marginal_cost_dfs["Offer_price"])).sum()
            + (self.bess_units_offer * (self.data.bess_units_marginal_cost_dfs["Offer_price"])).sum(),
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
