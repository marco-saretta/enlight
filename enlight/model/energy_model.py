from pathlib import Path
import numpy as np
import pandas as pd
import linopy

import enlight.utils as utils


class EnlightModel:
    """
    Electricity market optimization model using Linopy.

    Attributes:
        T (int): Number of time steps.
        Z (int): Number of zones.
        G (int): Number of conventional units.
        L (int): Number of transmission lines.
        G_hydro_res (int): Number of hydro reservoir units.
        G_hydro_ps (int): Number of pumped hydro storage units.
        G_bess (int): Number of battery energy storage system units.
        L_DH (int): Number of DH units
        L_PtX (int): Number of PtX units
    """

    def __init__(self, dataloader_obj, simulation_path, logger):
        # Initialize logger
        self.logger = logger
        self.data = dataloader_obj
        self.logger.info(
            f"INITIALIZING ENLIGHT MODEL"# FOR WEEK {self.data.week}"
        )

        self.simulation_path = simulation_path

        self.model = linopy.Model()

        self._aux_data()
        self._build_variables()
        self._build_constraints()
        self._build_objective()

    def _aux_data(self):
        """
        Extract and define core sets and their lengths from input data.
        """
        self.times = list(self.data.demand_inflexible_classic.index)  # Shape: (T,)
        self.time_index = pd.Index(np.arange(len(self.times)), name="T")
        self.bidding_zones = self.data.bidding_zones  # Shape: (Z,)

        self.T = len(self.times)     # Number of time steps
        self.Z = len(self.bidding_zones)  # Number of zones
        self.G = len(self.data.conventional_units_id)
        self.G_hydro_res = len(self.data.hydro_res_units_id)
        self.G_hydro_ps = len(self.data.hydro_ps_units_id)
        self.G_bess = len(self.data.bess_units_id)
        self.L_DH = len(self.data.dh_units_id)
        self.L_PtX = len(self.data.ptx_units_id)
        self.W = self.data.W  # Number of weeks included

    def _build_variables(self):
        """
        Declare model variables.

        Note: Unused variables will be excluded from the model unless referenced
                in the objective or constraints.
        """

        # Onshore wind production [MW]
        # Shape: (T, Z)
        self.wind_onshore_offer = self.model.add_variables(
            lower=0,
            upper=self.data.wind_onshore_production.values,   # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='wind_onshore_offer'
        )
        
        # Offshore wind production [MW]
        # Shape: (T, Z)
        self.wind_offshore_offer = self.model.add_variables(
            lower=0,
            upper=self.data.wind_offshore_production.values,   # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='wind_offshore_offer'
        )
        
        # Solar PV production [MW]
        # Shape: (T, Z)
        self.solar_pv_offer = self.model.add_variables(
            lower=0,
            upper=self.data.solar_pv_production.values,   # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='solar_pv_offer'
        )
        
        # Hydro ROR production [MW]
        # Shape: (T, Z)
        self.hydro_ror_offer = self.model.add_variables(
            lower=0,
            upper=self.data.hydro_ror_production.values,   # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='hydro_ror_offer'
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

        # Classic flexible demand [MW]
        # Shape: (T, Z)
        self.demand_flexible_classic_bid = self.model.add_variables(
            lower=0,
            upper=self.data.flexible_demands_dfs['demand_flexible_classic']['capacity'],  # Shape: (T, Z)
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name='demand_flexible_classic_bid'
        )
      
        # Thermal generation production variable (shape: T x G)
        # upper bound = generator capacity repeated for all time steps
        # np.outer(np.ones(T), capacities) produces a (T, G) matrix
        self.conventional_units_offer = self.model.add_variables(
            lower=0,
            upper=self.data.conventional_units_el_cap,
            coords=[self.times, self.data.conventional_units_id],
            dims=["T", "G"],
            name='conventional_units_offer'
        )

        # Hydro reservoir generator production variable (shape: T x G_hydro_res)
        # upper bound = generator capacity repeated for all time steps
        self.hydro_res_units_offer = self.model.add_variables(
            lower=0,
            upper=self.data.hydro_res_units_el_cap,  # np.array
            coords=[self.times, self.data.hydro_res_units_id],
            dims=["T", "G_hydro_res"],
            name='hydro_res_units_offer'
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

        # PtX bid
        self.ptx_units_bid = self.model.add_variables(
            lower=0,
            upper=self.data.ptx_units_el_cap,
            coords=[self.times, self.data.ptx_units_id],
            dims=["T", "L_PtX"],
            name='ptx_units_bid'
        )

        #  District heating bid for power-to-heat units
        self.dh_units_bid = self.model.add_variables(
            lower=0,
            upper=self.data.dh_units_el_cap,
            coords=[self.times, self.data.dh_units_id],
            dims=["T", "L_DH"],
            name='dh_units_bid'
        )

        # Electricity export
        self.electricity_export = self.model.add_variables(
            coords=[self.times, self.bidding_zones],
            dims=["T", "Z"],
            name = 'export'
        )
        
        self.lineflow = self.model.add_variables(
            lower = -self.data.lines_b_to_a_cap,  # shape: (T, L)
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
            (self.wind_onshore_offer
             + self.wind_offshore_offer
             + self.solar_pv_offer
             + self.hydro_ror_offer
             + self.conventional_units_offer.dot(self.data.G_Z_xr)  # type: ignore
             + self.hydro_res_units_offer.dot(self.data.G_hydro_res_Z_xr)
             + self.hydro_ps_units_offer.dot(self.data.G_hydro_ps_Z_xr)
             + self.bess_units_offer.dot(self.data.G_bess_Z_xr)
             ==
             self.demand_inflexible_classic_bid
             + self.demand_flexible_classic_bid
             + self.electricity_export
             + self.hydro_ps_units_bid.dot(self.data.G_hydro_ps_Z_xr)
             + self.bess_units_bid.dot(self.data.G_bess_Z_xr)
             + self.ptx_units_bid.dot(self.data.L_PtX_Z_xr)
             + self.dh_units_bid.dot(self.data.L_DH_Z_xr)
             ),
            name='power_balance'
            )

        self.electricity_exports = self.model.add_constraints(
            (self.lineflow.dot(self.data.L_Z_xr) == self.electricity_export),  # type: ignore
            name='electricity_exports'
            )

        # For specifically the following constraint we need to check if any
        #   hydro reservoir units are even in the model. Because if not, the
        #   model will break.
        self.hydro_res_energy_availability = {}
        if np.max(self.data.hydro_res_energy) > 0:
            # e.g. self.bess_units_bid is empty if there are no hydro_res_units but
            #   self.data.hydro_res_energy is not empty. It contains 0's in the
            #   chosen bidding zones if they don't have any hydro reservoir energy.
            '''MODIFIED constraint:'''
            # self.hydro_res_energy_availability[w] = self.model.add_constraints(
            #    # DESIRED formulation:
            #    # (self.data.T_W_xr.T.dot(
            #    #     self.hydro_res_units_offer.sum(dim='T').dot(
            #    #         self.data.G_hydro_res_Z_xr
            #    #     )
            #    # )
            #    <= self.data.hydro_res_energy.loc[w]),  # Shape: (W, Z)
            #    name=f'hydro_res_energy_availability_{w}'
            # )
            for w in range(1, self.data.W+1):
                # If this is not done as a loop we instead create W x Z constraints with TxG variables in each constraint (easily more than 2 million in each constraint where most are multiplied by 0)
                hours_in_week = self.data.T_W_xr.coords["T"][
                    self.data.T_W_xr.sel(W=w) == 1
                ]  # e.g. [1, 2, 3, ..., 167, 168] in week 1 or [8569, 8570, ..., 8759, 8760] in week 52
                self.hydro_res_energy_availability[w] = self.model.add_constraints(
                    (self.hydro_res_units_offer.sel(T=hours_in_week).dot(  # shape: (168, G_hydro_res). Last week has 24 extra hours...
                        self.data.G_hydro_res_Z_xr)  # shape: (168, Z)
                    ).sum("T")  # shape: (Z,)
                    <=
                    self.data.hydro_res_energy.loc[w]  # shape: (Z,)
                    ,
                    name=f"hydro_res_energy_availability_{w}"
                )

        '''MODIFIED constraint:'''
        self.demand_flexible_classic_limit = self.model.add_constraints(  # "amount" is the maximum weekly consumption of the flexible load in zone z
            (self.data.T_W_xr * self.demand_flexible_classic_bid).sum("T")  # shape: (W, Z)
            <=
            self.data.flexible_demands_dfs["demand_flexible_classic"]["amount"]
            ,
            name='demand_flexible_classic_limit'
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
                - self.demand_flexible_classic_bid * self.data.wtp_classic
                # Generators:
                + self.wind_onshore_offer * self.data.wind_onshore_bid_price
                + self.wind_offshore_offer * self.data.wind_offshore_bid_price
                + self.solar_pv_offer * self.data.solar_pv_bid_price
                + self.hydro_ror_offer * self.data.hydro_ror_bid_price
               ).sum()
           
            # Important: variables with different dimensions must be in different parenthesis to be summed correctly
            # Loads:
            - (self.hydro_ps_units_bid * (self.data.hydro_ps_units_marginal_cost_dfs["Bid_price"])).sum()
            - (self.bess_units_bid * (self.data.bess_units_marginal_cost_dfs["Bid_price"])).sum()
            - (self.ptx_units_bid * (self.data.ptx_units_bid_prices_df)).sum()
            - (self.dh_units_bid * (self.data.dh_units_bid_prices_df)).sum()
            # Generators:
            + (self.conventional_units_offer * (self.data.conventional_units_marginal_cost_df)).sum()
            + (self.hydro_res_units_offer * (self.data.hydro_res_units_marginal_cost_df)).sum()
            + (self.hydro_ps_units_offer * (self.data.hydro_ps_units_marginal_cost_dfs["Offer_price"])).sum()
            + (self.bess_units_offer * (self.data.bess_units_marginal_cost_dfs["Offer_price"])).sum(),
            sense="min"
        )

        print('obj added')

    def solve_model(self, solver_name):
        """
        Solve the model using the solver specified in yaml config file.
        """
        self.logger.info("Start solving model")
        if self.data.solver_name == "gurobi":
            self.model.solve(solver_name=solver_name, Method=1)  # use dual simplex instead of barrier algorithm immediately
        else:
            self.model.solve(solver_name=solver_name)
        self.logger.info('Model solved, good job champ!')

    def save_model_to_lp_file(self):
        """
        Export model to .lp file.
        """
        self.logger.info('Saving the .lp model file')
        Path('results').mkdir(parents=True, exist_ok=True)
        self.model.to_file(Path(self.simulation_path) / 'results' / 'debug_model_ptx_dh.lp', io_api='lp', explicit_coordinate_names=True)
        self.logger.info('Saved .lp model file')


    def run_model(self):
        """
        Solve the model using the specified solver.
        """
        self.solve_model(solver_name=self.data.solver_name)
        utils.save_model_results(self)#, week=self.data.week)
        # self.save_model_to_lp_file()
