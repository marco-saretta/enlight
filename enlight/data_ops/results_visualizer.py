from logging import Logger 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import enlight.utils as utils
from enlight.model import EnlightModel


class ResultsVisualizer:
    """
    A class for visualizing the model results data using pandas, matplotlib, and seaborn.

    Attributes:
    - data_loader (DataLoader): An instance of DataLoader to load the data.
    """
    englightmodel_obj: EnlightModel
    logger: Logger

    def __init__(self, enlightmodel_obj, palette, logger):
        self.logger = logger
        self.logger.info("INITIALIZING RESULTS VISUALIZER")

        self.enlight_model = enlightmodel_obj

        self.data = self.enlight_model.data  # for easier handling
        
        self.start_date = f"01-01-{self.data.prediction_year}"
        self.dates = pd.date_range(start=self.start_date, periods=8760, freq='h')
        # self.dtu_colors = ['#990000', '#2F3EEA', '#1FD082', '#030F4F', '#F6D04D', '#FC7634', '#F7BBB1', '#E83F48', '#008835', '#79238E']
        self.palette = palette

    def plot_aggregated_curves_with_zonal_prices(self, example_hour):
        '''
        Overlay the aggregated supply and demand curves with the zonal electricity prices
        - example_hour: int
        - bids_accepted: dict of xarrays with the model solutions for each demand type
        - zonal_prices: pd.DataFrame with the hourly electricity price in each chosen bidding zone
        '''
        # Retrieve the demand bid volumes accepted in the DA market from the model solution
        bids_sol_dict = {
                'inflex' :
                self.enlight_model.demand_inflexible_classic_bid.solution,
                'flex': 
                self.enlight_model.demand_flexible_classic_bid.solution,
                'bess':
                self.enlight_model.bess_units_bid.solution,
                'phs':
                self.enlight_model.hydro_ps_units_bid.solution,
                'ptx':
                self.enlight_model.ptx_units_bid.solution,
                'dh':
                self.enlight_model.dh_units_bid.solution
        }  # Used to calculate the full volume traded. An offers_sol_dict would also do the trick

        # Retrieve from the model solution the zonal spot power prices
        zonal_prices = self.enlight_model.results_dict['electricity_prices']

        # Get the unsorted market curves
        demand_curve_raw, supply_curve_raw = utils.get_unsorted_aggregated_market_curves_from_dataloader_object(example_hour=example_hour, dataloader_obj=self.data)

        # Sort according to the merit order principle and plot the aggregated curves
        fig, ax = utils.make_aggregated_supply_and_demand_curves(
                    demand_curve_unsorted=demand_curve_raw,
                    supply_curve_unsorted=supply_curve_raw,
                    colors=self.palette
        )
        ax.set_title(f"Aggregated supply and demand curves on {self.dates[example_hour]}")
        
        # Calculate the observed equilibrium quantity in the example hour from the optimal solution
        q_eq = sum(  # sum the total network consumption for each demand type in the chosen hour
            map(  # for each demand type: 1) retrieve the xarray, 2) select the chosen hour, 3) sum across bidding zones and 4) retrieve the number as a Python float
                lambda xarr: xarr.isel(T=example_hour).sum().item(),
                bids_sol_dict.values()
            )
        )  # to inspect the mapping: print(list(map_bids))
        p_eqs = zonal_prices.iloc[example_hour]

        df = pd.DataFrame({
            "q_eq": [q_eq] * len(p_eqs),  # repeat scalar
            "p_eq": p_eqs.values
        }, index=p_eqs.index)
        df["Z"] = df.index

        sns.scatterplot(ax=ax, data=df, x="q_eq", y="p_eq", s=150, style="Z")
        ax.set_ylim(ax.get_ylim()[0], max(1.25*max(p_eqs.values), 25))
        plt.show()

        return None

    def plot_price_duration_curve(self):
        df_prices = self.enlight_model.results_dict['electricity_prices'].copy()
        df_prices = df_prices.apply(lambda x: x.sort_values().values, axis=0)
        df_prices.index.name="H"  # renaming index because the hours are no longer chronogical
        df_prices.index = df_prices.index[::-1]

        fig, ax = plt.subplots(figsize=(10,6))

        sns.lineplot(ax=ax, data=df_prices)
        ax.set_title("Zonal price duration curves")
        ax.set_ylabel("Power price [€/MWh]")
        plt.show()

    def plot_DA_schedule(self, starting_hour: int):
        '''
        This function uses the list of model.variables along
        with the model solution from results_dict to produce a
        stack plot of the generation
        
        Only uses:
        - self.enlight_model.model.variables
        - self.enlight_model.results_dict
        '''

        # Build lists of all the offers (generators/storage) and
        #   bids (demands/storage) that are in the model.
        # The model variables are retrieved directly from the model
        #   instead of from results_dict to ensure an error if results_dict
        #   has not been updated according to model.variables.
        list_keys_gen = []
        list_keys_cons = []
        for var in list(self.enlight_model.model.variables):
            if var.endswith('offer'):
                list_keys_gen.append(var + '_sol')
            elif var.endswith('bid'):
                list_keys_cons.append(var + '_sol')

        # 
        self.df_dispatch_full = pd.DataFrame({
            # key=technology/demand type
            # value=the total hourly dispatch across the network
            var: self.enlight_model.results_dict[var].sum(axis=1)
            for var in self.enlight_model.results_dict.keys()
            # Exclude technologies that do not exist in the system (e.g. no pumped hydro in DK or NO) 
            if not self.enlight_model.results_dict[var].empty and
            # Exclude any result that is not a dispatch schedule e.g. electricity prices
            (var.endswith('offer_sol') or var.endswith('bid_sol'))
        })

        # For more informational plot, include only a single week and change the index to datetime
        df_dispatch = self.df_dispatch_full.loc[starting_hour: starting_hour+167]  # reduced to only a single week. Selecting rows is end-inclusive
        df_dispatch.index = self.dates[starting_hour: starting_hour+168]

        # Plot the generation as an area chart by technology
        fig, ax = plt.subplots(figsize=(16,8))

            # Get the intersection of the bids & offers dispatched and the full list of offers 
        gen_dispatched = list(set(df_dispatch) & set(list_keys_gen))  # get the intersection of the bids & offers dispatched and the full list of offers
        ax.stackplot(
            df_dispatch.index,
            *df_dispatch[gen_dispatched].T.values/1e3,
            labels=gen_dispatched,
            alpha=0.5)

            # Get the intersection of the bids & offers dispatched and the full list of bids
        cons_dispatched = list(set(df_dispatch) & set(list_keys_cons))  # get the intersection of the bids & offers dispatched and the full list of bids
        ax.plot(df_dispatch.index, df_dispatch[cons_dispatched]/1e3, label=cons_dispatched)
        ax.plot(df_dispatch.index, df_dispatch[cons_dispatched].sum(axis=1)/1e3, label="Total demand in DA", c='k', lw=3)

        ax.tick_params(axis='x', rotation=30)
        ax.set_ylabel("Power generation/consumption [GW]")
        ax.set_title(f"Dispatch schedule for {df_dispatch.index[0].strftime('%d/%b--%H:%M')} to {df_dispatch.index[-1].strftime('%d/%b--%H:%M')}")
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()
