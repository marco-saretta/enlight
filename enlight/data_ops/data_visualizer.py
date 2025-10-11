from logging import Logger
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import enlight.utils as utils
from enlight.data_ops import DataProcessor
from enlight.data_ops import DataLoader


class DataVisualizer:
    """
    A class for visualizing the model inputs data using pandas, matplotlib, and seaborn..

    Attributes:
    - dataprocessor_obj (DataProcessor): An instance of DataProcessor to load the raw data.
    - dataloader_obj (DataLoader): An instance of DataLoader to load the (model-ready) data.
    - week: the same week as was chosen for the DataLoader object. Only used for titles in plots.
    - logger: logger
    """
    
    def __init__(self, dataprocessor_obj, dataloader_obj, week, palette, logger):
        """Initialize the DataLoader instance."""
        self.logger = logger
        self.logger.info("INITIALIZING DATA VISUALIZER")

        self.data_raw = dataprocessor_obj  # rename for convenience
        self.data = dataloader_obj  # rename for convenience
        self.week = week

        self.start_date = f"01-01-{self.data.prediction_year}"
        self.dates = pd.date_range(start=self.start_date, periods=8760, freq='h')

        self.palette = palette

    def plot_annual_total_loads(self) -> None:
        """Plot the annual total load for all bidding zones. Chosen zones stand out."""
        # Simplify DataFrame name
        total_load = self.data_raw.projection_row_seriess['demand_inflexible_classic']
        total_load.index.name = "Bidding Zones"

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(ax=ax,
                    x=total_load.index,
                    y=total_load.div(1e6).values,
                    label='Other Bidding Zones',
                    color=self.palette[0])  # red
        sns.barplot(ax=ax,
                    x=total_load[self.data.bidding_zones].index,
                    y=total_load[self.data.bidding_zones].div(1e6).values,
                    color=self.palette[1],  # blue
                    label='Selected Bidding Zones')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('Annual Total Load (TWh)')
        fig.tight_layout()
        plt.show()

    def plot_total_installed_capacity(self) -> None:
        """Plot the system total installed capacity and average annual load."""
        # GENERATION:

        # Retrieve the installed capacities of conventional units by fuel type and technology
        cap_conv_by_fuel = self.data.conventional_units_df.groupby('fuel')['capacity_el'].sum()
        # Retrieve the installed capacities of hydro reservoirs
        cap_hydro_res = pd.Series({
            'Reservoir hydro': self.data.hydro_res_units.capacity_el.sum()
        })

        # Retrieve the installed capacities of VRE units by technology
        cap_VRE_by_tech = pd.Series()
        for k, tech_df in self.data_raw.cap_year_dfs.items():
            # The string operators are simply used to change the series index
            #   from simply e.g. "wind_onshore" to "Wind onshore".
            cap_VRE_by_tech.loc[k.replace('_', ' ').capitalize()] = tech_df[self.data.bidding_zones].sum().sum()

        # STORAGE:
        cap_bess = pd.Series({
            'BESS': self.data.bess_units_df.capacity_el.sum()
        })
        cap_hydro_ps = pd.Series({
            'Pumped hydro': self.data.hydro_ps_units.capacity_el.sum()
        })

        # Calculate the hourly inflexible demand by type by averaging over the year
        mean_inflex_dem_by_type = pd.Series()
        for k, dem_df in self.data_raw.projection_row_seriess.items():
            # Divide the total annual demand by 8760 to get the mean hourly demand
            # and sum over the selected bidding zones to the the hourly system demand
            mean_inflex_dem_by_type.loc[k] = dem_df[self.data.bidding_zones].div(8760).sum()
        
        # Flexible demand is represented by its capacity
        cap_flex_dem_classic = self.data.flexible_demands_dfs['demand_flexible_classic']['capacity'].sum(axis=1)[0]

        # exclude unused demand types
        mean_inflex_dem_by_type = mean_inflex_dem_by_type.drop(labels='demand_inflexible_ev')
        mean_inflex_dem_by_type.index = ['mean inflex class dem']

        # Combine all conventional and VRE installed capacities
        installed_caps = pd.concat([
            cap_conv_by_fuel,
            cap_hydro_res,
            cap_VRE_by_tech,
            cap_bess,
            cap_hydro_ps
        ])

        # Plot the installed capacities and mean inflexible demand
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(ax=ax,
                    x=installed_caps.index,
                    y=installed_caps.div(1e3).values,  # MW -> GW
                    color=self.palette[1])
        ax.axhline(y=mean_inflex_dem_by_type.div(1e3).values, color=self.palette[0], linestyle='--', label='Avg. hourly inflexible classic demand [GW]')
        ax.axhline(y=cap_flex_dem_classic/1e3, color=self.palette[0], linestyle='-.', label='Capacity of flexible demand [GW]')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("Installed capacity by feedstock, VRE technology, & inflexible demand type")
        ax.set_ylabel('Total Installed Capacity [GW]')
        fig.tight_layout()
        plt.show()

    def plot_profiles(self) -> None:
        '''
        Plot time series profiles for:
        - Hourly VRE (capacity factors)
        - Hourly Load (normalized over the year). So Load.sum() = 1
            This is multiplied by the annutal total load to get
            the hourly load used in the DA market model.
        - Weekly Hydro reservoir energy availability
        '''
        profile_dict = dict(self.data_raw.profile_dfs)  # copy the dict
        profile_dict.pop("demand_inflexible_ev")  # remove unused data

        fig, ax = plt.subplots(ncols=2, nrows=len(profile_dict.keys()), figsize=(12,18))
        
        # used to increase legibility of plot with yearly time series
        alphas = 0.5 + (1-0.5) * np.arange(start=len(self.data.bidding_zones),stop=0,step=-1) / len(self.data.bidding_zones)

        # drop the column indicating the week of the year
        # profile_dict['solar_pv'] = profile_dict['solar_pv'].drop(columns=['Week'])
        
        #  iterate through the different VRE and inflexible load types in profile_dict
        for profile_idx, profile_label in enumerate(profile_dict.keys()):
            # set the index to datetime for prettier plot
            profile_dict[profile_label].index = self.dates 

            # simplify naming and only plot the chosen bidding zones to avoid a crowded plot
            profile_z = profile_dict[profile_label][self.data.bidding_zones]

            # plot the profile for an example WEEK
            profile_z.iloc[4368:4368+168].plot.line(ax=ax[profile_idx, 0], color=self.palette[:len(profile_z.columns)])

            # iterate through the bidding zones. Iteration needed to include decreasing opaqueness for the zones
            for i, col in enumerate(profile_z.columns):
                # plot the profile for the entire YEAR of vre_label (e.g. "onshore_wind") for the chosen bidding zones
                profile_z[col].plot.line(ax=ax[profile_idx, 1], alpha=alphas[i], color=self.palette[i])
            
            # add labels, legends, and titles to increase legibility of plots
            if profile_idx < len(profile_dict.keys())-1:
                ax[profile_idx,0].set_ylabel("Capacity factor [-]")
            else:
                ax[profile_idx,0].set_ylabel("Year-normalized load factor [-]")
            ax[profile_idx,1].legend()
            ax[profile_idx,0].set_title(f"Time series of example week for {profile_label}")
            ax[profile_idx,1].set_title(f"Yearly time series for {profile_label}")
        ax[profile_idx,0].set_xlabel("Time [h]")
        ax[profile_idx,1].set_xlabel("Time [h]")
        fig.tight_layout()
        plt.show()

    def plot_aggregated_supply_and_demand_curves(self, example_hour) -> None:
        """Plot aggregated supply and demand curves."""

        # Get the unsorted market curves
        # This is kept as a method ONLY to save the raw curves for easier inspection
        self.demand_curve_raw, self.supply_curve_raw = utils.get_unsorted_aggregated_market_curves_from_dataloader_object(example_hour=example_hour, dataloader_obj=self.data)

        # Feed the unsorted demand and supply data into the function that
        #   aggregates them and readies them for a step plot
        fig, ax = utils.make_aggregated_supply_and_demand_curves(
            demand_curve_unsorted=self.demand_curve_raw,
            supply_curve_unsorted=self.supply_curve_raw,
            colors=self.palette
        )
        ax.set_title(f"Aggregated supply and demand curves on {self.dates[(self.week-1)*7*24+example_hour]}")
        
        # Show or return the plot object to allow for adding results on top of the figure
        plt.show()

    def visualize_NBS_inputs(self, z0: str, prices_path: Path):
        '''
        Visualizes the power production and load consumptions profiles.
        Currently is visualizes it on a bidding zone level, but it should be
        modified later to take specific PPA developer and off-taker profiles instead.
        '''
        fig, ax = plt.subplots(figsize=(12,6))

        # Access and plot the production and consumption profiles from the DataProcessor instance
        for tech, df_ in self.data_raw.prod_dfs.items():
            df = utils.hourly_int_index_to_datetime(df=df_, year0=self.data_raw.prediction_year)
            df[z0].plot(ax=ax, label=tech)

        df_inflex_cla = utils.hourly_int_index_to_datetime(
            df=self.data_raw.cons_dfs['demand_inflexible_classic'],
            year0=self.data_raw.prediction_year
        )
        df_inflex_cla[z0].plot(ax=ax, label="inflex classic")
        
        df_flex_cla = utils.hourly_int_index_to_datetime(
            df=self.data_raw.flex_demands_dfs['DEMAND_FLEX_CLA_capacity'],
            year0=self.data_raw.prediction_year
        )
        df_flex_cla[z0].plot(ax=ax, label="flex classic")

        # Load the yearly power prices
        ax2 = ax.twinx()

        # Load the prices csv file if it exists
        df_prices_ = utils.load_csv_if_exists(path=prices_path)

        # If the DataFrame is loaded correctly (is not empty), it is plotted:
        if not df_prices_.empty:
            self.df_prices = utils.hourly_int_index_to_datetime(
                df=df_prices_,
                year0=self.data_raw.prediction_year
            )
            self.df_prices[z0].plot(ax=ax2, label="Power price", c='k', lw=2, alpha=0.7, ls='--')
    
        # Plot labels and legend
        ax.set_title(f"NBS input for zone {z0}")
        ax.set_ylabel("Power generation/consumption [MW]")
        ax2.set_ylabel("Power price [€/MWh]")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        plt.show()