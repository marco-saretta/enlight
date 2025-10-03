from pathlib import Path
from logging import Logger 
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

import enlight.utils as utils
from enlight.data_ops import DataProcessor
from enlight.data_ops import DataLoader

@dataclass
class DataVisualizer:
    """
    A class for visualizing the model inputs data using xarray and matplotlib.

    Attributes:
        data_loader (DataLoader): An instance of DataLoader to load the data.
    """
    scenario_name: str  # req. DataProcessor
    config_yaml: dict  # req. DataProcessor
    week: int  # req. DataLoader
    input_path: Path  # req. DataLoader
    logger: Logger  # req. both

    def __post_init__(self):
        """Initialize the DataLoader instance."""
        self.logger.info("INITIALIZING DATA VISUALIZER")
        self.input_path = Path(self.input_path).resolve()  # to get the absolute data path

        self.data_raw = DataProcessor(
            scenario_name=self.scenario_name,  # Name of the scenario
            config_yaml=self.config_yaml,  # Configuration for the scenario
            logger=self.logger,  # Logger for logging messages
        )

        # We're interested in the entire year.
        self.data = DataLoader(week=self.week,
                              input_path=self.input_path / 'data',
                              logger=self.logger)
        
        self.start_date = f"01-01-{self.data.prediction_year}"
        self.dates = pd.date_range(start=self.start_date, periods=8760, freq='h')

        self.dtu_colors = ['#990000', '#2F3EEA', '#1FD082', '#030F4F', '#F6D04D', '#FC7634', '#F7BBB1', '#E83F48', '#008835', '#79238E']

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
                    color=self.dtu_colors[0])  # red
        sns.barplot(ax=ax,
                    x=total_load[self.data.bidding_zones].index,
                    y=total_load[self.data.bidding_zones].div(1e6).values,
                    color=self.dtu_colors[1],  # blue
                    label='Selected Bidding Zones')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('Annual Total Load (TWh)')
        fig.tight_layout()
        plt.show()

    def plot_total_installed_capacity(self) -> None:
        """Plot the system total installed capacity and average annual load."""
        # Retrieve the installed capacities of conventional units by fuel type and technology
        cap_conv_by_fuel = self.data.conventional_units_df.groupby('fuel')['capacity_el'].sum()

        # Retrieve the installed capacities of VRE units by technology
        cap_VRE_by_tech = pd.Series()
        for k, tech_df in self.data_raw.cap_year_dfs.items():
            cap_VRE_by_tech.loc[k] = tech_df[self.data.bidding_zones].sum().sum()

        # Calculate the hourly inflexible demand by type by averaging over the year
        mean_inflex_dem_by_type = pd.Series()
        for k, dem_df in self.data_raw.projection_row_seriess.items():
            # Divide the total annual demand by 8760 to get the mean hourly demand
            # and sum over the selected bidding zones to the the hourly system demand
            mean_inflex_dem_by_type.loc[k] = dem_df[self.data.bidding_zones].div(8760).sum()
        
        # exclude unused demand types
        mean_inflex_dem_by_type = mean_inflex_dem_by_type.drop(labels='demand_inflexible_ev')
        mean_inflex_dem_by_type.index = ['mean inflex class dem']

        # Combine all conventional and VRE installed capacities
        installed_caps = pd.concat([ cap_conv_by_fuel, cap_VRE_by_tech])  #, mean_inflex_dem_by_type])

        # Plot the installed capacities and mean inflexible demand
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(ax=ax,
                    x=installed_caps.index,
                    y=installed_caps.div(1e3).values,  # MW -> GW
                    color=self.dtu_colors[1])
        ax.axhline(y=mean_inflex_dem_by_type.div(1e3).values, color=self.dtu_colors[0], linestyle='--', label='Avg. hourly inflexible classic demand [GW]')
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
            profile_z.iloc[4368:4368+168].plot.line(ax=ax[profile_idx, 0], color=self.dtu_colors[:len(profile_z.columns)])

            # iterate through the bidding zones. Iteration needed to include decreasing opaqueness for the zones
            for i, col in enumerate(profile_z.columns):
                # plot the profile for the entire YEAR of vre_label (e.g. "onshore_wind") for the chosen bidding zones
                profile_z[col].plot.line(ax=ax[profile_idx, 1], alpha=alphas[i], color=self.dtu_colors[i])
            
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

    def plot_aggregated_supply_and_demand_curves(self, example_hour, show=True) -> None:
        """Plot aggregated supply and demand curves."""
        # Placeholder for actual implementation
        h = example_hour  # descriptive name as input, but simple name is more functional

        # Collect all of the demand
        self.demand_curve_raw = np.vstack([
            # 1st column is the bid quantity
                # Inflexible demands
                    # classic
            [self.data.demand_inflexible_classic.sum(axis=1).iloc[h]  #,
                # Flexible demands
                    # classic
                    # 1/4 of the inflexible demand is currently visualized as flexible until merging with the branch that has flexible demand
            #self.data.demand_inflexible_classic.sum(axis=1).div(4).iloc[h]
            ]
            ,
            # 2nd column is the bid price (VOLL or WTP)
            [self.data.voll_classic  #,
            #self.data.voll_classic/10
            ]
        ])

        # Collect all of the supply
        self.supply_curve_raw = np.vstack([
            # 1st column is the offer quantity
            np.hstack([
                # VREs
                self.data.solar_pv_production.sum(axis=1).iloc[h],
                self.data.wind_onshore_production.sum(axis=1).iloc[h],
                self.data.wind_offshore_production.sum(axis=1).iloc[h],
                # Dispatchable
                self.data.conventional_units_df.capacity_el.values,
                self.data.hydro_res_units.capacity_el.values
            ])
            ,
            # 2nd column is the offer price (marginal cost)
            np.hstack([
                # VREs
                self.data.solar_pv_bid_price,
                self.data.wind_onshore_bid_price,
                self.data.wind_offshore_bid_price,
                # Dispatchable
                self.data.conventional_units_df.prodcost.values,
                self.data.hydro_res_units.prodcost.values
            ])
        ])

        # Feed the unsorted demand and supply data into the function that
        #   aggregates them and readies them for a step plot
        fig, ax = utils.make_aggregated_supply_and_demand_curves(
            demand_curve_unsorted=self.demand_curve_raw,
            supply_curve_unsorted=self.supply_curve_raw,
            colors=self.dtu_colors
        )
        ax.set_title(f"Aggregated supply and demand curves on {self.dates[self.week*7*24+h]}")
        
        # Show or return the plot object to allow for adding results on top of the figure
        if show:
            plt.show()
            return None
        return fig, ax
    
    def plot_aggregated_curves_with_zonal_prices(self, example_hour, bids_accepted, zonal_prices):
        '''
        Overlay the aggregated supply and demand curves with the zonal electricity prices
        '''
        fig, ax = self.plot_aggregated_supply_and_demand_curves(example_hour=example_hour, show=False)
        q_eq = bids_accepted.isel(T=example_hour).sum()
        p_eqs = zonal_prices.iloc[example_hour]

        df = pd.DataFrame({
            "q_eq": [q_eq.item()] * len(p_eqs),  # repeat scalar
            "q": bids_accepted.isel(T=example_hour),
            "p_eq": p_eqs.values
        }, index=p_eqs.index)
        df["Z"] = df.index

        sns.scatterplot(ax=ax, data=df, x="q_eq", y="p_eq", s=200, style="Z")
        ax.set_ylim(ax.get_ylim()[0], 1.25*max(p_eqs.values))
        plt.show()

        return None

