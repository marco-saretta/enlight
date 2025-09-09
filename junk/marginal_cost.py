import pandas as pd

class MarginalCostCalculator:
    def __init__(
        self,
        generator_file="generators.csv",
        technology_file="technology_data.csv",
        emissions_file="emissions.csv",
        fuel_price_file="fuel_price_projections.csv",
        price_year=2040,
        output_file="generators_final.csv"
    ):
        self.generator_file = generator_file
        self.technology_file = technology_file
        self.emissions_file = emissions_file
        self.fuel_price_file = fuel_price_file
        self.price_year = price_year
        self.output_file = output_file

        self.df = None
        self.start_cols = None

    def load_data(self):
        # Read raw generator data
        df_raw = pd.read_csv(self.generator_file, index_col=0)
        self.df = df_raw.copy()
        self.start_cols = self.df.columns.tolist()

        # Read technology data
        df_tech_raw = pd.read_csv(
            self.technology_file, header=[0, 1], index_col=0, na_values=["---", "NA", "n/a"]
        )
        df_tech = df_tech_raw.copy()
        df_tech.columns = df_tech.columns.get_level_values(0)

        # Read emissions data
        df_emissions_raw = pd.read_csv(
            self.emissions_file, header=[0, 1], index_col=0, na_values=["---", "NA", "n/a"]
        )
        df_emissions = df_emissions_raw.copy()
        df_emissions.columns = df_emissions.columns.get_level_values(0)

        # Specify columns to import from technology data
        cols_to_keep = [
            "Electric efficiency CHP",
            "Heat efficiency CHP",
            "Electric efficiency condensing",
            "Heat efficiency",
            "CV",
            "CM",
            "CO2 capture rate (amount of emission)",
            'Var. O&M (el)'
        ]

        # Merge selected technology columns into generator data
        self.df = self.df.merge(
            df_tech[cols_to_keep],
            left_on="technology",
            right_index=True,
            how="left",
        )

        # Merge selected emissions columns into generator data
        self.df = self.df.merge(
            df_emissions[["co2_emission_pu", "so2_emissions_pu"]],
            left_on="fuel",
            right_index=True,
            how="left",
        )

    def calculate_costs(self):
        df = self.df

        # Calculate fuel consumption
        df["fuel_consumption"] = df["electric_capacity"] / df["Electric efficiency CHP"]

        # Calculate CO2 emissions
        df["co2_emission"] = (
            df["fuel_consumption"]
            * df["co2_emission_pu"]
            * 1e-3
            * (1 - df["CO2 capture rate (amount of emission)"])
        )

        # Read fuel price projections and select relevant year
        df_fuel_price = pd.read_csv(self.fuel_price_file, index_col=0)
        df_fuel_price_select = df_fuel_price[df_fuel_price.index == self.price_year].head()
        fuel_price_map = df_fuel_price_select.iloc[0].to_dict()

        # Map fuel prices to generator data
        df['fuel_price'] = df['fuel_type'].map(fuel_price_map)

        # Calculate fuel cost
        df['fuel_cost'] = df['fuel_consumption'] * df['fuel_price']

        # Calculate CO2 quota cost
        df['co2_quota_cost'] = df['co2_emission'] * fuel_price_map['CO2 quota']

        # Calculate variable O&M cost
        df['om_cost'] = df['Var. O&M (el)'] * df['electric_capacity']

        # Calculate total cost
        df['total_cost'] = df['fuel_cost'] + df['co2_quota_cost'] + df['om_cost']

        # Calculate marginal cost per unit of electric capacity
        df['marginal_cost'] = df['total_cost'] / df['electric_capacity']

        self.df = df

    def save_results(self):
        df_out = self.df[self.start_cols + ['marginal_cost']].copy()
        df_out.to_csv(self.output_file, index=True)

    def run(self):
        self.load_data()
        self.calculate_costs()
        self.save_results()
        
        
runner = MarginalCostCalculator()
runner.run()