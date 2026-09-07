import yfinance as yf
import matplotlib.pyplot as plt

class GasPricesProfile:
    """
    Class to download, process, and visualize Dutch TTF gas prices from Yahoo Finance.

    Attributes:
        ticker_symbol (str): Yahoo Finance ticker for TTF futures.
        profile_years (list): List of years to create profiles for.
        df_daily (pd.Series): Daily TTF closing prices in UTC.
        profiles_dict (dict): Raw yearly price series.
        profiles_scaled (dict): Yearly prices scaled by mean.
        profiles_volatility (dict): Volatility measures per year.
    """
    def __init__(self, profile_years=['2018','2019','2020','2023','2024','2025']):
        self.ticker_symbol = "TTF=F"
        self.profile_years = profile_years
        self._get_raw_data()
        self._process_data()
        self._generate_profiles()
        self._plot_results()
        
    def _get_raw_data(self):
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.raw_data = self.ticker.history(period="max", interval="1d")
        
    def _process_data(self):
        self.df = self.raw_data.copy()
        self.df.index = self.df.index.tz_convert("UTC")
        self.df = self.df.resample('M').mean()
        self.df.ffill(inplace=True)
        self.df_daily = self.df['Close']
        
    def _generate_profiles(self):
        self.profiles_dict = {}
        self.profiles_scaled = {}
        self.profiles_volatility = {}
        
        for year in self.profile_years:
            yearly_data = self.df_daily[self.df_daily.index.year == int(year)]
            scaled_yearly_data = yearly_data / yearly_data.mean()
            self.profiles_dict[year] = yearly_data
            self.profiles_scaled[year] = scaled_yearly_data
            self.profiles_volatility[year] = scaled_yearly_data.std() 
            
    def _plot_results(self):  
        for year, profile in self.profiles_scaled.items():
            profile.plot(label=year)
        plt.title("Scaled Daily TTF Gas Prices")
        plt.xlabel("Date")
        plt.ylabel("Scaled Price")
        plt.legend()
        plt.show()


# Instantiate the class
p = GasPricesProfile()
p.profiles_scaled['2019'].plot()
(p.profiles_scaled['2019'] * 30).plot()
