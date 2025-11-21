from pathlib import Path
from typing import Dict, List
import yaml
from tqdm import tqdm
from enlight.data_ops import DataProcessor
from enlight.data_ops import DataLoader
from enlight.model import EnlightModel
import enlight.utils as utils
from enlight.data_ops import DataVisualizer
from enlight.data_ops import ResultsVisualizer
import matplotlib.pyplot as plt


class EnlightRunner:
    """
    Handles configuration loading, data preparation, and model execution
    for Enlight energy modeling scenarios.
    """

    def __init__(self, root_path : Path) -> None:
        """Initialize the EnlightRunner."""
        
        self.root_path: Path = root_path
        self.logger = utils.setup_logging()

        self._load_config()
        utils.load_plot_config()

    def _load_config(self) -> None:
        self.config_path = self.root_path / 'config'
        self.config_yaml_path = self.config_path / "config.yaml"

        if not self.config_yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_yaml_path}")

        with open(self.config_yaml_path, "r", encoding="utf-8") as file:
            self.config_yaml = yaml.safe_load(file)

        # Parse scenario_list properly
        self.scenarios = {}   # dict: scenario_name → config_dict

        for entry in self.config_yaml.get("scenario_list", []):
            if not isinstance(entry, dict):
                raise ValueError("Each scenario entry must be a dictionary")

            scenario_name = list(entry.keys())[0]
            scenario_cfg = list(entry.values())[0]

            self.scenarios[scenario_name] = scenario_cfg

            # Create simulation folder structure
            for subfolder in ("data", "results"):
                path = Path(f"simulations/{scenario_name}/{subfolder}")
                path.mkdir(parents=True, exist_ok=True)

        # Global settings
        self.solver_name = str(self.config_yaml.get("solver_name"))
        self.bidding_zones = list(self.config_yaml.get("bidding_zones", []))

        # Log minimal information (no undefined week info)
        self.logger.info(
            "Loaded %d scenarios and %d bidding zones.",
            len(self.scenarios),
            len(self.bidding_zones)
        )

    def run_scenario(self, scenario_name):
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found in scenario_list")

        config = self.scenarios[scenario_name]
        mode = config["run_mode"]

        if mode == "yearly":
            self._run_yearly(scenario_name, config)

        elif mode == "weekly":
            self._run_weekly(scenario_name, config)

        else:
            raise ValueError("Unknown run_mode")


        
    def _run_yearly(self,scenario_name, config):
        """Prepare input data for each scenario."""
        print('ciaooo yearly')
        # DATA PREOPROCESSING
        # Prepare data using DataProcessor for each scenario
        # self.data_processor = DataProcessor(
        #     scenario_name=scenario_name,  # Name of the scenario
        #     config_yaml=self.config_yaml,  # Configuration for the scenario
        #     logger=self.logger,  # Logger for logging messages
        # )
        
        # # DATA LOADING
        # self.data = DataLoader(
        #     week=week,
        #     input_path=Path(simulation_path) / 'data',
        #     logger=self.logger)

        # # RUN MODEL
        # # Initialize EnlightModel with the given data #for the given week and path
        # self.enlight_model = EnlightModel(
        #     dataloader_obj=self.data,
        #     simulation_path=simulation_path,
        #     logger=self.logger
        # )
        # # Run the model
        # self.enlight_model.run_model()
        
        
    def _run_weekly(self, scenario_name, config):
        print('ciaooo weekly')
        # w1 = config["weeks"]["start_week"]
        # w2 = config["weeks"]["end_week"]

        # outputs = []

        # for week in range(w1, w2 + 1):
        #     dp = DataProcessor(scenario_name, config, week=week)
        #     weekly_result = Model(dp).run()
        #     outputs.append(weekly_result)

        # final = concat(outputs)
        # save(final)


    def prepare_data_single_scenario(self, scenario_name) -> None:
        """Prepare input data for each scenario."""
        # Prepare data using DataProcessor for each scenario
        self.data_processor = DataProcessor(
            scenario_name=scenario_name,  # Name of the scenario
            config_yaml=self.config_yaml,  # Configuration for the scenario
            logger=self.logger,  # Logger for logging messages
        )

        self.logger.info(f"{scenario_name} : Data preparation completed.")

    def prepare_data_all_scenarios(self) -> None:
        """Prepare input data for each scenario."""
        # Prepare data using DataProcessor for each scenario
        for scenario_name in tqdm(self.scenario_list, desc="Preparing the input data"):
            DataProcessor(
                scenario_name=scenario_name,  # Name of the scenario
                config_yaml=self.config_yaml,  # Configuration for the scenario
                logger=self.logger,  # Logger for logging messages
            )

            self.logger.info(f"{scenario_name} : Data preparation completed.")

    def load_data_single_simulation(self, week: int, simulation_path: Path) -> None:
        # Initialize DataLoader object to be used in EnlightModel:
        self.data = DataLoader(
            week=week,
            input_path=Path(simulation_path) / 'data',
            logger=self.logger)
        
    def load_data_all_simulations(self, simulation_path: Path) -> None:
        '''
        This method loads data for an entire year (the year given
        in the scenario fed to DataProcessor.)
        
        The method loads all of the processed data from that DataProcessor
        instance into 52 separate DataLoader objects. Afterwards this is
        used to instantiate 52 EnlightModels to get the electricity profiles
        and DA dispatch for the entire year.
        '''
        # Initialize empty dict to store all of the DataLoader instances.
        self.data_loader_dict = {}

        # Load the number of weeks defined in the yaml file.
        self.weeks = range(self.start_week, self.end_week+1)
        self.num_weeks = len(self.weeks)

        # Only not "self.weeks" during testing of the classes
        # self.sim_weeks = self.weeks
        self.sim_weeks = [self.weeks[0], self.weeks[-1]]

        # Load all of the data into separate DataLoader instances:
        for w in self.sim_weeks:
            self.data_loader_dict[f"week_{w}"] = (
                DataLoader(
                    week=w,
                    input_path=Path(simulation_path) / 'data',
                    logger=self.logger
                )
            )

    def run_single_simulation(self, simulation_path) -> None:
        """
        Run a single simulation for a given week and simulation path.

        Args:
            week: The week number for the simulation
            simulation_path: The path to the simulation data
        """
        
        # Initialize EnlightModel with the given data #for the given week and path
        self.enlight_model = EnlightModel(
            dataloader_obj=self.data,
            simulation_path=simulation_path,
            logger=self.logger
        )
        # Run the model
        self.enlight_model.run_model()

    def run_all_simulations(self, simulation_path: Path) -> None:
        """
        Run all simulations (i.e. weeks) for the configured scenarios.
        """
        # Initialize empty dict to store all of the DataLoader instances.
        self.enlight_models_dict = {}

        # Load all of the data into separate DataLoader instances:
        for w in self.sim_weeks:
            # Create EnlightModel instance for the given week w
            self.enlight_models_dict[f"week_{w}"] = (
                EnlightModel(
                    dataloader_obj=self.data_loader_dict[f"week_{w}"],
                    simulation_path=simulation_path,
                    logger=self.logger
                )
            )
            # Run the DA market model for the given week w and save the results
            self.enlight_models_dict[f"week_{w}"].run_model()

        # Combine the weekly electricity prices to get a single dataframe with
        #   all of the electricity prices in the year specified in the config file.
        result = 'electricity_prices'
        df_result = utils.combine_simulations_result(
            weeks=self.sim_weeks,
            result_path=simulation_path / 'results',
            result=result)
        self.df_result = df_result.set_index("T")

        utils.save_data(
            data=self.df_result,
            filename=f"test_yearly_{result}.csv",
            output_dir=simulation_path / 'results',
            logger=self.logger
        )
        # save df prics..

    def visualize_data(self, week: int, example_hour: int) -> None:
        """Visualize the data using DataVisualizer (placeholder method)."""
        # Check if the attribute has already been initialized
        #   by e.g. visualize_NBS_data().
        if not hasattr(self, "data_vis"):
            self.data_vis = DataVisualizer(
                dataprocessor_obj=self.data_processor,
                dataloader_obj=self.data,
                week=week,  # used only in a plot title
                palette=self.palette,  # used to ensure consistent plots
                logger=self.logger
            )
        self.data_vis.plot_annual_total_loads()
        self.data_vis.plot_total_installed_capacity()
        self.data_vis.plot_profiles()
        self.data_vis.plot_aggregated_supply_and_demand_curves(example_hour=example_hour)

        self.logger.info("Data visualization completed.")

    def visualize_results(self, example_hour: int):
        '''
        Visualize the market clearing with the zonal prices.
        '''
        if self.enlight_model.model.status != 'ok':
            self.logger.info("No results can be shown. Please run the model first")
        else:
            self.res_vis = ResultsVisualizer(
                enlightmodel_obj=self.enlight_model,
                week=self.enlight_model.data.week,
                palette=self.palette,  # used to ensure consistent plots
                logger=self.logger
            )
            self.res_vis.plot_aggregated_curves_with_zonal_prices(example_hour=example_hour)
            self.res_vis.plot_price_duration_curve()
            self.res_vis.plot_DA_schedule()

            self.logger.info("Results visualization completed.")

    def visualize_NBS_data(self, z0: str, prices_path: Path, week: int):
        '''
        Visualizes any interesting input data for the NBS.
        The week is currently not interesting but may become
        so in the future. It is needed to initialize the
        DataVisualizer instance.
        '''
        # Check if the attribute has already been initialized
        #   by e.g. visualize_data().
        if not hasattr(self, "data_vis"):
            self.data_vis = DataVisualizer(
                dataprocessor_obj=self.data_processor,
                dataloader_obj=self.data,
                week=week,  # used only in a plot title
                palette=self.palette,  # used to ensure consistent plots
                logger=self.logger
            )
        
        # Calls methods
        self.data_vis.visualize_NBS_inputs(z0=z0, prices_path=prices_path)
        