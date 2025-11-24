from pathlib import Path
from typing import Dict
import yaml
from tqdm import tqdm
from enlight.data_ops import DataProcessor
from enlight.data_ops import DataLoader
from enlight.data_ops import DataExporter
from enlight.data_ops import DataVisualizer
from enlight.data_ops import ResultsVisualizer
from enlight.model import EnlightModel
import enlight.utils as utils
from enlight.utils import Timer, log_time

class EnlightRunner:
    """
    Handles configuration loading, data preparation, and model execution
    for Enlight energy modeling scenarios.
    """

    def __init__(self, root_path: Path) -> None:
        """Initialize the EnlightRunner."""
        
        self.logger = utils.setup_logging()
        self.logger.info('========== ENLIGHT object  initialization ==========')
        
        # Start timing to load the configuration
        config_timer = Timer(self.logger, "Loading general configuration")
        self.root_path: Path = root_path

        self._load_config()
        utils.load_plot_config()
        
        # Stop timing the entire scenario
        config_timer.stop()
        self.logger.info("Loading general configuration completed successfully\n")

    def _load_config(self) -> None:
        """Load configuration from YAML file and setup scenarios."""
        self.config_path = self.root_path / 'config'
        self.config_yaml_path = self.config_path / "config.yaml"

        if not self.config_yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_yaml_path}")

        with open(self.config_yaml_path, "r", encoding="utf-8") as file:
            self.config_yaml = yaml.safe_load(file)

        # Parse scenario_list properly
        self.scenarios_dict = {}   # dict: scenario_name → config_dict

        for entry in self.config_yaml.get("scenario_list", []):
            if not isinstance(entry, dict):
                raise ValueError("Each scenario entry must be a dictionary")

            scenario_name = list(entry.keys())[0]
            scenario_cfg = list(entry.values())[0]

            self.scenarios_dict[scenario_name] = scenario_cfg

            # Create simulation folder structure
            for subfolder in ("data", "results"):
                path = Path(f"simulations/{scenario_name}/{subfolder}")
                path.mkdir(parents=True, exist_ok=True)

        # Global settings
        self.solver_name = str(self.config_yaml.get("solver_name"))
        self.bidding_zones = list(self.config_yaml.get("bidding_zones", []))

        # Log minimal information
        self.logger.info(
            "Loaded %d scenarios and %d bidding zones.",
            len(self.scenarios_dict),
            len(self.bidding_zones)
        )

    def run_scenario(self, scenario_name: str) -> None:
            """
            Run a specific scenario based on its configuration.
            """
            # Start timing the entire scenario
            scenario_timer = Timer(self.logger, f"Scenario '{scenario_name}'")
            
            if scenario_name not in self.scenarios_dict:
                raise ValueError(f"Scenario '{scenario_name}' not found in scenario_list")

            # Get configuration for this specific scenario
            scenario_config = self.scenarios_dict[scenario_name]
            
            # Extract run mode
            mode = scenario_config.get("run_mode")
            
            self.logger.info(f"Running scenario '{scenario_name}' in mode: {mode}")

            if mode == "yearly":
                self._run_yearly(scenario_name, scenario_config)

            elif mode == "weekly":
                self._run_weekly(scenario_name, scenario_config)

            else:
                raise ValueError(f"Unknown run_mode: '{mode}'. Expected 'yearly' or 'weekly'")

            # Stop timing the entire scenario
            scenario_timer.stop()
            self.logger.info(f"Scenario '{scenario_name}' completed successfully\n")

    def _run_yearly(self, scenario_name, scenario_config: Dict) -> None:
        """
        Execute a yearly simulation scenario.
        
        Args:
            scenario_name: Name of the scenario
            config: Configuration dictionary for this scenario
        """
        
        self.logger.info(f"========== STARTING YEARLY RUN: {scenario_name} ==========")
        
        # STEP 1: DATA PREPROCESSING
        timer_preprocess = Timer(self.logger, "Data preprocessing")
        
        self.data_processor = DataProcessor(
            scenario_name=scenario_name,
            scenario_config=scenario_config,
            config_yaml=self.config_yaml,
            root_path=self.root_path,
            logger=self.logger,
        )
        timer_preprocess.stop()
        
        # STEP 2: DATA LOADING
        timer_loading = Timer(self.logger, "Data loading")
        
        self.data_y = DataLoader(
            scenario_name=scenario_name,
            scenario_config=scenario_config,
            config_yaml=self.config_yaml,
            root_path=self.root_path,
            logger=self.logger,
            )
        
        timer_loading.stop()
        
        # # STEP 3: MODEL EXECUTION
        # timer_model = Timer(self.logger, "Model execution")
        
        # self.enlight_model = EnlightModel(
        #     data=self.data_y,
        #     scenario_name=scenario_name,
        #     scenario_config=scenario_config,
        #     config_yaml=self.config_yaml,
        #     root_path=self.root_path,
        #     logger=self.logger,
        # )
        # self.enlight_model.run_model()
        
        # timer_model.stop()
        
        # # STEP 4: RESULTS EXTRACTION
        # timer_results = Timer(self.logger, "Results extraction")
        
        # # Assuming EnlightModel has a method to save results
        # # Adjust based on your actual implementation
        # if hasattr(self.enlight_model, 'save_results'):
        #     self.enlight_model.save_results(week=None)  # or year=year
        # else:
        #     self.logger.warning("No save_results method found in EnlightModel")
        
        # timer_results.stop()
        
        # self.logger.info(f"Yearly scenario '{scenario_name}' completed")
        
    def _run_weekly(self, scenario_name, scenario_config: Dict) -> None:
        """
        Execute a weekly simulation scenario.
        
        Args:
            scenario_name: Name of the scenario
            config: Configuration dictionary for this scenario
        """

        self.logger.info(f"========== STARTING YEARLY RUN: {scenario_name} ==========")
        
        # STEP 1: DATA PREPROCESSING
        timer_preprocess = Timer(self.logger, "Data preprocessing")
        
        self.data_processor = DataProcessor(
            scenario_name=scenario_name,
            scenario_config=scenario_config,
            config_yaml=self.config_yaml,
            root_path=self.root_path,
            logger=self.logger,
        )
        
        timer_preprocess.stop()
        week_dict = scenario_config['week_range']
        start_week = week_dict['start_week']
        end_week = week_dict['end_week']
        
        week_range = range(start_week, end_week + 1)
        
        for week in tqdm(week_range, desc="Running weekly simulations"):
            
            # STEP 2: DATA LOADING
            self.data_w = DataLoader(
                scenario_name=scenario_name,
                scenario_config=scenario_config,
                config_yaml=self.config_yaml,
                root_path=self.root_path,
                logger=self.logger,
                week=week
                )
            
            # STEP 3: MODEL EXECUTION
            # timer_model = Timer(self.logger, "Model execution")
            
            # self.enlight_model = EnlightModel(
            #     data=self.data_w,
            #     scenario_name=scenario_name,
            #     scenario_config=scenario_config,
            #     config_yaml=self.config_yaml,
            #     root_path=self.root_path,
            #     logger=self.logger,
            # )
            # self.enlight_model.run_model()
            
            # timer_model.stop()
            
            # Extract results
            # extractor  =DataExtractor(
            #     enlightmodel_obj=self.enlight_model,
        

        
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
        