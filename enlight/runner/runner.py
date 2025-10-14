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


class EnlightRunner:
    """
    Handles configuration loading, data preparation, and model execution
    for Enlight energy modeling scenarios.
    """

    def __init__(self) -> None:
        """Initialize the EnlightRunner."""
        self.config_path: Path = Path("config") / "config.yaml"
        self.config: Dict = {}

        self.logger = utils.setup_logging()

        self._load_config()
        self._create_directories()
        self._load_plot_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Load the configuration from the YAML file
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config_yaml = yaml.safe_load(file)

        # Extract configuration values
        self.scenario_list: List[str] = self.config_yaml.get("scenario_list", [])

        # Extract solver name
        self.solver_name: str = self.config_yaml.get("solver_name", [])

        # Extract bidding zones
        self.bidding_zones: List[str] = self.config_yaml.get("bidding_zones", [])

        # Register into the logger
        self.logger.info("Loaded config for %d scenarios.", len(self.scenario_list))

    def _create_directories(self) -> None:
        """Create required directories for each scenario."""
        for scenario in self.scenario_list:
            for subfolder in ("data", "results"):
                # Create directories for data and results for each scenario
                path = Path(f"simulations/{scenario}/{subfolder}")
                path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Directories for simulations created.")

    def _load_plot_config(self) -> None:
        # Ensure consistent color palette across plots
        dtu_colors = ['#990000', '#2F3EEA', '#1FD082', '#030F4F', '#F6D04D', '#FC7634', '#F7BBB1', '#E83F48', '#008835', '#79238E']
        self.palette = dtu_colors
        utils.load_plot_config(palette=self.palette)

    def prepare_data_single_scenario(self, scenario_name: str, overwrite: bool=True) -> None:
        """Prepare input data for each scenario."""
        # Prepare data using DataProcessor for each scenario
        self.data_processor = DataProcessor(
            scenario_name=scenario_name,  # Name of the scenario
            config_yaml=self.config_yaml,  # Configuration for the scenario
            logger=self.logger,  # Logger for logging messages
            overwrite_preprocessed_data=overwrite
        )

        if overwrite:
            self.logger.info(f"{scenario_name} : Data pre-processing completed.")
        else:
            self.logger.info(f"{scenario_name} : Raw data successfully loaded.")

    # def prepare_data_all_scenarios(self) -> None:
    #     """Prepare input data for each scenario."""
    #     # Prepare data using DataProcessor for each scenario
    #     for scenario_name in tqdm(self.scenario_list, desc="Preparing the input data"):
    #         DataProcessor(
    #             scenario_name=scenario_name,  # Name of the scenario
    #             config_yaml=self.config_yaml,  # Configuration for the scenario
    #             logger=self.logger,  # Logger for logging messages
    #         )

    #         self.logger.info(f"{scenario_name} : Data preparation completed.")

    def load_data_single_simulation(self, simulation_path: Path) -> None:
        # Initialize DataLoader object to be used in EnlightModel:
        self.data = DataLoader(
            input_path=Path(simulation_path) / 'data',
            logger=self.logger)

    def run_single_simulation(self, simulation_path) -> None:
        """
        Run a single simulation and simulation path.

        Args:
            simulation_path: The path to the simulation data
        """
        
        # Initialize EnlightModel with the given data object
        self.enlight_model = EnlightModel(
            dataloader_obj=self.data,
            simulation_path=simulation_path,
            logger=self.logger
        )
        # Run the model
        self.enlight_model.run_model()

    def visualize_data(self, example_hour: int) -> None:
        """Visualize the data using DataVisualizer (placeholder method)."""
        # Check if the attribute has already been initialized
        #   by e.g. visualize_NBS_data().
        if not hasattr(self, "data_vis"):
            self.data_vis = DataVisualizer(
                dataprocessor_obj=self.data_processor,
                dataloader_obj=self.data,
                palette=self.palette,  # used to ensure consistent plots
                logger=self.logger
            )
        self.data_vis.plot_annual_total_loads()
        self.data_vis.plot_total_installed_capacity()
        self.data_vis.plot_profiles(starting_hour=example_hour)
        self.data_vis.plot_aggregated_supply_and_demand_curves(example_hour=example_hour)

        self.logger.info("Data visualization completed.")

    def visualize_results(self, example_hour: int) -> None:
        '''
        Visualize the market clearing with the zonal prices.
        '''
        if self.enlight_model.model.status != 'ok':
            self.logger.info("No results can be shown. Please run the model first")
        else:
            self.res_vis = ResultsVisualizer(
                enlightmodel_obj=self.enlight_model,
                palette=self.palette,  # used to ensure consistent plots
                logger=self.logger
            )
            self.res_vis.plot_aggregated_curves_with_zonal_prices(example_hour=example_hour)
            self.res_vis.plot_price_duration_curve()
            self.res_vis.plot_DA_schedule(starting_hour=example_hour)

            self.logger.info("Results visualization completed.")

    # Not yet ready to be used:
    # def visualize_NBS_data(self, bidding_zone: str, prices_path: Path) -> None:
    #     '''
    #     Visualizes any interesting input data for the NBS.
    #     The week is currently not interesting but may become
    #     so in the future. It is needed to initialize the
    #     DataVisualizer instance.
    #     '''
    #     # Check if the attribute has already been initialized
    #     #   by e.g. visualize_data().
    #     if not hasattr(self, "data_vis"):
    #         self.data_vis = DataVisualizer(
    #             dataprocessor_obj=self.data_processor,
    #             dataloader_obj=self.data,
    #             palette=self.palette,  # used to ensure consistent plots
    #             logger=self.logger
    #         )
        
    #     # Calls methods
    #     self.data_vis.visualize_NBS_inputs(z0=bidding_zone, prices_path=prices_path)
        