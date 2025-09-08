from pathlib import Path
from typing import Dict, List
import yaml
from tqdm import tqdm
from enlight.data_ops import DataProcessor
from enlight.model import EnlightModel
import enlight.utils as utils


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

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Load the configuration from the YAML file
        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config_yaml = yaml.safe_load(file)

        # Extract configuration values
        self.scenario_list: List[str] = self.config_yaml.get("scenario_list", [])

        # Extract duration settings
        duration = self.config_yaml.get("duration", {})
        self.start_week: int = duration.get("start_week")
        self.end_week: int = duration.get("end_week")

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

    def prepare_data_single_simulation(self, scenario_name) -> None:
        """Prepare input data for each scenario."""
        # Prepare data using DataProcessor for each scenario
        self.data_processor = DataProcessor(
            scenario_name=scenario_name,  # Name of the scenario
            config_yaml=self.config_yaml,  # Configuration for the scenario
            logger=self.logger,  # Logger for logging messages
        )

        self.logger.info(f"{scenario_name} : Data preparation completed.")

    def prepare_data_all_simulations(self) -> None:
        """Prepare input data for each scenario."""
        # Prepare data using DataProcessor for each scenario
        for scenario_name in tqdm(self.scenario_list, desc="Preparing the input data"):
            DataProcessor(
                scenario_name=scenario_name,  # Name of the scenario
                config_yaml=self.config_yaml,  # Configuration for the scenario
                logger=self.logger,  # Logger for logging messages
            )

            self.logger.info(f"{scenario_name} : Data preparation completed.")

    def run_single_simulation(self, week: int, simulation_path: Path) -> None:
        """
        Run a single simulation for a given week and simulation path.

        Args:
            week: The week number for the simulation
            simulation_path: The path to the simulation data
        """
        # Initialize EnlightModel for the given week and path
        self.enlight_model = EnlightModel(
            week=week, simulation_path=simulation_path, logger=self.logger
        )
        # Run the model
        self.enlight_model.run_model()

    def run_all_simulations(self) -> None:
        """Run all simulations for the configured scenarios (placeholder method)."""
        pass
