from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from enlight.data_ops import DataProcessor, DataLoader, DataExporter
from enlight.model import EnlightModel
import enlight.utils as utils
from enlight.utils import Timer
from enlight.utils.validation import validate_simulation_config

class EnlightRunner:
    """
    Runner for the ENLIGHT simulation pipeline.

    Scope:
        - Validate and store configuration
        - Create output folder structure
        - Dispatch to the correct run mode
        - Coordinate preprocessing, data loading, model execution, and export
    """

    def __init__(self, config: DictConfig) -> None:
        """
        Initialise the runner from a Hydra DictConfig.

        Sets up logging, stores configuration references, creates output
        directories, and logs a summary of the active simulation settings.

        Args:
            config: Top-level Hydra config (default_config.yaml merged with
                    the active simulation file).
        """
        self.logger = utils.setup_logging(log_dir=str(Path(config.paths.log)))
        utils.log_section(self.logger, "ENLIGHT initialisation")

        self.global_config: DictConfig  = config
        self.sim_config: DictConfig     = config.simulations
        self.root_path: Path            = Path(config.paths.root)
        self.solver_name: str           = config.solver_name
        self.bidding_zones: list        = list(config.bidding_zones)

        self._setup_simulation_folders()
        utils.load_plot_config()

        self.logger.info(
            "Configuration loaded -- simulation: '%s', mode: '%s', zones: %d",
            self.sim_config.label,
            self.sim_config.run.mode,
            len(self.bidding_zones),
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> None:
        """
        Run the active simulation end-to-end. Single entry point from main.py.

        Validates the simulation config, then dispatches to the appropriate
        run-mode method (_run_yearly or _run_rolling_horizon).

        Args:
            dry_run: When True, preprocessing and data loading run normally
                     but model execution and export are skipped. Useful for
                     checking that input data is complete before a long solve.
        """
        mode  = self.sim_config.run.mode
        label = self.sim_config.label

        validate_simulation_config(self.sim_config)
        self.logger.info("Config validation passed.")

        utils.log_section(self.logger, f"RUN: {label}  [{mode}]")
        sim_timer = Timer(self.logger, f"Simulation '{label}'")

        if mode == "yearly":
            self._run_yearly(dry_run)
        elif mode == "rolling_horizon":
            self._run_rolling_horizon(dry_run)
        else:
            raise ValueError(f"Unknown run mode '{mode}'. Valid options: yearly, rolling_horizon")

        sim_timer.stop()
        self.logger.info("Simulation '%s' completed successfully", label)

    # -------------------------------------------------------------------------
    # Run modes
    # -------------------------------------------------------------------------

    def _run_yearly(self, dry_run: bool) -> None:
        """
        Execute a single full-year optimisation (8760 hours).

        Preprocessing runs once, then data is loaded for the full year,
        the model is solved, and results are exported.
        """
        self._preprocess()
        self._load_data()
        self._solve(dry_run)
        if not dry_run:
            self._export()

    def _run_rolling_horizon(self, dry_run: bool) -> None:
        """
        Execute a rolling-horizon simulation over a configured range of weeks.

        Preprocessing runs once before the loop. Each week is then loaded,
        solved, and exported independently. After the loop, per-week result
        files are concatenated into a single annual output.
        """
        start_week = self.sim_config.rolling_horizon.start_week
        end_week   = self.sim_config.rolling_horizon.end_week

        self._preprocess()

        for week in tqdm(range(start_week, end_week + 1), desc="Rolling horizon"):
            self.logger.info("Week %d / %d", week, end_week)
            self._load_data(week=week)
            self._solve(dry_run, week=week)
            if not dry_run:
                self._export(week=week)

        if not dry_run:
            self._concatenate_weekly_results(start_week, end_week)

    # -------------------------------------------------------------------------
    # Pipeline steps
    # -------------------------------------------------------------------------

    def _preprocess(self) -> None:
        """
        Run DataProcessor to transform raw inputs into model-ready CSV files.

        This step reads from data/ and writes to simulations/<label>/data/.
        It only needs to run once per simulation, even in rolling-horizon mode.
        """
        timer = Timer(self.logger, "Data preprocessing")
        self.data_processor = DataProcessor(
            sim_config=self.sim_config,
            global_config=self.global_config,
            logger=self.logger,
        )
        timer.stop()

    def _load_data(self, week: int | None = None) -> None:
        """
        Run DataLoader to build linopy-compatible arrays from preprocessed CSVs.

        In yearly mode, loads all 8760 hours at once.
        In rolling-horizon mode, called once per week with the week index.

        Args:
            week: Week index (1-52) for rolling-horizon runs; None for yearly.
        """
        label = "Data loading" if week is None else f"Data loading -- week {week}"
        timer = Timer(self.logger, label)
        self.data = DataLoader(
            scenario_name=self.sim_config.label,
            scenario_config=OmegaConf.to_container(self.sim_config, resolve=True),
            config_yaml=OmegaConf.to_container(self.global_config, resolve=True),
            logger=self.logger,
            root_path=self.root_path,
            week=week,
        )
        timer.stop()

    def _solve(self, dry_run: bool, week: int | None = None) -> None:
        """
        Build and solve the market-clearing optimisation model.

        Skipped entirely when dry_run is True.

        Args:
            dry_run: When True, logs a message and returns without solving.
            week: Week index for rolling-horizon runs; None for yearly.
        """
        if dry_run:
            self.logger.info("Dry run -- skipping model execution.")
            return

        label = "Model execution" if week is None else f"Model execution -- week {week}"
        timer = Timer(self.logger, label)
        self.enlight_model = EnlightModel(
            data=self.data,
            scenario_name=self.sim_config.label,
            scenario_config=OmegaConf.to_container(self.sim_config, resolve=True),
            config_yaml=OmegaConf.to_container(self.global_config, resolve=True),
            root_path=self.root_path,
            logger=self.logger,
        )
        self.enlight_model.run_model()
        timer.stop()

    def _export(self, week: int | None = None) -> None:
        """
        Export optimisation results to simulations/<label>/results/.

        Args:
            week: Week index for rolling-horizon runs; None for yearly.
        """
        label = "Results export" if week is None else f"Results export -- week {week}"
        timer = Timer(self.logger, label)
        DataExporter(
            enlight_model=self.enlight_model,
            scenario_name=self.sim_config.label,
            scenario_config=OmegaConf.to_container(self.sim_config, resolve=True),
            root_path=self.root_path,
            logger=self.logger,
        ).export_solution()
        timer.stop()

    def _concatenate_weekly_results(self, start_week: int, end_week: int) -> None:
        """
        Merge per-week result CSVs into a single annual file.

        Called once after the rolling-horizon loop completes.

        Args:
            start_week: First week that was simulated.
            end_week:   Last week that was simulated.
        """
        timer = Timer(self.logger, "Concatenating weekly results")
        results_path = self.root_path / "simulations" / self.sim_config.label / "results"
        utils.combine_simulations_result(
            weeks=list(range(start_week, end_week + 1)),
            result_path=results_path,
            result="electricity_prices",
        )
        timer.stop()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _setup_simulation_folders(self) -> None:
        """Create data/ and results/ subdirectories under simulations/<label>/."""
        for subfolder in ("data", "results"):
            (self.root_path / "simulations" / self.sim_config.label / subfolder).mkdir(
                parents=True, exist_ok=True
            )