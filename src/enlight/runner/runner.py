from pathlib import Path
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from enlight.data_ops import DataProcessor, DataLoader, DataExporter, DataVisualizer, ResultsVisualizer
from enlight.model import EnlightModel
import enlight.utils as utils
from enlight.utils import Timer
from enlight.utils.validation import validate_simulation_config


class EnlightRunner:
    """
    Handles configuration loading, data preparation, and model execution
    for ENLIGHT energy market simulations.
    """

    def __init__(self, config: DictConfig) -> None:
        # Logger is the first thing we set up so all subsequent steps can use it
        self.logger = utils.setup_logging(log_dir=str(Path(config.paths.log)))
        utils.log_section(self.logger, "ENLIGHT initialisation")

        # Store global config params
        self.global_config: DictConfig = config             # Global config dict
        self.sim_config: DictConfig = config.simulations    # Simulation config dict
        self.global_config.paths.root = Path(config.paths.root)  # Convert root as path
        self.root_path: Path = Path(config.paths.root)      # Rooth path
        self.solver_name: str = config.solver_name          # Solver
        self.bidding_zones: list = list(config.bidding_zones)   # Bidding zones list

        self._setup_simulation_folders()    # Make simulations directories
        utils.load_plot_config()            #  Init plot configs

        self.logger.info(
            "Configuration loaded — simulation: '%s', mode: '%s', zones: %d",
            self.sim_config.label,
            self.sim_config.run.mode,
            len(self.bidding_zones),
        )

    def _setup_simulation_folders(self) -> None:
        """Create output folder structure for the active simulation."""
        for subfolder in ("data", "results"):
            path = self.root_path / "simulations" / self.sim_config.label / subfolder
            path.mkdir(parents=True, exist_ok=True)

    def _validate_config(self) -> None:
        """Validate simulation config. Full logic lives in utils/validation.py."""
        validate_simulation_config(self.sim_config)
        self.logger.info("Config validation passed.")

    def run(self, dry_run: bool = False) -> None:
        """
        Run the active simulation based on its configured mode.
        Entry point called from main.py.
        """
        mode = self.sim_config.run.mode
        label = self.sim_config.label

        self._validate_config()
        utils.log_section(self.logger, f"RUN: {label}  [{mode}]")

        if mode == "yearly":
            self._run_yearly(dry_run)
        elif mode == "rolling_horizon":
            self._run_rolling_horizon(dry_run)
        else:
            raise ValueError(f"Unknown run mode: '{mode}'.")

        self.logger.info("Simulation '%s' completed successfully\n", label)

    def _run_rolling_horizon(self, dry_run: bool) -> None:
        """Execute a rolling-horizon simulation over a range of weeks."""
        # Read config parameters for week start and end
        start_week: int = self.sim_config.rolling_horizon.start_week
        end_week: int = self.sim_config.rolling_horizon.end_week

        self.logger.info("Preprocessing raw data for simulation - '%s'", self.sim_config.label)
        self._preprocess()

        for week in tqdm(range(start_week, end_week + 1), desc="Rolling horizon"):
             self.logger.info("--- Week %d / %d ---", week, end_week)
        #     self._load_data(week=week)
        #     self._solve(dry_run, week=week)
        #     if not dry_run:
        #         self._export(week=week)

        # if not dry_run:
        #     self._concatenate_weekly_results(start_week, end_week)

    def _run_yearly(self, dry_run: bool) -> None:
        """Execute a full-year simulation."""
        self._preprocess()
        #self._load_data()
        #self._solve(dry_run)
        #if not dry_run:
        #    self._export()

    def _preprocess(self) -> None:
        """Run data preprocessing (once per simulation)."""
        timer = Timer(self.logger, "Data preprocessing")
        
        self.data_processor = DataProcessor(
        sim_config=self.sim_config,
        global_config=self.global_config,
        logger=self.logger,
        )
        
        timer.stop()

    def _load_data(self, week: int | None = None) -> None:
        """Load model-ready data arrays (once per year, or once per week)."""
        label = "Data loading" if week is None else f"Data loading — week {week}"
        timer = Timer(self.logger, label)
        self.data = DataLoader(
            sim_config=self.sim,
            config=self.config,
            root_path=self.root_path,
            logger=self.logger,
            week=week,
        )
        timer.stop()

    def _solve(self, dry_run: bool, week: int | None = None) -> None:
        """Build and solve the optimisation model."""
        if dry_run:
            self.logger.info("Dry run — skipping model execution.")
            return

        label = "Model execution" if week is None else f"Model execution — week {week}"
        timer = Timer(self.logger, label)
        self.enlight_model = EnlightModel(
            data=self.data,
            sim_config=self.sim,
            config=self.config,
            root_path=self.root_path,
            logger=self.logger,
        )
        self.enlight_model.run_model()
        timer.stop()

    def _run_yearly(self, dry_run: bool) -> None:
        """Execute a full-year simulation."""
        label = self.sim_config.label

        utils.log_section(self.logger, f"YEARLY RUN - {label}")

        # Step 1 — preprocessing
        timer = Timer(self.logger, "Data preprocessing")
        self.data_processor = DataProcessor(
            sim_config=self.sim_config,
            config=self.global_config,
            root_path=self.root_path,
            logger=self.logger,
        )
        timer.stop()

        # Step 2 — loading
        timer = Timer(self.logger, "Data loading")
        self.data = DataLoader(
            sim_config=self.sim_config,
            config=self.global_config,
            root_path=self.root_path,
            logger=self.logger,
        )
        timer.stop()

        # Step 3 — model
        timer = Timer(self.logger, "Model execution")
        self.enlight_model = EnlightModel(
            data=self.data,
            sim_config=self.sim_config,
            config=self.global_config,
            root_path=self.root_path,
            logger=self.logger,
        )

        if dry_run:
            self.logger.info("Dry run — skipping model execution.")
        else:
            self.enlight_model.run_model()
        timer.stop()

        # Step 4 — export
        if not dry_run:
            timer = Timer(self.logger, "Results export")
            exporter = DataExporter(
                enlight_model=self.enlight_model,
                sim_config=self.sim_config,
                root_path=self.root_path,
                logger=self.logger,
            )
            exporter.export_solution()
            timer.stop()

    # def _run_rolling_horizon(self, dry_run: bool) -> None:
    #     """Execute a rolling-horizon simulation over a range of weeks."""
    #     label = self.sim_config.label

    #     utils.log_section(self.logger, f"ROLLING HORIZON RUN - {label}")

    #     start_week = int(self.sim_config.rolling_horizon.start_week)
    #     end_week = int(self.sim_config.rolling_horizon.end_week)

    #     # Preprocessing (once, outside the weekly loop)
    #     timer = Timer(self.logger, "Data preprocessing")
    #     self.data_processor = DataProcessor(
    #         sim_config=self.sim_config,
    #         config=self.global_config,
    #         root_path=self.root_path,
    #         logger=self.logger,
    #     )
    #     timer.stop()

    #     # Step 2+3 — load and solve per week
    #     for week in tqdm(range(start_week, end_week + 1), desc="Rolling horizon"):
    #         self.logger.info("--- Week %d ---", week)

    #         self.data_w = DataLoader(
    #             sim_config=self.sim_config,
    #             config=self.global_config,
    #             root_path=self.root_path,
    #             logger=self.logger,
    #             week=week,
    #         )

    #         if not dry_run:
    #             self.enlight_model = EnlightModel(
    #                 data=self.data_w,
    #                 sim_config=self.sim_config,
    #                 config=self.global_config,
    #                 root_path=self.root_path,
    #                 logger=self.logger,
    #             )
    #             self.enlight_model.run_model()

    #             # TODO: export per-week results
    #             # TODO: concatenate weekly results after loop




# TODO

# def _run_yearly(self, dry_run: bool) -> None:
#     """Execute a full-year simulation."""
#     self._preprocess()
#     self._load_data()
#     self._solve(dry_run)
#     if not dry_run:
#         self._export()


# def visualize_data(self, week: int, example_hour: int) -> None:
#     """Visualize preprocessed input data."""
#     if not hasattr(self, "data_vis"):
#         self.data_vis = DataVisualizer(
#             dataprocessor_obj=self.data_processor,
#             dataloader_obj=self.data,
#             palette=self.palette,
#             logger=self.logger,
#         )
#     self.data_vis.plot_annual_total_loads()
#     self.data_vis.plot_total_installed_capacity()
#     self.data_vis.plot_profiles(starting_hour=example_hour)
#     self.data_vis.plot_aggregated_supply_and_demand_curves(example_hour=example_hour)
#     self.logger.info("Data visualisation completed.")

# def visualize_results(self, example_hour: int) -> None:
#     """Visualize market clearing results and zonal prices."""
#     if self.enlight_model.model.status != "ok":
#         self.logger.info("No results to show — run the model first.")
#         return

#     self.res_vis = ResultsVisualizer(
#         enlightmodel_obj=self.enlight_model,
#         palette=self.palette,
#         logger=self.logger,
#     )
#     self.res_vis.plot_aggregated_curves_with_zonal_prices(example_hour=example_hour)
#     self.res_vis.plot_price_duration_curve()
#     self.res_vis.plot_DA_schedule(starting_hour=example_hour)
#     self.logger.info("Results visualisation completed.")