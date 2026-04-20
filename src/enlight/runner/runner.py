from pathlib import Path
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from enlight.data_ops import DataProcessor, DataLoader, DataExporter, DataVisualizer, ResultsVisualizer
from enlight.model import EnlightModel
import enlight.utils as utils
from enlight.utils import Timer


class EnlightRunner:
    """
    Handles configuration loading, data preparation, and model execution
    for ENLIGHT energy market simulations.
    """

    def __init__(self, config: DictConfig) -> None:
        self.logger = utils.setup_logging()
        
        utils.log_section(self.logger, "ENLIGHT initialisation")

        timer = Timer(self.logger, "Loading configuration")

        self.config: DictConfig = config
        self.sim: DictConfig    = config.simulations       # shorthand used throughout
        self.root_path: Path    = Path(config.paths.root)
        self.solver_name: str   = config.solver_name
        self.bidding_zones: list = list(config.bidding_zones)

        self._setup_simulation_folders()
        utils.load_plot_config()

        timer.stop()
        self.logger.info(
            "Configuration loaded — simulation: '%s', zones: %d\n",
            self.sim.label,
            len(self.bidding_zones),
        )

    def _setup_simulation_folders(self) -> None:
        """Create output folder structure for the active simulation."""
        for subfolder in ("data", "results"):
            path = self.root_path / "simulations" / self.sim.label / subfolder
            path.mkdir(parents=True, exist_ok=True)

    def run(self, dry_run: bool = False) -> None:
        """
        Run the active simulation based on its configured mode.
        Entry point called from main.py.
        """
        mode = self.sim.run.mode
        label = self.sim.label

        self.logger.info("Running simulation '%s' in mode: %s", label, mode)
        scenario_timer = Timer(self.logger, f"Simulation '{label}'")

        if mode == "yearly":
            self._run_yearly(dry_run)
        elif mode == "rolling_horizon":
            self._run_rolling_horizon(dry_run)
        else:
            raise ValueError(f"Unknown run mode: '{mode}'. Expected 'yearly' or 'rolling_horizon'.")

        scenario_timer.stop()
        self.logger.info("Simulation '%s' completed successfully\n", label)

    def _run_yearly(self, dry_run: bool) -> None:
        """Execute a full-year simulation."""
        label = self.sim.label
        self.logger.info("========== YEARLY RUN: %s ==========", label)

        # Step 1 — preprocessing
        timer = Timer(self.logger, "Data preprocessing")
        self.data_processor = DataProcessor(
            sim_config=self.sim,
            config=self.config,
            root_path=self.root_path,
            logger=self.logger,
        )
        timer.stop()

        # Step 2 — loading
        timer = Timer(self.logger, "Data loading")
        self.data = DataLoader(
            sim_config=self.sim,
            config=self.config,
            root_path=self.root_path,
            logger=self.logger,
        )
        timer.stop()

        # Step 3 — model
        timer = Timer(self.logger, "Model execution")
        self.enlight_model = EnlightModel(
            data=self.data,
            sim_config=self.sim,
            config=self.config,
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
                sim_config=self.sim,
                root_path=self.root_path,
                logger=self.logger,
            )
            exporter.export_solution()
            timer.stop()

    def _run_rolling_horizon(self, dry_run: bool) -> None:
        """Execute a rolling-horizon simulation over a range of weeks."""
        label = self.sim.label
        self.logger.info("========== ROLLING HORIZON RUN: %s ==========", label)

        start_week = self.sim.run.rolling_horizon.start_week
        end_week   = self.sim.run.rolling_horizon.end_week

        # Step 1 — preprocessing (once, outside the weekly loop)
        timer = Timer(self.logger, "Data preprocessing")
        self.data_processor = DataProcessor(
            sim_config=self.sim,
            config=self.config,
            root_path=self.root_path,
            logger=self.logger,
        )
        timer.stop()

        # Step 2+3 — load and solve per week
        for week in tqdm(range(start_week, end_week + 1), desc="Rolling horizon"):
            self.logger.info("--- Week %d ---", week)

            self.data_w = DataLoader(
                sim_config=self.sim,
                config=self.config,
                root_path=self.root_path,
                logger=self.logger,
                week=week,
            )

            if not dry_run:
                self.enlight_model = EnlightModel(
                    data=self.data_w,
                    sim_config=self.sim,
                    config=self.config,
                    root_path=self.root_path,
                    logger=self.logger,
                )
                self.enlight_model.run_model()

                # TODO: export per-week results
                # TODO: concatenate weekly results after loop

    def visualize_data(self, week: int, example_hour: int) -> None:
        """Visualize preprocessed input data."""
        if not hasattr(self, "data_vis"):
            self.data_vis = DataVisualizer(
                dataprocessor_obj=self.data_processor,
                dataloader_obj=self.data,
                palette=self.palette,
                logger=self.logger,
            )
        self.data_vis.plot_annual_total_loads()
        self.data_vis.plot_total_installed_capacity()
        self.data_vis.plot_profiles(starting_hour=example_hour)
        self.data_vis.plot_aggregated_supply_and_demand_curves(example_hour=example_hour)
        self.logger.info("Data visualisation completed.")

    def visualize_results(self, example_hour: int) -> None:
        """Visualize market clearing results and zonal prices."""
        if self.enlight_model.model.status != "ok":
            self.logger.info("No results to show — run the model first.")
            return

        self.res_vis = ResultsVisualizer(
            enlightmodel_obj=self.enlight_model,
            palette=self.palette,
            logger=self.logger,
        )
        self.res_vis.plot_aggregated_curves_with_zonal_prices(example_hour=example_hour)
        self.res_vis.plot_price_duration_curve()
        self.res_vis.plot_DA_schedule(starting_hour=example_hour)
        self.logger.info("Results visualisation completed.")