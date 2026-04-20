from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from enlight.data_ops import DataProcessor, DataLoader, DataExporter
from enlight.model import EnlightModel
from enlight.utils import validation
import enlight.utils as utils
from enlight.utils.utils import Timer

# ---------------------------------------------------------------------------
# Run mode registry — add new modes here only
# ---------------------------------------------------------------------------
_RUN_MODES: dict[str, str] = {
    "yearly":           "_run_yearly",
    "rolling_horizon":  "_run_rolling_horizon",
}


class EnlightRunner:
    """
    Orchestrates the ENLIGHT simulation pipeline.

    Responsibilities:
        - Validate and load configuration
        - Create output folder structure
        - Dispatch to the correct run mode
        - Coordinate preprocessing, data loading, model execution, and export

    Not responsible for:
        - Data logic (DataProcessor / DataLoader)
        - Model math (EnlightModel)
        - Post-run visualisation (use notebooks or a PostProcessor)
    """

    def __init__(self, config: DictConfig) -> None:
        # Logger is the first thing we set up so all subsequent steps can use it
        self.logger = utils.setup_logging(log_dir=str(Path(config.paths.log)))
        utils.log_section(self.logger, "ENLIGHT initialisation")

        timer = Timer(self.logger, "Loading and validating configuration")

        # Validate sim config before doing anything else
        sim_raw = OmegaConf.to_container(config.simulations, resolve=True)
        validation.validate_sim_config(sim_raw, self.logger)

        # Store config references
        self.config: DictConfig      = config
        self.sim: DictConfig         = config.simulations
        self.root_path: Path         = Path(config.paths.root)
        self.solver_name: str        = config.solver_name
        self.bidding_zones: list     = list(config.bidding_zones)

        self._setup_simulation_folders()
        utils.load_plot_config()

        timer.stop()
        self.logger.info(
            "Configuration loaded — simulation: '%s', mode: '%s', zones: %d",
            self.sim.label,
            self.sim.run.mode,
            len(self.bidding_zones),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> None:
        """
        Run the active simulation. Single public entry point called from main.py.

        Args:
            dry_run: If True, skips model execution and export. Useful for
                     validating configuration and data loading only.
        """
        mode  = self.sim.run.mode
        label = self.sim.label

        if mode not in _RUN_MODES:
            raise ValueError(
                f"Unknown run mode '{mode}'. Valid options: {list(_RUN_MODES)}"
            )

        utils.log_section(self.logger, f"RUN: {label}  [{mode}]")
        sim_timer = Timer(self.logger, f"Simulation '{label}'")

        method = getattr(self, _RUN_MODES[mode])
        method(dry_run)

        sim_timer.stop()
        self.logger.info("Simulation '%s' completed successfully\n", label)

    # -----------------------------------------------------------------------
    # Private — run modes
    # -----------------------------------------------------------------------

    def _run_yearly(self, dry_run: bool) -> None:
        """Execute a full-year simulation."""
        self._preprocess()
        self._load_data()
        self._solve(dry_run)
        if not dry_run:
            self._export()

    def _run_rolling_horizon(self, dry_run: bool) -> None:
        """Execute a rolling-horizon simulation over a range of weeks."""
        start_week = self.sim.rolling_horizon.start_week
        end_week   = self.sim.rolling_horizon.end_week

        self._preprocess()

        for week in tqdm(range(start_week, end_week + 1), desc="Rolling horizon"):
            self.logger.info("--- Week %d / %d ---", week, end_week)
            self._load_data(week=week)
            self._solve(dry_run, week=week)
            if not dry_run:
                self._export(week=week)

        if not dry_run:
            self._concatenate_weekly_results(start_week, end_week)

    # -----------------------------------------------------------------------
    # Private — pipeline steps (shared across run modes)
    # -----------------------------------------------------------------------

    def _preprocess(self) -> None:
        """Run data preprocessing (once per simulation)."""
        timer = Timer(self.logger, "Data preprocessing")
        self.data_processor = DataProcessor(
            sim_config=self.sim,
            config=self.config,
            root_path=self.root_path,
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

    def _export(self, week: int | None = None) -> None:
        """Export results to simulations/<label>/results/."""
        label = "Results export" if week is None else f"Results export — week {week}"
        timer = Timer(self.logger, label)
        exporter = DataExporter(
            enlight_model=self.enlight_model,
            sim_config=self.sim,
            root_path=self.root_path,
            logger=self.logger,
            week=week,
        )
        exporter.export_solution()
        timer.stop()

    def _concatenate_weekly_results(self, start_week: int, end_week: int) -> None:
        """Concatenate per-week CSVs into a single file after rolling horizon run."""
        timer = Timer(self.logger, "Concatenating weekly results")
        results_path = self.root_path / "simulations" / self.sim.label / "results"
        weeks = list(range(start_week, end_week + 1))
        utils.combine_simulations_result(
            weeks=weeks,
            result_path=results_path,
            result="electricity_prices",   # extend as needed
        )
        timer.stop()

    # -----------------------------------------------------------------------
    # Private — setup
    # -----------------------------------------------------------------------

    def _setup_simulation_folders(self) -> None:
        """Create output folder structure for the active simulation."""
        for subfolder in ("data", "results"):
            path = self.root_path / "simulations" / self.sim.label / subfolder
            path.mkdir(parents=True, exist_ok=True)