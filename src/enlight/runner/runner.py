from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from enlight.data_ops import DataProcessor, DataLoader, DataExporter
from enlight.model import EnlightModel
import enlight.utils as utils
from enlight.utils import Timer
from enlight.utils.validation import validate_simulation_config

log = utils.get_logger(__name__)


class EnlightRunner:
    """Workflow manager for the ENLIGHT simulation pipeline.

    Stages run in order:
      1. preprocess  — raw data  → simulations/<label>/data/
      2. load_data   — CSVs      → linopy-compatible arrays
      3. solve       — market-clearing optimisation (Gurobi / HiGHS)
      4. export      — results   → simulations/<label>/results/
    """

    def __init__(self, config: DictConfig) -> None:
        utils.setup_logging(log_dir=config.paths.log)
        utils.log_section(log, "ENLIGHT initialisation")

        self.config = config

        self._setup_simulation_folders()
        utils.load_plot_config()

        log.info(
            "Simulation: '%s'  mode: %s  zones: %d",
            config.simulations.label,
            config.simulations.run.mode,
            len(config.bidding_zones),
        )

    def run(self, dry_run: bool = False) -> None:
        """Run the active simulation end-to-end."""
        mode  = self.config.simulations.run.mode
        label = self.config.simulations.label

        validate_simulation_config(self.config.simulations)
        utils.log_section(log, f"RUN: {label}  [{mode}]")
        timer = Timer(log, f"Simulation '{label}'")

        if mode == "yearly":
            self._run_yearly(dry_run)
        elif mode == "rolling_horizon":
            self._run_rolling_horizon(dry_run)
        else:
            raise ValueError(f"Unknown run mode '{mode}'. Valid: yearly | rolling_horizon")

        timer.stop()

    def _run_yearly(self, dry_run: bool) -> None:
        """Single full-year optimisation (8760 h)."""
        self._preprocess()
        self._load_data()
        self._solve(dry_run)
        if not dry_run:
            self._export()

    def _run_rolling_horizon(self, dry_run: bool) -> None:
        """Week-by-week optimisation; results concatenated after the loop."""
        rh = self.config.simulations.rolling_horizon
        self._preprocess()

        for week in tqdm(range(rh.start_week, rh.end_week + 1), desc="Rolling horizon"):
            log.info("Week %d / %d", week, rh.end_week)
            self._load_data(week=week)
            self._solve(dry_run, week=week)
            if not dry_run:
                self._export(week=week)

        if not dry_run:
            self._concatenate_weekly_results(rh.start_week, rh.end_week)

    def _preprocess(self) -> None:
        """data/ → simulations/<label>/data/ (runs once per scenario)."""
        pass

    def _load_data(self, week: int | None = None) -> None:
        """simulations/<label>/data/ → linopy-compatible arrays."""
        pass

    def _solve(self, dry_run: bool, week: int | None = None) -> None:
        """Build and solve the market-clearing model; skipped when dry_run=True."""
        if dry_run:
            log.info("Dry run — skipping solve.")
            return

    def _export(self, week: int | None = None) -> None:
        """Model results → simulations/<label>/results/."""
        pass

    def _concatenate_weekly_results(self, start_week: int, end_week: int) -> None:
        """Merge per-week result CSVs into a single annual file."""
        pass

    def _setup_simulation_folders(self) -> None:
        """Create data/ and results/ subdirectories under simulations/<label>/."""
        root  = Path(self.config.paths.root)
        label = self.config.simulations.label
        for subfolder in ("data", "results"):
            (root / "simulations" / label / subfolder).mkdir(parents=True, exist_ok=True)
