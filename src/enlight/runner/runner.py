from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm

from enlight.data_ops import DataPreprocessor, DataLoader, DataExporter
from enlight.model import EnlightModel
import enlight.utils as utils
from enlight.utils import Timer
from enlight.utils.validation import validate_simulation_config

log = utils.get_logger(__name__)


class EnlightRunner:
    """
    Instantiate the ENLIGHT runner object to execute the pipeline.

    Runner works in 4 steps:
    1. Preprocess - raw data into simulation input .csv files
    2. Load       - .csv files into linopy-compatible arrays
    3. Solve      - build and solve the market-clearing model
    4. Export     - model results into result .csv files
    """

    def __init__(self, cfg: DictConfig) -> None:
        utils.setup_logging(log_dir=cfg.paths.log)
        utils.log_section(log, "ENLIGHT initialisation")

        self.cfg = cfg

        self._setup_simulation_folders()
        utils.load_plot_config()

        log.info(
            "Simulation: '%s'  mode: %s  zones: %d",
            cfg.simulations.label,
            cfg.simulations.run.mode,
            len(cfg.simulations.bidding_zones),
        )

    def run(self, dry_run: bool = False) -> None:
        """
        Run the active simulation end-to-end.
        """

        mode = self.cfg.simulations.run.mode
        label = self.cfg.simulations.label

        validate_simulation_config(self.cfg.simulations)
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
        """
        Run a single full-year optimisation (8760 h).
        """
        self._preprocess()
        self._load_data()
        self._solve(dry_run)
        if not dry_run:
            self._export()

    def _run_rolling_horizon(self, dry_run: bool) -> None:
        """
        Run a week-by-week optimisation, concatenating results after the loop.
        """
        rh = self.cfg.simulations.rolling_horizon
        self._preprocess()  # raw data is scenario-wide, so this runs once, not per week

        for week in tqdm(range(rh.start_week, rh.end_week + 1), desc="Rolling horizon"):
            log.info("Week %d / %d", week, rh.end_week)
            self._load_data(week=week)
            self._solve(dry_run, week=week)
            if not dry_run:
                self._export(week=week)

        if not dry_run:
            self._concatenate_weekly_results(rh.start_week, rh.end_week)

    def _preprocess(self) -> None:
        """
        Preprocess the raw data into the simulation input as .csv files
        """
        DataPreprocessor(self.cfg)  # writes CSVs as a side effect; nothing to keep here

    def _load_data(self, week: int | None = None) -> None:
        """
        Load the preprocessed simulation data into linopy-compatible arrays.
        """
        self.data = DataLoader(self.cfg)    # Loads simultaiton data into self.data 

    def _solve(self, dry_run: bool, week: int | None = None) -> None:
        """
        Build and solve the market-clearing model; skipped when dry_run=True.
        """
        if dry_run:
            log.info("Dry run — skipping solve.")
            return
        # TODO: not yet implemented, awaits _load_data

    def _export(self, week: int | None = None) -> None:
        """
        Export the model results into simulations/<label>/results/.
        """
        pass  # TODO: not yet implemented, awaits _solve

    def _concatenate_weekly_results(self, start_week: int, end_week: int) -> None:
        """
        Merge the per-week result CSVs into a single annual file.
        """
        pass  # TODO: not yet implemented; only matters once rolling_horizon produces per-week results

    def _setup_simulation_folders(self) -> None:
        """
        Create the data/ and results/ subdirectories under simulations/<label>/.
        """
        # Created up front, in __init__, so every stage has somewhere to write
        # regardless of which one runs first.
        root = Path(self.cfg.paths.root)
        label = self.cfg.simulations.label
        for subfolder in ("data", "results"):
            (root / "simulations" / label / subfolder).mkdir(parents=True, exist_ok=True)
