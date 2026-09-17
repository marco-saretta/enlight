import gc
import logging
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

import enlight.utils as utils
from enlight.data_ops import DataExporter, DataLoader, DataPreprocessor
from enlight.model import EnlightModel
from enlight.utils.validation import validate_simulation_config

log = utils.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEPS = ("preprocess", "load", "build", "solve", "export")


def load_config(simulation: str = "default", overrides: Sequence[str] = ()) -> DictConfig:
    """
    Compose the Hydra config without `python main.py`, e.g. in a notebook:

        cfg = load_config("demo1", overrides=["simulations.run.solver=highs"])
        runner = EnlightRunner(cfg)
        runner.preprocess()
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "config"), version_base=None):
        # paths.root normally comes from ${hydra:runtime.cwd}, which only exists under @hydra.main
        return compose("config", overrides=[f"simulations={simulation}", f"paths.root={PROJECT_ROOT}", *overrides])


class EnlightRunner:
    """
    Instantiate the ENLIGHT runner object to execute the pipeline.

    Runner works in 5 steps, each a public method:
    1. preprocess() - raw data into simulation input .csv files
    2. load()       - .csv files into linopy-compatible arrays
    3. build()      - build the market-clearing model
    4. solve()      - solve it
    5. export()     - model results into result .csv files

    run() executes the steps listed in cfg.steps; call the methods directly to
    run only part of the pipeline. In rolling_horizon mode, load to export
    repeat once per week.
    """

    def __init__(self, cfg: DictConfig) -> None:
        utils.setup_logging(log_dir=cfg.paths.log)
        validate_simulation_config(cfg.simulations)

        self.cfg = cfg
        self.data: DataLoader | None = None
        self.model: EnlightModel | None = None

        self._setup_simulation_folders()
        utils.load_plot_config()

        log.info(
            "simulation '%s': mode %s, %d zones, prediction year %d, solver %s",
            cfg.simulations.label,
            cfg.simulations.run.mode,
            len(cfg.simulations.bidding_zones),
            cfg.simulations.run.prediction_year,
            cfg.simulations.run.solver,
        )

    def run(self) -> None:
        """
        Run the steps listed in cfg.steps, in pipeline order.
        """
        start = time.perf_counter()
        steps = set(self.cfg.steps)
        unknown = steps - set(STEPS)
        if unknown:
            raise ValueError(f"Unknown steps {sorted(unknown)}. Valid: {', '.join(STEPS)}")

        if "preprocess" in steps:
            self.preprocess()

        weeks = self._weeks()
        if weeks == [None]:
            self._run_steps(steps)
        else:
            log_file = Path(self.cfg.paths.log).relative_to(self.cfg.paths.root) / "enlight.log"
            log.info("solving %d weeks; each week's step details are only in %s", len(weeks), log_file)
            for week in weeks:
                week_start = time.perf_counter()
                with utils.console_level(logging.WARNING):  # warnings still reach the console
                    self._run_steps(steps, week)
                objective = f", objective {self.model.model.objective.value:.4e}" if "solve" in steps else ""
                log.info("week %d (%d-%d) done in %.1f s%s", week, weeks[0], weeks[-1], time.perf_counter() - week_start, objective)

            if "export" in steps:
                self.join_weekly_results(weeks)

        log.info(
            "simulation '%s' finished in %.1f s",
            self.cfg.simulations.label, time.perf_counter() - start,
        )

    def _run_steps(self, steps: set[str], week: int | None = None) -> None:
        """
        Run load to export for the whole year, or for one week.
        """
        if "load" in steps:
            self.load(week)
        if "build" in steps:
            self.build()
        if "solve" in steps:
            self.solve()
        if "export" in steps:
            self.export(week)

    def preprocess(self) -> None:
        """
        Preprocess the raw data into the simulation input as .csv files
        """
        with utils.stage("preprocess", log):
            DataPreprocessor(self.cfg)  # writes CSVs as a side effect; nothing to keep here

    def load(self, week: int | None = None) -> None:
        """
        Load the preprocessed simulation data (one week of it in rolling_horizon mode).
        Any previously loaded data and model are dropped.
        """
        # Linopy objects reference each other, so an old week's model is only
        # freed when Python's garbage collector runs; without this, memory grows
        # by about one week's model per iteration.
        self.data = self.model = None
        gc.collect()

        with utils.stage("load", log):
            self.data = DataLoader(self.cfg, week=week)

    def build(self) -> None:
        """
        Build the market-clearing model from the loaded data.
        """
        if self.data is None:
            raise RuntimeError("build() needs data: call load() first")
        with utils.stage("build", log):
            self.model = EnlightModel(self.data, self.cfg)

    def solve(self) -> None:
        """
        Solve the built model.
        """
        if self.model is None:
            raise RuntimeError("solve() needs a model: call build() first")
        with utils.stage("solve", log):
            self.model.solve()

    def export(self, week: int | None = None) -> None:
        """
        Export the model results into simulations/<label>/results/ (results/week_<nn>/ for a week).
        """
        if self.model is None or self.model.model.status != "ok":
            raise RuntimeError("export() needs a solved model: call build() and solve() first")
        with utils.stage("export", log):
            DataExporter(self.model, week=week)  # writes CSVs as a side effect; nothing to keep here

    def join_weekly_results(self, weeks: list[int]) -> None:
        """
        Join the weekly results into annual files in simulations/<label>/results/,
        then delete the weekly folders unless rolling_horizon.keep_weekly_results.
        """
        with utils.stage("export", log):
            DataExporter.concatenate(self._results_path(), weeks)
            if not self.cfg.simulations.rolling_horizon.keep_weekly_results:
                for week in weeks:
                    shutil.rmtree(DataExporter.week_path(self._results_path(), week))
                log.info("deleted %d weekly result folders", len(weeks))

    def _weeks(self) -> list[int | None]:
        """
        [None] for a yearly run (one pass), otherwise the weeks to solve one by one.
        """
        if self.cfg.simulations.run.mode == "yearly":
            return [None]
        rh = self.cfg.simulations.rolling_horizon
        return list(range(rh.start_week, rh.end_week + 1))

    def _results_path(self) -> Path:
        return Path(self.cfg.paths.processed) / self.cfg.simulations.label / "results"

    def _setup_simulation_folders(self) -> None:
        """
        Create the data/ and results/ subdirectories under simulations/<label>/.
        """
        # Created up front, in __init__, so every step has somewhere to write
        # regardless of which one runs first.
        root = Path(self.cfg.paths.root)
        label = self.cfg.simulations.label
        for subfolder in ("data", "results"):
            (root / "simulations" / label / subfolder).mkdir(parents=True, exist_ok=True)
