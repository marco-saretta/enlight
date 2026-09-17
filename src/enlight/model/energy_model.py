from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import linopy
from linopy.expressions import merge
from omegaconf import DictConfig

import enlight.utils as utils
from enlight.model.build_demand_inflexible import build_demand_inflexible
from enlight.model.build_hydro_res import build_hydro_res
from enlight.model.build_hydro_ror import build_hydro_ror
from enlight.model.build_lines import build_lines
from enlight.model.build_solar_pv import build_solar_pv
from enlight.model.build_thermal import build_thermal
from enlight.model.build_wind_offshore import build_wind_offshore
from enlight.model.build_wind_onshore import build_wind_onshore

if TYPE_CHECKING:
    import xarray as xr

    from enlight.data_ops import DataLoader

log = utils.get_logger(__name__)

# Solver options that keep the solver's own progress output off the console;
# it goes to results/solver.log instead.
QUIET_SOLVER_OPTIONS = {
    "highs": {"log_to_console": False},
    "gurobi": {"LogToConsole": 0},
}

# Each step adds its own variables to the model and registers its power-balance
# and objective terms. To add a technology: write model/build_<tech>.py and list
# its function here.
BUILD_STEPS = (
    build_wind_onshore,
    build_wind_offshore,
    build_solar_pv,
    build_hydro_ror,
    build_hydro_res,
    build_thermal,
    build_demand_inflexible,
    build_lines,
)


class EnlightModel:
    """
    Instantiate the market-clearing model from one build_<tech>.py script per technology.

    Each script adds its own variables, then registers terms
    for the two parts every technology shares:
    1. Power balance - injections (+) and withdrawals (-) per zone and hour
    2. Objective     - costs (+) and consumer benefits (-), minimised

    The power-balance constraint and the objective are added once, after
    every script has run.
    """

    def __init__(self, data: DataLoader, cfg: DictConfig) -> None:
        self.data = data
        self.cfg = cfg
        self.model = linopy.Model()

        # (label, term) pairs; the labels let DataExporter report energy per technology
        self.power_balance_terms: list[tuple[str, linopy.LinearExpression]] = []
        self._objective_terms: list[linopy.LinearExpression] = []

        # Extra results for DataExporter, registered by the build scripts
        self.curtailment_terms: dict[str, tuple[xr.DataArray, linopy.Variable]] = {}
        self.unit_dispatch: dict[str, linopy.Variable] = {}

        for build_step in BUILD_STEPS:
            build_step(self)

        self._add_power_balance()
        self._add_objective()

        log.info("%s variables, %s constraints", f"{self.model.nvars:,}", f"{self.model.ncons:,}")

    def add_to_power_balance(self, label: str, expr: linopy.Variable | linopy.LinearExpression) -> None:
        """
        Register a zonal injection (+) or withdrawal (-) with dims (T, Z), under
        a label (usually the technology) used in the exported energy balance.
        """
        # merge() only accepts expressions, not raw variables
        if isinstance(expr, linopy.Variable):
            expr = expr.to_linexpr()
        self.power_balance_terms.append((label, expr))

    def add_curtailment(self, label: str, potential: xr.DataArray, dispatched: linopy.Variable) -> None:
        """
        Register a renewable whose unused potential is exported as curtailment:
        potential [MW] minus dispatched [MW], both with dims (T, Z).
        """
        self.curtailment_terms[label] = (potential, dispatched)

    def add_unit_dispatch(self, label: str, dispatch: linopy.Variable) -> None:
        """
        Register a per-unit variable with dims (T, unit), exported in full as <label>_dispatch.csv.
        """
        self.unit_dispatch[label] = dispatch

    def add_to_objective(self, expr: linopy.Variable | linopy.LinearExpression) -> None:
        """
        Register a cost (+) or benefit (-) term; it is summed over all its dims.
        """
        self._objective_terms.append(expr.sum())

    def solve(self) -> None:
        """
        Solve the model with the solver set in run.solver.
        """
        solver = self.cfg.simulations.run.solver
        log_file = Path(self.cfg.paths.processed) / self.cfg.simulations.label / "results" / "solver.log"
        log.info("solving with %s (solver output in %s)", solver, log_file.relative_to(self.cfg.paths.root))

        # io_api="direct" hands the model to the solver in memory; the default
        # ("lp") writes and re-parses a text file, doubling peak memory.
        status, condition = self.model.solve(
            solver_name=solver,
            io_api="direct",
            log_fn=log_file,
            **QUIET_SOLVER_OPTIONS.get(solver, {}),
        )
        if status != "ok":
            raise RuntimeError(f"{solver} did not solve the model: status={status}, condition={condition}")
        log.info("%s, objective %.4e", condition, self.model.objective.value)

        # The solution and duals are already copied into the linopy model, but
        # linopy keeps the solver object attached, and it holds its own copy of
        # the whole problem (2.5 GB for the per-unit year).
        self.model.solver_model = None

    def _add_power_balance(self) -> None:
        """
        Add generation - demand - net export = 0 for every zone and hour.
        """
        # join="outer": a term may cover only some zones (e.g. no line starts in
        # a zone); missing zones just contribute nothing to that term.
        lhs = merge([expr for _, expr in self.power_balance_terms], join="outer")
        self.power_balance = self.model.add_constraints(lhs == 0, name="power_balance")

    def _add_objective(self) -> None:
        """
        Minimise total cost minus consumer benefit (i.e. maximise social welfare).
        """
        self.model.add_objective(merge(self._objective_terms), sense="min")
