from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

import enlight.utils as utils

if TYPE_CHECKING:
    from enlight.model import EnlightModel

log = utils.get_logger(__name__)

MW_DECIMALS = 3  # hourly MW files are rounded to 1 kW; the per-unit files are large


class DataExporter:
    """
    Instantiate the exporter and write the solved model's results to
    simulations/<label>/results/ (results/week_<nn>/ in rolling_horizon mode):

    1. electricity_prices.csv - zonal price per hour [EUR/MWh]
    2. dispatch.csv           - hourly MW per technology and zone;
                                + injection, - withdrawal, lines = net import
    3. curtailment.csv        - hourly MW each renewable could have produced but
                                did not (potential = dispatched + curtailed)
    4. <tech>_dispatch.csv    - hourly MW of every unit, for unit-based
                                technologies such as thermal
    5. energy_balance.csv     - dispatch per technology and zone [TWh], with a
                                <tech>_curtailed row under each renewable
    6. capture_prices.csv     - production-weighted price per generating
                                technology and zone, plus baseload [EUR/MWh]

    Tables 5 and 6 are derived from 1-3, so concatenate() can rebuild them for a
    whole year from the weekly results.
    """

    def __init__(self, em: EnlightModel, week: int | None = None) -> None:
        results_path = Path(em.cfg.paths.processed) / em.cfg.simulations.label / "results"
        if week is not None:
            results_path = self.week_path(results_path, week)

        prices = em.power_balance.dual.to_pandas()  # (T x Z)
        unit_dispatch = {label: variable.solution.to_pandas() for label, variable in em.unit_dispatch.items()}
        self._write(
            results_path,
            prices,
            dispatch=self._dispatch(em, zones=prices.columns),
            curtailment=self._curtailment(em, zones=prices.columns),
            unit_dispatch=unit_dispatch,
        )

    @staticmethod
    def week_path(results_path: Path, week: int) -> Path:
        """
        Folder holding one week's results in rolling_horizon mode.
        """
        return results_path / f"week_{week:02d}"

    @classmethod
    def concatenate(cls, results_path: Path, weeks: list[int]) -> None:
        """
        Join the weekly hourly results into annual files and rebuild the annual
        energy balance and capture prices from them.
        """
        week_paths = [cls.week_path(results_path, week) for week in weeks]

        def join(filename: str, **read_csv_kwargs) -> pd.DataFrame:
            return pd.concat(pd.read_csv(p / filename, index_col=0, **read_csv_kwargs) for p in week_paths)

        prices = join("electricity_prices.csv")
        log.info("joining %d weeks (%d hours)", len(weeks), len(prices))
        cls._write(
            results_path,
            prices,
            dispatch=join("dispatch.csv", header=[0, 1]),
            curtailment=join("curtailment.csv", header=[0, 1]),
            unit_dispatch={
                path.name.removesuffix("_dispatch.csv"): join(path.name)
                for path in sorted(week_paths[0].glob("*_dispatch.csv"))
            },
        )

    @staticmethod
    def _dispatch(em: EnlightModel, zones: pd.Index) -> pd.DataFrame:
        """
        Hourly MW of every power-balance term, summed per label: T x (term, zone).
        """
        by_label: dict[str, pd.DataFrame] = {}
        for label, expr in em.power_balance_terms:
            # a term may cover only some zones (e.g. no line starts there)
            mw = expr.solution.to_pandas().reindex(columns=zones, fill_value=0)
            by_label[label] = by_label[label] + mw if label in by_label else mw
        return pd.concat(by_label, axis=1, names=["term", "zone"])

    @staticmethod
    def _curtailment(em: EnlightModel, zones: pd.Index) -> pd.DataFrame:
        """
        Hourly MW each registered renewable could have produced but did not: T x (term, zone).
        """
        curtailed = {
            # clip: solver tolerances can leave -1e-9 where nothing is curtailed
            label: (potential - dispatched.solution).to_pandas().reindex(columns=zones, fill_value=0).clip(lower=0)
            for label, (potential, dispatched) in em.curtailment_terms.items()
        }
        return pd.concat(curtailed, axis=1, names=["term", "zone"])

    @classmethod
    def _write(
        cls,
        results_path: Path,
        prices: pd.DataFrame,
        dispatch: pd.DataFrame,
        curtailment: pd.DataFrame,
        unit_dispatch: dict[str, pd.DataFrame],
    ) -> None:
        """
        Write the hourly results and the tables derived from them.
        """
        cls._save(results_path, "electricity_prices", prices)
        cls._save(results_path, "dispatch", dispatch.round(MW_DECIMALS))
        cls._save(results_path, "curtailment", curtailment.round(MW_DECIMALS))
        for label, df in unit_dispatch.items():
            cls._save(results_path, f"{label}_dispatch", df.round(MW_DECIMALS))
        cls._save(results_path, "energy_balance", cls._energy_balance(dispatch, curtailment, zones=prices.columns))
        cls._save(results_path, "capture_prices", cls._capture_prices(dispatch, prices))

    @staticmethod
    def _energy_balance(dispatch: pd.DataFrame, curtailment: pd.DataFrame, zones: pd.Index) -> pd.DataFrame:
        """
        TWh per technology and zone, with each renewable's curtailment on the row
        below it. The dispatch rows sum to zero per zone; curtailed rows are
        energy that was not produced, so they are not part of that sum.
        """
        dispatched = (dispatch.sum() / 1e6).unstack("zone").reindex(columns=zones)
        curtailed = (curtailment.sum() / 1e6).unstack("zone").reindex(columns=zones)

        rows = []
        for term in dispatch.columns.unique("term"):
            rows.append(dispatched.loc[[term]])
            if term in curtailed.index:
                rows.append(curtailed.loc[[term]].rename(index={term: f"{term}_curtailed"}))
        energy = pd.concat(rows)
        energy["total"] = energy.sum(axis=1)
        return energy

    @staticmethod
    def _capture_prices(dispatch: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Production-weighted price for every term that only ever injects
        (generators); NaN where a technology produces nothing in a zone.
        """
        capture = {
            term: (dispatch[term] * prices).sum() / dispatch[term].sum()
            for term in dispatch.columns.unique("term")
            if (dispatch[term] >= -1e-6).all().all()
        }
        capture["baseload"] = prices.mean()
        return pd.DataFrame(capture).T[prices.columns]

    @staticmethod
    def _save(results_path: Path, name: str, df: pd.DataFrame) -> None:
        """
        Write one result table to the results folder.
        """
        utils.save_data(df, f"{name}.csv", output_dir=results_path)
        log.info("wrote %s.csv (%d rows x %d columns)", name, *df.shape)
