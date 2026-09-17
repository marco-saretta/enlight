from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

import enlight.utils as utils

if TYPE_CHECKING:
    from enlight.model import EnlightModel

log = utils.get_logger(__name__)


class DataExporter:
    """
    Instantiate the exporter and write the solved model's results to
    simulations/<label>/results/ (results/week_<nn>/ in rolling_horizon mode):

    1. electricity_prices.csv - zonal price per hour [EUR/MWh]
    2. dispatch.csv           - hourly MW per power-balance term and zone;
                                + injection, - withdrawal, lines = net import
    3. energy_balance.csv     - dispatch summed over the hours [TWh]
    4. capture_prices.csv     - production-weighted price per generating
                                technology and zone, plus baseload [EUR/MWh]

    Tables 3 and 4 are derived from 1 and 2 only, so concatenate() can rebuild
    them for a whole year from the weekly results.
    """

    def __init__(self, em: EnlightModel, week: int | None = None) -> None:
        results_path = Path(em.cfg.paths.processed) / em.cfg.simulations.label / "results"
        if week is not None:
            results_path = self.week_path(results_path, week)

        prices = em.power_balance.dual.to_pandas()  # (T x Z)
        self._write(results_path, prices, self._dispatch(em, zones=prices.columns))

    @staticmethod
    def week_path(results_path: Path, week: int) -> Path:
        """
        Folder holding one week's results in rolling_horizon mode.
        """
        return results_path / f"week_{week:02d}"

    @classmethod
    def concatenate(cls, results_path: Path, weeks: list[int]) -> None:
        """
        Join the weekly prices and dispatch into annual files and rebuild the
        annual energy balance and capture prices from them.
        """
        week_paths = [cls.week_path(results_path, week) for week in weeks]
        prices = pd.concat(pd.read_csv(p / "electricity_prices.csv", index_col=0) for p in week_paths)
        dispatch = pd.concat(pd.read_csv(p / "dispatch.csv", index_col=0, header=[0, 1]) for p in week_paths)
        log.info("joining %d weeks (%d hours)", len(weeks), len(prices))
        cls._write(results_path, prices, dispatch)

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

    @classmethod
    def _write(cls, results_path: Path, prices: pd.DataFrame, dispatch: pd.DataFrame) -> None:
        """
        Write prices and dispatch, and the tables derived from them.
        """
        energy = (dispatch.sum() / 1e6).unstack("zone")  # [TWh]
        energy = energy.loc[dispatch.columns.unique("term"), prices.columns]  # keep registration and zone order
        energy["total"] = energy.sum(axis=1)

        cls._save(results_path, "electricity_prices", prices)
        cls._save(results_path, "dispatch", dispatch)
        cls._save(results_path, "energy_balance", energy)
        cls._save(results_path, "capture_prices", cls._capture_prices(dispatch, prices))

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
