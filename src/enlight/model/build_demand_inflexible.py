from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_demand_inflexible(em: EnlightModel) -> None:
    """
    Inflexible demand bid per zone, up to the hourly demand, at the value of lost load.

    Demand that isn't served is load shedding: it costs voll in lost benefit.
    """
    for category, inflex in em.data.demand_inflexible.items():
        bid = em.model.add_variables(lower=0, upper=inflex.demand, name=f"{category}_bid")  # [MW], (T, Z)

        em.add_to_power_balance(category, -bid)
        em.add_to_objective(-inflex.voll * bid)
