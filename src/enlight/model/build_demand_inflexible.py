from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_demand_inflexible(em: EnlightModel) -> None:
    """
    Inflexible demand bid per category, zone and hour: a volume up to the load,
    at the value of lost load.

    Demand that isn't served is load shedding: it costs voll in lost benefit.
    """
    for category, demand_inflexible in em.data.demand_inflexible.items():
        demand_inflexible_load = demand_inflexible.demand     # [MW], (T, Z): input, what consumers want
        demand_inflexible_bid_price = demand_inflexible.voll  # [EUR/MWh]: input, value of lost load

        demand_inflexible_bid_volume = em.model.add_variables(
            lower=0, upper=demand_inflexible_load, name=f"{category}_bid_volume",
        )  # [MW], (T, Z): decision, the volume the market serves

        em.add_to_power_balance(category, -demand_inflexible_bid_volume)
        em.add_to_objective(-demand_inflexible_bid_price * demand_inflexible_bid_volume)
