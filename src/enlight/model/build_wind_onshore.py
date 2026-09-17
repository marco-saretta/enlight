from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_wind_onshore(em: EnlightModel) -> None:
    """
    Onshore wind offer per zone, up to the available production, at its bid price.
    """
    wind = em.data.wind_onshore

    offer = em.model.add_variables(lower=0, upper=wind.production, name="wind_onshore_offer")  # [MW], (T, Z)

    em.add_to_power_balance("wind_onshore", offer)
    em.add_to_objective(offer * wind.bid_price)
