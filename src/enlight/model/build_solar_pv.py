from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_solar_pv(em: EnlightModel) -> None:
    """
    Solar PV offer per zone, up to the available production, at its bid price.
    """
    vre = em.data.solar_pv

    offer = em.model.add_variables(lower=0, upper=vre.production, name="solar_pv_offer")  # [MW], (T, Z)

    em.add_to_power_balance("solar_pv", offer)
    em.add_to_objective(offer * vre.bid_price)
