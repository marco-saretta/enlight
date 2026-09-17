from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_hydro_ror(em: EnlightModel) -> None:
    """
    Run-of-river hydro offer per zone, up to the available production, at its bid price.
    """
    vre = em.data.hydro_ror

    offer = em.model.add_variables(lower=0, upper=vre.production, name="hydro_ror_offer")  # [MW], (T, Z)

    em.add_to_power_balance("hydro_ror", offer)
    em.add_to_objective(offer * vre.bid_price)
