from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_hydro_ror(em: EnlightModel) -> None:
    """
    Run-of-river hydro bid per zone and hour: a volume up to the run-of-river
    potential, at a fixed bid price.
    """
    hydro_ror_potential = em.data.hydro_ror.production  # [MW], (T, Z): input, what the river flow could produce
    hydro_ror_bid_price = em.data.hydro_ror.bid_price   # [EUR/MWh]: input, price asked for each MWh

    hydro_ror_bid_volume = em.model.add_variables(
        lower=0, upper=hydro_ror_potential, name="hydro_ror_bid_volume",
    )  # [MW], (T, Z): decision, the volume the market accepts

    em.add_to_power_balance("hydro_ror", hydro_ror_bid_volume)
    em.add_curtailment("hydro_ror", hydro_ror_potential, hydro_ror_bid_volume)
    em.add_to_objective(hydro_ror_bid_volume * hydro_ror_bid_price)
