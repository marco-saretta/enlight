from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_wind_onshore(em: EnlightModel) -> None:
    """
    Onshore wind bid per zone and hour: a volume up to the wind potential, at a
    fixed bid price.
    """
    wind_onshore_potential = em.data.wind_onshore.production  # [MW], (T, Z): input, what the wind could produce
    wind_onshore_bid_price = em.data.wind_onshore.bid_price   # [EUR/MWh]: input, price asked for each MWh

    wind_onshore_bid_volume = em.model.add_variables(
        lower=0, upper=wind_onshore_potential, name="wind_onshore_bid_volume",
    )  # [MW], (T, Z): decision, the volume the market accepts

    em.add_to_power_balance("wind_onshore", wind_onshore_bid_volume)
    em.add_curtailment("wind_onshore", wind_onshore_potential, wind_onshore_bid_volume)
    em.add_to_objective(wind_onshore_bid_volume * wind_onshore_bid_price)
