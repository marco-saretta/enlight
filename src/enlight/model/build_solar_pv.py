from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_solar_pv(em: EnlightModel) -> None:
    """
    Solar PV bid per zone and hour: a volume up to the solar potential, at a
    fixed bid price.
    """
    solar_pv_potential = em.data.solar_pv.production  # [MW], (T, Z): input, what the sun could produce
    solar_pv_bid_price = em.data.solar_pv.bid_price   # [EUR/MWh]: input, price asked for each MWh

    solar_pv_bid_volume = em.model.add_variables(
        lower=0, upper=solar_pv_potential, name="solar_pv_bid_volume",
    )  # [MW], (T, Z): decision, the volume the market accepts

    em.add_to_power_balance("solar_pv", solar_pv_bid_volume)
    em.add_curtailment("solar_pv", solar_pv_potential, solar_pv_bid_volume)
    em.add_to_objective(solar_pv_bid_volume * solar_pv_bid_price)
