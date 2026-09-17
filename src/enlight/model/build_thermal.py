from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_thermal(em: EnlightModel) -> None:
    """
    Thermal bid per unit and hour: a volume up to the unit's capacity, at its
    marginal cost.

    Each unit injects into its own zone: grouping the bids by the units' zone
    labels does the job of a unit-to-zone incidence matrix.
    """
    thermal = em.data.thermal
    if thermal.zone.size == 0:  # no thermal units in the active zones
        return

    thermal_capacity = thermal.capacity         # [MW], (G): input, maximum output of each unit
    thermal_bid_price = thermal.marginal_cost   # [EUR/MWh], (G): input, marginal cost of each unit

    thermal_bid_volume = em.model.add_variables(
        lower=0,
        upper=thermal_capacity,  # 1-D over units, broadcast across hours by linopy
        coords=[em.data.times, thermal_capacity.indexes["G"]],
        name="thermal_bid_volume",
    )  # [MW], (T, G): decision, the volume the market accepts

    em.add_to_power_balance("thermal", thermal_bid_volume.groupby(thermal.zone.rename("Z")).sum())
    em.add_unit_dispatch("thermal", thermal_bid_volume)
    em.add_to_objective(thermal_bid_volume * thermal_bid_price)
