from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_thermal(em: EnlightModel) -> None:
    """
    Thermal offer per unit, up to its capacity, at its marginal cost.

    Each unit injects into its own zone: grouping the offers by the units'
    zone labels does the job of a unit-to-zone incidence matrix.
    """
    thermal = em.data.thermal
    if thermal.zone.size == 0:  # no thermal units in the active zones
        return

    offer = em.model.add_variables(
        lower=0,
        upper=thermal.capacity,  # 1-D over units, broadcast across hours by linopy
        coords=[em.data.times, thermal.capacity.indexes["G"]],
        name="thermal_offer",
    )  # [MW], (T, G)

    em.add_to_power_balance("thermal", offer.groupby(thermal.zone.rename("Z")).sum())
    em.add_to_objective(offer * thermal.marginal_cost)
