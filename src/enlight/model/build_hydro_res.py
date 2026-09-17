from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_hydro_res(em: EnlightModel) -> None:
    """
    Reservoir hydro bid per unit and hour: a volume up to the unit's capacity,
    at its production cost.

    All reservoirs in a zone share that zone's weekly energy budget: their
    total production in a week cannot exceed the energy available that week.
    A yearly run adds one such constraint per zone and week (52 per zone); a
    rolling-horizon run loads a single week, so one per zone.
    """
    hydro_res = em.data.hydro_res
    if hydro_res.zone.size == 0:  # no reservoir units in the active zones
        return

    hydro_res_capacity = hydro_res.capacity             # [MW], (G): input, maximum output of each unit
    hydro_res_bid_price = hydro_res.marginal_cost       # [EUR/MWh], (G): input, production cost of each unit
    hydro_res_weekly_energy = hydro_res.weekly_energy   # [MWh], (W, Z): input, energy budget per zone and week

    hydro_res_bid_volume = em.model.add_variables(
        lower=0,
        upper=hydro_res_capacity,  # 1-D over units, broadcast across hours by linopy
        coords=[em.data.times, hydro_res_capacity.indexes["G"]],
        name="hydro_res_bid_volume",
    )  # [MW], (T, G): decision, the volume the market accepts

    hydro_res_zone_volume = hydro_res_bid_volume.groupby(hydro_res.zone.rename("Z")).sum()  # [MW], (T, Z)
    em.add_to_power_balance("hydro_res", hydro_res_zone_volume)
    em.add_to_objective(hydro_res_bid_volume * hydro_res_bid_price)

    hydro_res_weekly_volume = hydro_res_zone_volume.groupby(em.data.week_of_hour).sum()  # [MWh] with hourly steps, (W, Z)
    hydro_res_budget = hydro_res_weekly_energy.sel(
        W=hydro_res_weekly_volume.indexes["W"], Z=hydro_res_weekly_volume.indexes["Z"]
    )
    em.model.add_constraints(hydro_res_weekly_volume <= hydro_res_budget, name="hydro_res_weekly_energy")
