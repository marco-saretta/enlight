from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enlight.model.energy_model import EnlightModel


def build_lines(em: EnlightModel) -> None:
    """
    Flow on each line, limited by its capacity in each direction.

    A positive flow goes from_zone -> to_zone: it is withdrawn from from_zone
    and injected into to_zone. Grouping the flows by those two zone labels does
    the job of a line-to-zone incidence matrix without building one.
    """
    lines = em.data.lines
    if lines.from_zone.size == 0:  # a single zone, or no lines between active zones
        return

    lines_capacity_from_to = lines.capacity_from_to  # [MW], (T, L): input, capacity from_zone -> to_zone
    lines_capacity_to_from = lines.capacity_to_from  # [MW], (T, L): input, capacity to_zone -> from_zone

    lines_flow = em.model.add_variables(
        lower=-lines_capacity_to_from,
        upper=lines_capacity_from_to,
        name="lines_flow",
    )  # [MW], (T, L): decision, positive from_zone -> to_zone

    em.add_to_power_balance("lines", -lines_flow.groupby(lines.from_zone.rename("Z")).sum())
    em.add_to_power_balance("lines", lines_flow.groupby(lines.to_zone.rename("Z")).sum())
