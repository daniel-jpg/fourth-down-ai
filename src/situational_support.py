"""Frozen-development support notice for unusually deep OWN territory.

This is a display-only research qualification. No action is removed,
no model weights/probabilities are changed, and no future outcome data is used.

The recorded 2014-2022 training subset of modeling_split contains
one normal 4th-and-2/3 GO at own 1-10 (zero inside own 1-5).
See reports/issue9_deep_territory_site_audit.md for counts and methods.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def deep_own_short_go_notice(
    state: Mapping, selected_strategy: str
) -> str | None:
    """Return a narrow caution only for the unsupported GO region.

    yardline_100 is measured from the opposing end zone: own 10
    corresponds to yardline_100 >= 90. This does not change GO eligibility.
    """
    if selected_strategy != "GO FOR IT":
        return None
    quarter = int(state["qtr"])
    yardline = float(state["yardline_100"])
    distance = float(state["ydstogo"])
    if not (math.isfinite(yardline) and math.isfinite(distance)):
        return None
    if not (1 <= quarter <= 4 and yardline >= 90.0
            and 1.5 <= distance <= 3.5):
        return None
    return (
        "**Limited historical GO support.** For fourth-and-2/3 inside "
        "a team's own 10, the 2014–2022 development set contains only "
        "one normal GO attempt (none inside the own 5). The GO-versus-PUNT "
        "edge relies on extrapolation and is not a dependable coaching "
        "advantage."
    )

def supported_recommendation_display(
    state: Mapping, selected_strategy: str
) -> str:
    """Abstain from a supported coaching pick when the selected GO lacks data.

    This is a DISPLAY-only reliability decision, not a PUNT substitution:
    unchanged engine action rows and probabilities remain inspectable.
    Historical development has 0 comparable own-5 and only 1 own-10
    fourth-and-2/3 normal GO training examples.
    """
    if deep_own_short_go_notice(state, selected_strategy):
        return "INSUFFICIENT DATA"
    return selected_strategy
