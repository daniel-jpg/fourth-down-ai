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

def late_q4_goal_line_site_notice(
    state: Mapping, eligible_strategies: set[str]
) -> str | None:
    """Small, display-only warning in the empirically sparse Q4 GO/FG region.

    The 2023-24 retrospective factual WP study observed 28 analogous
    receiving-team own-10, leading-1..8, final-five-minute scrimmage
    states in 13 games. They are NOT all necessarily after a failed GO.
    In the even narrower leading-by-4 area: 7 states, 3 games, zero
    HOME-offense examples. Never infer causal GO/FG superiority here.
    """
    if not {"GO FOR IT", "FIELD GOAL"}.issubset(eligible_strategies):
        return None
    try:
        quarter = int(state["qtr"])
        clock = float(state["game_seconds_remaining"])
        yardline = float(state["yardline_100"])
        margin = float(state["score_differential"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (clock, yardline, margin))):
        return None
    if not (
        quarter == 4 and 45 <= clock <= 300
        and .5 <= yardline <= 10
        and -8 <= margin <= -1
    ):
        return None
    return (
        "Q4 site caution: model WP after a failed goal-line GO can vary "
        "substantially with HOME/AWAY assumptions. Analogous 2023–2024 "
        "backed-up, leading 1–8 Q4 possessions covered only 28 play states "
        "from 13 games (not all following GO failures). The GO-versus-FG "
        "edge has substantial uncertainty beyond simulation error."
    )
