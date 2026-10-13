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
    if deep_own_long_go_notice(state, selected_strategy):
        return "INSUFFICIENT DATA"
    if late_q4_trailing_punt_notice(state, selected_strategy):
        return "INSUFFICIENT DATA"
    # Do not advertise a supported field-goal recommendation when there
    # are no comparable real post-made-FG kickoffs in the final two minutes.
    if (selected_strategy == "FIELD GOAL"
            and late_q4_trailing_after_fg_notice(state, {"FIELD GOAL"})):
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


def late_q4_trailing_after_fg_notice(
    state: Mapping, eligible_strategies: set[str]
) -> str | None:
    """Display-only guard for made-FG-then-still-behind-by-one extrapolation.

    Issue #12 counted every first post-made-FG REG Q4 standard-35 kickoff
    within 120 seconds during 2014-2025, including announced onsides:
    ZERO had a receiving team leading by exactly one point.
    This is not an estimate of onside recovery or a new WP adjustment.
    """
    if "FIELD GOAL" not in eligible_strategies:
        return None
    try:
        quarter = int(state["qtr"])
        clock = float(state["game_seconds_remaining"])
        margin = float(state["score_differential"])
    except (KeyError, ValueError, TypeError):
        return None
    if not (math.isfinite(clock) and math.isfinite(margin)):
        return None
    if not (quarter == 4 and 0 < clock <= 120 and margin == -4.):
        return None
    return (
        "**Limited late-game field-goal support.** A made field goal would "
        "still leave this offense trailing by one point. A read-only "
        "2014–2025 regular-season audit found **zero** qualifying standard-35 "
        "post-made-field-goal kickoffs in the final two minutes with the "
        "receiving team leading by exactly one point. The frozen WP "
        "estimator can also give nonterminal win probabilities when the "
        "regulation clock has expired. The displayed FIELD GOAL probability "
        "is an **unvalidated model extrapolation**, NOT a demonstrated chance "
        "to win or an empirically supported reason to attempt a field goal."
    )



def deep_own_long_go_notice(
    state: Mapping, selected_strategy: str
) -> str | None:
    """Abstain, but leave raw model WP intact, on long GO from own goal line.

    Reproduced in observed 2025 Week 3 ARI-SF (Q4 14:11, fourth-and-11,
    own 1): the frozen model puts GO 37.23% and PUNT 36.51% at seed 43.
    A small modeled edge is not proof GO is safe. This warning does NOT
    assert a specific historical GO count or replace the modeled values.
    """
    if selected_strategy != "GO FOR IT":
        return None
    try:
        quarter = int(state["qtr"])
        yardline = float(state["yardline_100"])
        distance = float(state["ydstogo"])
    except (TypeError, ValueError, KeyError):
        return None
    if not all(math.isfinite(x) for x in (yardline, distance)):
        return None
    if not (1 <= quarter <= 4 and yardline >= 95 and distance >= 7):
        return None
    return (
        "**Insufficient support for a long GO deep in own territory.** "
        "Fourth-and-7+ at or inside the offense's own 5 is an unusually "
        "high-consequence decision. A 2025 game-state audit reproduced a "
        "model GO preference on fourth-and-11 at its own 1 with only a "
        "0.72 percentage-point modeled edge over PUNT. That is not "
        "validated coaching evidence. No alternative is automatically "
        "substituted; compare the unmodified values below with caution."
    )


def late_q4_trailing_punt_notice(
    state: Mapping, selected_strategy: str
) -> str | None:
    """Unvalidated final-minute punt when trailing with no timeouts.

    The production model still returns its original Q4 PUNT/GO probabilities;
    do not promote a narrow simulated PUNT lead into live-game advice.
    """
    if selected_strategy != "PUNT":
        return None
    try:
        quarter = int(state["qtr"])
        clock = float(state["game_seconds_remaining"])
        margin = float(state["score_differential"])
        to = float(state["posteam_timeouts_remaining"])
        yardline = float(state["yardline_100"])
    except (TypeError, ValueError, KeyError):
        return None
    if not all(math.isfinite(x) for x in (clock, margin, to, yardline)):
        return None
    if not (
        quarter == 4 and 0 < clock <= 60 and margin < 0
        and to == 0 and yardline >= 50
    ):
        return None
    return (
        "**Insufficient support for a final-minute trailing PUNT.** "
        "Punting surrenders possession while trailing with no offensive "
        "timeouts and at most one minute remaining. The frozen simulator "
        "can still assign a slightly higher modeled win probability to "
        "PUNT; that gap is not validated coaching evidence. It does not "
        "account reliably for the very short recovery-and-scoring window. "
        "The original numerical estimates remain shown for audit, but "
        "this is **not a supported PUNT recommendation**."
    )
