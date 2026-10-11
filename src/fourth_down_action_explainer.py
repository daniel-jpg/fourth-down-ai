"""Human-readable immediate fourth-down outcome assumptions.

The production engine already returns action-level estimated winning chances,
Monte Carlo standard errors and a structured "detail" string containing
GO conversion, FG made/miss/block, and punt regular/block/broken rates.
This module ONLY parses and formats the existing model outputs. It does not
generate probabilities, infer unobserved outcomes, retrain, sample, or change
the production recommendation.

A conversion/made kick is a short-run outcome. It is NOT synonymous with
the estimated probability of eventually winning the game.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

import pandas as pd

# These are the NORMAL eligible actions only, never experimental fakes.
NORMAL_ACTIONS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "NORMAL_GO_PASS": (
        "GO — PASS / DROPBACK",
        ("conversion",),
    ),
    "NORMAL_GO_RUN": (
        "GO — RUN",
        ("conversion",),
    ),
    "FIELD_GOAL": (
        "FIELD GOAL",
        ("make", "miss", "block"),
    ),
    "PUNT": (
        "PUNT",
        ("regular", "blocked", "broken"),
    ),
}

REQUIRED_RESULT_COLUMNS = {
    "action", "eligible", "expected_win_probability", "mc_se", "detail",
}
# The source prints 3 or 4 decimal places. Strict but allow rounding.
PROBABILITY_SUM_TOLERANCE = 0.005


def parse_engine_detail(action: str, detail: str) -> dict[str, float]:
    """Parse exact frozen-engine action labels, never estimate missing inputs."""
    if action not in NORMAL_ACTIONS:
        raise ValueError("Action is not a supported normal fourth-down play.")
    if not isinstance(detail, str) or not detail.strip():
        raise ValueError("No model outcome assumptions available.")
    expected = NORMAL_ACTIONS[action][1]
    parts = [part.strip() for part in detail.split(",")]
    values = {}
    for part in parts:
        match = re.fullmatch(
            r"([a-z_]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)", part
        )
        if not match:
            raise ValueError("Unrecognized frozen model detail format.")
        key = match.group(1)
        number = float(match.group(2))
        if key in values or not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("Invalid or duplicate model outcome probability.")
        values[key] = number
    if tuple(values) != tuple(expected):
        raise ValueError("Model detail keys differ from the frozen action schema.")
    if len(expected) > 1 and abs(sum(values.values()) - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError("Modeled FG/punt outcome probabilities do not total 100%.")
    return values


def formatted_outcomes(action: str, rates: Mapping[str, float]) -> str:
    """Readably distinguish short-run outcomes from game-winning chances."""
    pct = lambda x: f"{100 * rates[x]:.1f}%"
    if action in ("NORMAL_GO_PASS", "NORMAL_GO_RUN"):
        return f"Convert: {pct('conversion')}"
    if action == "FIELD_GOAL":
        return (
            f"Make: {pct('make')}  |  "
            f"Miss: {pct('miss')}  |  "
            f"Block: {pct('block')}"
        )
    if action == "PUNT":
        return (
            f"Regular: {pct('regular')}  |  "
            f"Blocked: {pct('blocked')}  |  "
            f"Broken play: {pct('broken')}"
        )
    raise ValueError("Unsupported action.")


def action_assumptions_table(results: pd.DataFrame) -> pd.DataFrame:
    """Render one row per eligible NORMAL play without modifying its WP.

    Fail closed if any eligible normal action's structured outputs are
    missing or inconsistent. Never silently fabricate a probability.
    """
    if missing := REQUIRED_RESULT_COLUMNS.difference(results.columns):
        raise ValueError(f"Missing frozen output columns: {sorted(missing)}")
    out = []
    selected = results.loc[
        results["eligible"] & results["action"].isin(NORMAL_ACTIONS)
    ]
    if selected["action"].duplicated().any():
        raise ValueError("Duplicate normal action in recommendation output.")
    for _, row in selected.iterrows():
        action = str(row["action"])
        wp = float(row["expected_win_probability"])
        mc_se = float(row["mc_se"])
        if (
            not math.isfinite(wp) or not 0 <= wp <= 1
            or not math.isfinite(mc_se) or mc_se < 0
        ):
            raise ValueError("Invalid simulation WP or MC standard error.")
        probabilities = parse_engine_detail(action, row["detail"])
        out.append({
            "Play": NORMAL_ACTIONS[action][0],
            "Immediate modeled outcome": formatted_outcomes(action, probabilities),
            "Estimated game WP": f"{100 * wp:.2f}%",
            "MC SE (pp)": f"{100 * mc_se:.2f}",
        })
    return pd.DataFrame(out, columns=[
        "Play", "Immediate modeled outcome", "Estimated game WP", "MC SE (pp)",
    ])
