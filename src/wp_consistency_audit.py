"""Read-only diagnostics of learned post-score regulation win probabilities.

Compares the win-probability model's kickoff-pending estimate with an exact
2024-return-donor-weighted expectation of post-kickoff possession states under
explicit 2025+ touchback-rate scenarios. This is a consistency stress test,
NOT a new transition model, forecast, or causal accuracy estimate.

The only data donors are 2024 Q4 ordinary returns and touchback types.
No 2025/2026 play-by-play is loaded and no model is fitted or modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


RECEIVER_OWN_GRID = (2, 4, 6, 10, 15, 20, 25, 30, 35, 40)
TOUCHBACK_RATE_SCENARIOS = (0.20, 0.30, 0.40)


@dataclass(frozen=True)
class KickoffDonors:
    returns_own: np.ndarray
    returns_clock: np.ndarray
    touchback_own20_probability: float
    touchback_count: int

    @property
    def return_count(self) -> int:
        return len(self.returns_own)


def validate_2024_q4_donors(pool: pd.DataFrame) -> KickoffDonors:
    """Require ordinary 2024 Q4 donors; don't mix evaluation seasons into them."""
    required = {"season", "quarter", "transition_type", "own_yardline", "elapsed_seconds"}
    missing = required.difference(pool.columns)
    if missing:
        raise ValueError(f"Missing donor columns: {sorted(missing)}")
    if pool.empty or not pool["season"].eq(2024).all():
        raise ValueError("Only 2024 development kickoff donors are permitted.")
    q4 = pool.loc[pool["quarter"].eq(4)]
    if q4.empty or not q4["transition_type"].isin(["touchback", "return"]).all():
        raise ValueError("Need ordinary Q4 touchbacks and returns.")
    tb = q4.loc[q4["transition_type"].eq("touchback")]
    ret = q4.loc[q4["transition_type"].eq("return")]
    if tb.empty or ret.empty:
        raise ValueError("Both Q4 touchbacks and Q4 returns are required.")
    if not tb["own_yardline"].isin([20, 30]).all():
        raise ValueError("2024 touchbacks must start at own 20 or own 30.")
    if not tb["elapsed_seconds"].between(0, 35).all():
        raise ValueError("Invalid 2024 touchback transition clock values.")
    if not ret["own_yardline"].between(1, 99).all():
        raise ValueError("Invalid ordinary return starting position.")
    if not ret["elapsed_seconds"].between(1, 35).all():
        raise ValueError("Invalid ordinary return game-clock runoff.")
    return KickoffDonors(
        returns_own=ret["own_yardline"].to_numpy(dtype=float),
        returns_clock=ret["elapsed_seconds"].to_numpy(dtype=float),
        touchback_own20_probability=float(tb["own_yardline"].eq(20).mean()),
        touchback_count=len(tb),
    )


def weighted_continuation(
    touchback_rate: float,
    wp_20: float,
    wp_35: float,
    return_wps: Sequence[float],
    *,
    touchback_own20_probability: float,
) -> float:
    """Compute EXACT donor expectation, without Monte Carlo draws or noise."""
    if not np.isfinite(touchback_rate) or not 0 <= touchback_rate <= 1:
        raise ValueError("Touchback rate must be in [0, 1].")
    if (not np.isfinite(touchback_own20_probability)
            or not 0 <= touchback_own20_probability <= 1):
        raise ValueError("Touchback-at-20 share must be in [0, 1].")
    returns = np.asarray(return_wps, dtype=float)
    if returns.ndim != 1 or returns.size == 0 or not np.isfinite(returns).all():
        raise ValueError("Need finite, nonempty ordinary return win probabilities.")
    if not np.isfinite(wp_20) or not np.isfinite(wp_35):
        raise ValueError("Touchback win probabilities must be finite.")
    expected_tb = (touchback_own20_probability * wp_20
                   + (1 - touchback_own20_probability) * wp_35)
    return float(touchback_rate * expected_tb
                 + (1 - touchback_rate) * returns.mean())


def field_position_reversals(
    own_yards: Sequence[float],
    wp_from_kicking_team: Sequence[float],
    *,
    minimum_pp: float = 0.5,
) -> list[dict]:
    """Find materially counterintuitive increases in kicker WP as receiver advances.

    Not a claim of formal model invalidity: model WP is not monotonic-constrained.
    """
    own = np.asarray(own_yards, dtype=float)
    wp = np.asarray(wp_from_kicking_team, dtype=float)
    if (own.ndim != 1 or wp.ndim != 1 or len(own) != len(wp)
            or len(own) < 2 or not np.isfinite(own).all()
            or not np.isfinite(wp).all() or not np.all(np.diff(own) > 0)):
        raise ValueError("Field positions must increase strictly with finite matching WP values.")
    if minimum_pp < 0 or not np.isfinite(minimum_pp):
        raise ValueError("minimum_pp must be finite and nonnegative.")
    changes_pp = 100 * np.diff(wp)
    return [
        {"from_own": float(own[i]), "to_own": float(own[i + 1]),
         "kicker_wp_increase_pp": float(change)}
        for i, change in enumerate(changes_pp)
        if change > minimum_pp
    ]


def evaluate_made_fg_case(
    engine: Mapping,
    state: Mapping,
    donors: KickoffDonors,
    *,
    touchback_rates: Sequence[float] = TOUCHBACK_RATE_SCENARIOS,
    fg_clock_seconds: float = 5,
) -> dict:
    """Compare like-for-like Q4 post-FG kickoff-pending and receiving possessions.

    A five-second FG snap clock cost is applied equally to kickoff-pending
    and first-down states; returns consume their donor-observed extra clock.
    Before the kick the original offense scores +3, then the OTHER team receives.
    There are no XP/2PT branches in a made-FG transition.
    """
    if int(state["qtr"]) != 4 or float(state["game_seconds_remaining"]) <= 40:
        raise ValueError("Use regulation Q4 cases with >40 seconds remaining.")
    if fg_clock_seconds < 0 or not np.isfinite(fg_clock_seconds):
        raise ValueError("Invalid FG clock cost.")
    original_home = bool(state["_wp_original_on_home_axis"])
    neutral = state["site"] == "NEUTRAL"

    def post_state(own: float, return_clock: float = 0.0, *, is_kickoff: bool = False):
        return engine["make_wp_state"](
            base=state,
            possession_original=False,
            yardline_100=100.0 - float(own),
            down=1,
            ydstogo=10,
            elapsed_seconds=float(fg_clock_seconds) + float(return_clock),
            score_change_original=3.0,
            is_kickoff=is_kickoff,
        )

    # Single vectorized model call makes every WP value internally comparable.
    states = [post_state(30, is_kickoff=True)]
    states += [post_state(20), post_state(35)]
    states += [post_state(float(yard)) for yard in RECEIVER_OWN_GRID]
    states += [post_state(float(yard), float(clock)) for yard, clock in
               zip(donors.returns_own, donors.returns_clock)]

    wp = np.asarray(engine["predict_original_team_wp"](
        states, original_home, neutral), dtype=float)
    if wp.shape != (len(states),) or not np.isfinite(wp).all():
        raise ValueError("WP model returned incorrect, nonfinite output.")
    # The original model's kickoff pending state has down/yardline as NaN;
    # equivalent post-kickoff states have down=1 and receiver-owned yardlines.
    kickoff_wp, wp_20, wp_35 = map(float, wp[:3])
    grid_end = 3 + len(RECEIVER_OWN_GRID)
    field_wp = wp[3:grid_end]
    return_wp = wp[grid_end:]

    scenario_values = {
        float(rate): weighted_continuation(
            float(rate), wp_20, wp_35, return_wp,
            touchback_own20_probability=donors.touchback_own20_probability,
        )
        for rate in touchback_rates
    }
    return {
        "site": state["site"],
        "time_seconds": int(state["game_seconds_remaining"]),
        "pre_fg_score_diff": float(state["score_differential"]),
        "post_fg_score_diff": float(state["score_differential"]) + 3.0,
        "kickoff_pending_wp": kickoff_wp,
        "post_kick_return_wp": float(return_wp.mean()),
        "post_kick_own20_wp": wp_20,
        "post_kick_own35_wp": wp_35,
        "mixed_wp": scenario_values,
        "field_positions": tuple(RECEIVER_OWN_GRID),
        "field_wps": tuple(map(float, field_wp)),
        "reversals": field_position_reversals(RECEIVER_OWN_GRID, field_wp),
    }
