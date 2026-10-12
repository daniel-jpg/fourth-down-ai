"""Fast, deterministic reliability check for known Q4 post-FG kickoff inputs.

The official WP evaluator currently passes down, ydstogo and yardline_100
as missing on an unresolved kickoff after a made field goal. The research
candidate instead supplies native normal-35 kickoff metadata:
  down=NaN, ydstogo=0, yardline_100=35.

For eligible Q4 FG situations, compare the SAME frozen WP model on
these two otherwise identical hypothetical states, *conditional on a
made field goal and an explicitly assumed 5-second execution clock*.

This is not a full field-goal action WP, not a GO-vs-FG decision test,
not a 95% confidence interval, and not a model correction. A 5 pp
display threshold is a pragmatic material-difference indicator,
not a calibrated statistical significance threshold.

The check uses a fresh isolated model module and cannot change
production weights or the main recommendation. Never reads 2026 data.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .kickoff_q4_native_features_candidate import (
    native_q4_normal_kickoff,
)

FG_EXECUTION_SECONDS = 5.0
DISPLAY_ABS_DIFFERENCE_PP = 5.0
LAST_Q4_STANDARD_KICKOFF_SECONDS = 40.0
Q4_LENGTH_SECONDS = 900.0


def inspect_made_fg_kickoff_encoding(
    state: Mapping, *,
    isolated_engine: Mapping | None = None,
    material_threshold_pp: float = DISPLAY_ABS_DIFFERENCE_PP,
) -> dict:
    """Evaluate one *made FG* kickoff-input sensitivity, without sampling.

    Only call this when the normal FG is eligible in the official
    recommendation. Site means the ORIGINAL offensive team's site.
    All outcomes are in original-offense-win probability coordinates.
    """
    threshold = float(material_threshold_pp)
    if not np.isfinite(threshold) or threshold <= 0 or threshold > 100:
        raise ValueError("Materiality threshold must be in (0,100] pp.")
    qtr = int(state["qtr"])
    seconds = float(state["game_seconds_remaining"])
    if not np.isfinite(seconds):
        raise ValueError("Invalid game clock for kickoff reliability check.")
    # The historical reliability cohort requires >40 seconds *at the
    # kickoff*. Subtract the fixed five-second FG before that scope check.
    if qtr != 4 or not (
        LAST_Q4_STANDARD_KICKOFF_SECONDS + FG_EXECUTION_SECONDS
        < seconds <= Q4_LENGTH_SECONDS
    ):
        return {
            "applicable": False,
            "reason": "Only Q4 made-FG kickoffs with >40 post-FG seconds",
        }

    if isolated_engine is None:
        # Avoid sharing a module namespace or monkey-patching live WP
        # callbacks across different Streamlit user sessions.
        from .streamlit_kickoff_preview import isolated_engine_for_preview
        isolated_engine = isolated_engine_for_preview()

    engine = isolated_engine
    base = engine["normalize_state"](dict(state))
    pending = engine["make_wp_state"](
        base,
        possession_original=False,
        yardline_100=np.nan,
        down=np.nan,
        ydstogo=np.nan,
        elapsed_seconds=FG_EXECUTION_SECONDS,
        score_change_original=3.0,
        is_kickoff=True,
    )
    if (
        int(pending["qtr"]) != 4
        or float(pending["is_kickoff"]) != 1.0
        or bool(pending.get("_kickoff_from_50", False))
        or float(pending["game_seconds_remaining"]) <= 0
    ):
        return {
            "applicable": False,
            "reason": "Special or unresolved nonstandard FG kickoff",
        }

    native = native_q4_normal_kickoff(pending)
    if (
        not np.isnan(float(native["down"]))
        or float(native["ydstogo"]) != 0.0
        or float(native["yardline_100"]) != 35.0
        or float(native["is_home_posteam"]) != float(pending["is_home_posteam"])
        or float(native["home_score_differential"])
           != float(pending["home_score_differential"])
        or float(native["game_seconds_remaining"])
           != float(pending["game_seconds_remaining"])
    ):
        raise RuntimeError("Research candidate failed kickoff state invariants.")

    wp = np.asarray(
        engine["evaluate_post_play_states"](base, [pending, native]),
        dtype=float,
    )
    if wp.shape != (2,) or not np.isfinite(wp).all() or not np.all(
        (0 <= wp) & (wp <= 1)
    ):
        raise RuntimeError("Kickoff reliability probe returned invalid win probabilities.")
    original, native_probability = map(float, wp)
    delta_pp = 100.0 * (native_probability - original)

    return {
        "applicable": True,
        "material": bool(abs(delta_pp) >= threshold),
        "materiality_threshold_pp": threshold,
        "legacy_conditional_made_fg_wp": original,
        "native_conditional_made_fg_wp": native_probability,
        "native_minus_legacy_pp": delta_pp,
        "absolute_difference_pp": abs(delta_pp),
        "assumed_fg_execution_seconds": FG_EXECUTION_SECONDS,
        "post_fg_seconds_remaining": float(pending["game_seconds_remaining"]),
        "original_offense_site": str(base["site"]),
        "models_equal_weights": True,
        "is_action_level_go_vs_fg_comparison": False,
        "2026_data_read": False,
    }


def reliability_warning_markdown(diagnostic: Mapping) -> str | None:
    """Build a precise UI warning; no claim of real play-call optimality."""
    if not diagnostic.get("applicable") or not diagnostic.get("material"):
        return None
    return (
        "**Model reliability warning — Q4 field-goal kickoff encoding.** "
        "The frozen win-probability model is sensitive to how the "
        "kickoff following a hypothetical **made field goal** is "
        "represented: "
        f"original {100*diagnostic['legacy_conditional_made_fg_wp']:.2f}% "
        f"versus native-like input {100*diagnostic['native_conditional_made_fg_wp']:.2f}% "
        "for the original offense "
        f"({diagnostic['absolute_difference_pp']:.2f} percentage-point difference; "
        f"assumes a {diagnostic['assumed_fg_execution_seconds']:.0f}-second FG). "
        "**These are conditional post-FG win probabilities, NOT the "
        "field-goal action's win probability or its edge over GO FOR IT.** "
        "The underlying frozen-model weights are unchanged. "
        "The research-demo Q4 result uses native kickoff inputs "
        "and marks representation-sensitive choices as close calls. "
        "This does NOT prove GO FOR IT or a different play is better."
    )
