"""Opt-in train-aligned encoding for standard Q4 scoring kickoffs.

The frozen production WP model was trained with NFL PBP kickoff metadata:
down=NaN, ydstogo=0, yardline_100=35 (ordinary kick from own 35),
goal_to_go=0. The current engine instead supplies NaN for yardline and
ydstogo. This module can compare the two WITHOUT editing production.

Important: kickoff yardline_100=35 is metadata for the KICKING SPOT, not
the receiver's eventual possession yard line. Do not apply this encoding to
post-return scrimmage states, onside kicks, unusual kickoff spots, or
2025+ kickoff outcomes without additional validation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import numpy as np


def _number(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def native_like_q4_scoring_kickoffs(
    states: Sequence[Mapping], *, kickoff_yardline_100: float = 35.0
) -> list[Mapping]:
    """Only transform unresolved, synthetic, post-score Q4 kickoff states.

    Copy-on-write: no changes to caller's states or other model features.
    Uses a fixed STANDARD kickoff location; unusual spots are not modeled.
    """
    if not np.isfinite(kickoff_yardline_100) or not 0 < kickoff_yardline_100 < 100:
        raise ValueError("Kickoff yardline must be a finite number between 0 and 100.")
    result = []
    for state in states:
        q = _number(state.get("qtr"))
        is_kickoff = _number(state.get("is_kickoff"), 0)
        score_delta = _number(state.get("_score_change_original"), 0)
        current_yard = _number(state.get("yardline_100"))
        should_encode = (
            q == 4
            and is_kickoff == 1
            and np.isfinite(score_delta)
            and score_delta != 0
            and np.isnan(current_yard)
        )
        if should_encode:
            copied = dict(state)
            copied["down"] = np.nan
            copied["ydstogo"] = 0.0
            copied["yardline_100"] = float(kickoff_yardline_100)
            copied["goal_to_go"] = 0.0
            result.append(copied)
        else:
            result.append(state)
    return result


@contextmanager
def use_native_like_q4_kickoff_wp(
    engine: Mapping, *, kickoff_yardline_100: float = 35.0
) -> Iterator[None]:
    """Temporarily intercept kickoff WP inputs; restore even on exceptions.

    Works with engine=runpy.run_path(...), whose functions share globals.
    Independent of the existing opt-in post-kickoff continuation context.
    Not safe for concurrent production serving, so never use in app.py.
    """
    if not np.isfinite(kickoff_yardline_100) or not 0 < kickoff_yardline_100 < 100:
        raise ValueError("Kickoff yardline must be between 0 and 100.")
    original = engine["predict_original_team_wp"]
    namespace = original.__globals__
    if namespace.get("predict_original_team_wp") is not original:
        raise RuntimeError("Kickoff WP already overridden; nested encoding unsupported.")

    def patched_predict(wp_states, original_on_home_axis, neutral_site=False):
        transformed = native_like_q4_scoring_kickoffs(
            wp_states, kickoff_yardline_100=kickoff_yardline_100
        )
        return original(transformed, original_on_home_axis, neutral_site)

    namespace["predict_original_team_wp"] = patched_predict
    try:
        yield
    finally:
        namespace["predict_original_team_wp"] = original
