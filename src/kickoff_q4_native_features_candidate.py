"""Experimental Q4 normal-kickoff WP input-encoding correction.

The frozen production engine evaluates made FG/TD kickoff-pending states
with yardline_100, ydstogo and down ALL missing, even though observed
historical native kickoff PBP has standard-35 yardline_100=35,
ydstogo=0 and down missing. This creates extreme HOME/AWAY conditional WP
sensitivity in the Q4 7:38 game.

This candidate changes ONLY the three input fields on NORMAL Q4
post-score kickoff-pending states in a PRIVATE model namespace. It does not
change kickoff rules, outcome probabilities or any model coefficients, and
never maps re-kicks/from-midfield kicks or already resolved states.

It must be evaluated as research first; do NOT auto-promote to production.
No new 2026 outcomes are read.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import numpy as np

NATIVE_STANDARD_35 = 35.0


def native_q4_normal_kickoff(state: Mapping) -> dict:
    """Rewrite only an unresolved normal Q4 scoring-kickoff feature vector."""
    row=dict(state)
    required=("qtr","is_kickoff","_score_change_original",
              "game_seconds_remaining","down","ydstogo","yardline_100")
    if not all(k in row for k in required):
        return row
    if not (
        int(row["qtr"]) == 4
        and float(row["game_seconds_remaining"]) > 0
        and float(row["is_kickoff"]) == 1.
        and float(row["_score_change_original"]) != 0.
        and not bool(row.get("_kickoff_from_50", False))
        and all(np.isnan(float(row[k])) for k in (
            "down","ydstogo","yardline_100"
        ))
    ):
        return row
    row.update({
        "down":np.nan,
        "ydstogo":0.0,
        "yardline_100":NATIVE_STANDARD_35,
        "goal_to_go":0.0,
    })
    return row


@contextmanager
def use_q4_native_kickoff_candidate(engine:Mapping)->Iterator[None]:
    """Isolated reversible correction for experimental 2025+ WP only."""
    original=engine["evaluate_post_play_states"]
    ns=original.__globals__
    if ns.get("evaluate_post_play_states") is not original:
        raise RuntimeError("Nested or preexisting evaluator override denied.")

    def evaluate(base,states):
        if int(base["qtr"])!=4:
            return original(base,states)
        return original(base,[native_q4_normal_kickoff(s) for s in states])

    ns["evaluate_post_play_states"]=evaluate
    try:
        yield
    finally:
        ns["evaluate_post_play_states"]=original
