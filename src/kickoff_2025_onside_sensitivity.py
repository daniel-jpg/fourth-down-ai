"""Optional Q4 onside-kick what-if layered on ordinary 2025 kickoff donors.

Only when a 2026-regulation Q4 made FG leaves its kicking team TRAILING,
allow a user-assumed onside attempt probability and user-assumed conditional
recovery probability. The two probabilities are NOT estimated from 2025.
Conditional on either recovery outcome, sample the WHOLE verified 2025
onside-to-first-snap transition (field position + elapsed time) jointly.

The scoring game clock and score do not change on the kickoff itself.
Kicker recovery flips possession away from the expected receiving team.

No failed- or successful-onside branch is replaced with ordinary kickoff
returns. We sample actual onside first snaps for both branches. These 37
selected observations, especially only FOUR kicking-team recoveries, are
NOT independently calibrated 2026 policy advice.

All monkeypatches are performed ONLY in a PRIVATE engine namespace.
Never mutate Streamlit's cached production model or engine.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .kickoff_2025_empirical import (
    FROZEN_PATH, load_frozen_pool, resolve_post_score_q4_2025,
)

ROOT = Path(__file__).resolve().parents[1]
ONSIDE_JSON = ROOT / "data/kickoff_2025_onside_outcomes.json"
ONSIDE_CSV = ROOT / "data/kickoff_2025_onside_first_snaps.csv"


def load_onside_verified(
    path: Path = ONSIDE_CSV, report_path: Path = ONSIDE_JSON,
) -> tuple[pd.DataFrame, dict]:
    """Require matching, fully reviewed 2025 files; fail closed otherwise."""
    if not path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Verified frozen 2025 onside evidence is missing.")
    report = json.loads(report_path.read_text())
    if (
        report.get("year") != 2025
        or report.get("all_on_side_eligible") != 41
        or report.get("flag_known") != 41
        or report.get("official_kicker_recoveries") != 4
        or report.get("matched_reliable") != 37
        or report.get("no_2026_data") is not True
        or report.get("not_a_policy_probability") is not True
    ):
        raise ValueError("Unverified or stale 2025 onside metadata.")
    raw = pd.read_csv(path)
    required = {
        "game_id", "play_id", "week", "posteam_kick", "posteam_snap",
        "official_kicker_recovered", "game_seconds_remaining",
        "game_seconds_remaining_snap", "kickoff_to_snap_elapsed",
        "yardline_100_snap", "penalty_flag", "return_td_flag",
        "matched_reliable", "unmatched_reason",
    }
    if missing := required.difference(raw.columns):
        raise ValueError(f"Frozen onside donor fields missing: {sorted(missing)}")
    if len(raw) != 41 or raw.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Onside donor universe must have exactly 41 unique kicks.")
    reliable = raw.loc[raw["matched_reliable"].eq(True)].copy()
    rec = pd.to_numeric(reliable["official_kicker_recovered"], errors="coerce")
    observed_change = reliable["posteam_snap"].ne(reliable["posteam_kick"])
    if (
        len(reliable) != 37
        or int(rec.eq(1).sum()) != 4
        or int(rec.eq(0).sum()) != 33
        or not rec.isin([0, 1]).all()
        or not observed_change.eq(rec.eq(1)).all()
        or not reliable["unmatched_reason"].eq("verified").all()
        or reliable[["penalty_flag", "return_td_flag"]].any().any()
        or not reliable["yardline_100_snap"].between(1, 99).all()
        or not reliable["kickoff_to_snap_elapsed"].between(0, 35).all()
        or not (
            reliable["game_seconds_remaining_snap"]
            <= reliable["game_seconds_remaining"]
        ).all()
        or not reliable["week"].between(1, 18).all()
    ):
        raise ValueError("Verified onside outcomes are inconsistent.")
    clean = reliable[[
        "game_id", "play_id", "official_kicker_recovered",
        "yardline_100_snap", "kickoff_to_snap_elapsed",
    ]].reset_index(drop=True)
    return clean, report


def validate_what_if(p_attempt: float, p_recovery: float) -> None:
    if any(
        not np.isfinite(value) or not 0 <= value <= 1
        for value in (p_attempt, p_recovery)
    ):
        raise ValueError("What-if probabilities must both be in [0, 1].")


def resolve_q4_fg_onside_what_if(
    base: Mapping, pending: list[dict], ordinary: pd.DataFrame,
    onside: pd.DataFrame, *, rng_ordinary: np.random.Generator,
    rng_special: np.random.Generator, p_attempt: float,
    p_recovery: float,
) -> list[dict]:
    """Replace only selected made-FG continuation states with observed onside.

    This leaves the original 2025 ordinary sampler's RNG sequence unchanged
    for the OTHER states, giving the zero-attempt scenario exact parity.
    """
    validate_what_if(p_attempt, p_recovery)
    output = resolve_post_score_q4_2025(
        pending, ordinary, rng=rng_ordinary, rules_year=2026,
    )
    if not (
        int(base.get("qtr", 0)) == 4
        and float(base.get("score_differential", 0)) + 3.0 < 0
    ):
        return output
    eligible = [
        i for i, state in enumerate(pending)
        if (
            int(state.get("qtr", 0)) == 4
            and float(state.get("is_kickoff", 0)) > .5
            and float(state.get("_score_change_original", 0)) == 3.0
            and float(state.get("game_seconds_remaining", 0)) > 40
            and state.get("_touchdown_original", 0) == 0
        )
    ]
    if not eligible or p_attempt == 0:
        return output
    # Draw all counterfactual uniforms independently of the sliders.
    # Changing an assumed attempt/recovery percentage must not reshuffle
    # all subsequent onside donor draws. Ordinary-kickoff RNG is separate.
    attempt_draws = rng_special.random(len(eligible))
    recovery_draws = rng_special.random(len(eligible))
    donor_draws = rng_special.random(len(eligible))
    for i, attempt_draw, recovery_draw, donor_draw in zip(
        eligible, attempt_draws, recovery_draws, donor_draws
    ):
        if attempt_draw >= p_attempt:
            continue
        recovered = bool(recovery_draw < p_recovery)
        subset = onside.loc[
            onside["official_kicker_recovered"].eq(float(recovered))
        ]
        if subset.empty:
            raise ValueError(
                "No verified onside first-snap outcomes for an attempted branch."
            )
        donor_index = min(int(donor_draw * len(subset)), len(subset)-1)
        donor = subset.iloc[donor_index]
        elapsed = float(donor["kickoff_to_snap_elapsed"])
        original = pending[i]
        new_clock = max(0., float(original["game_seconds_remaining"]) - elapsed)
        next_state = dict(output[i])
        next_state.update({
            "is_kickoff": 0.0,
            "is_home_posteam": (
                float(1 - int(original["is_home_posteam"]))
                if recovered else float(original["is_home_posteam"])
            ),
            "yardline_100": float(donor["yardline_100_snap"]),
            "down": 1.0,
            "ydstogo": min(10., float(donor["yardline_100_snap"])),
            "goal_to_go": float(
                float(donor["yardline_100_snap"]) <= 10.
            ),
            "game_seconds_remaining": new_clock,
            "half_seconds_remaining": max(
                0., float(original["half_seconds_remaining"]) - elapsed
            ),
            "final_five_minutes": float(new_clock <= 300.),
            "_kickoff_transition_type": (
                "onside_kicker_recovered" if recovered
                else "onside_receiver_recovered"
            ),
        })
        output[i] = next_state
    return output


@contextmanager
def use_q4_onside_what_if(
    engine: Mapping, ordinary: pd.DataFrame, onside: pd.DataFrame, *,
    p_attempt: float, p_recovery: float,
    ordinary_seed: int = 2026, onside_seed: int = 2075,
) -> Iterator[None]:
    """Isolated, reproducible, nonproduction experimental FG continuation."""
    validate_what_if(p_attempt, p_recovery)
    # Verify the donor data comes from the explicit approved frozen sources.
    if not len(ordinary) >= 300 or not len(onside) == 37:
        raise ValueError("Unverified kickoff data size.")
    if onside["official_kicker_recovered"].eq(1).sum() != 4:
        raise ValueError("Onside recovery donor count changed.")
    original = engine["evaluate_post_play_states"]
    namespace = original.__globals__
    if namespace.get("evaluate_post_play_states") is not original:
        raise RuntimeError("Evaluator already overridden. No nested scenarios.")
    ordinary_rng = np.random.default_rng(ordinary_seed)
    onside_rng = np.random.default_rng(onside_seed)

    def evaluate(base, states):
        if int(base["qtr"]) != 4:
            return original(base, states)
        modified = resolve_q4_fg_onside_what_if(
            base, states, ordinary, onside,
            rng_ordinary=ordinary_rng, rng_special=onside_rng,
            p_attempt=p_attempt, p_recovery=p_recovery,
        )
        return original(base, modified)

    namespace["evaluate_post_play_states"] = evaluate
    try:
        yield
    finally:
        namespace["evaluate_post_play_states"] = original
