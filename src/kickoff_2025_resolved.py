"""Resolved 2025 Q4 nonscoring kickoff transitions (ordinary + penalties).

The original 488 frozen 2025 ordinary kickoffs are combined with exactly 50
verified 2025 penalty-affected kickoffs that led to clean, score-neutral first
scrimmage snaps. The two re-kicks, 41 declared onside attempts, one return
touchdown and two unmatched outcomes remain UNMODELED.

This is a historical selected distribution (538/584 eligible Q4 kicks).
Not a validated 2026+ policy, does not extrapolate penalty frequencies to
other kickoff spots, and NEVER modifies the production engine namespace.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .kickoff_2025_empirical import (
    FROZEN_PATH as ORDINARY_PATH, COLUMNS as ORDINARY_COLUMNS,
    load_frozen_pool, validate_frozen_pool,
)

ROOT = Path(__file__).resolve().parents[1]
PENALTY_REPORT = ROOT / "data/kickoff_2025_penalty_return_coverage.json"
PENALTY_CSV = ROOT / "data/kickoff_2025_penalty_first_snaps.csv"
ORDINARY_COUNT = 488
PENALTY_COUNT = 50
TOTAL_COUNT = ORDINARY_COUNT + PENALTY_COUNT
NEW_TYPE = "penalty_first_snap"
ALLOWED_TYPES = {"touchback", "return", NEW_TYPE}


def load_resolved_2025_pool(
    ordinary_path: Path = ORDINARY_PATH,
    penalty_path: Path = PENALTY_CSV,
    report_path: Path = PENALTY_REPORT,
) -> pd.DataFrame:
    """Build the reviewed pool from versioned files, never live PBP downloads."""
    ordinary = load_frozen_pool(ordinary_path)
    if len(ordinary) != ORDINARY_COUNT:
        raise ValueError("The frozen 2025 ordinary donor count changed.")
    if not penalty_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Verified 2025 penalty kickoff evidence missing.")
    report = json.loads(report_path.read_text())
    if (
        report.get("year") != 2025
        or report.get("special_total") != 55
        or report.get("penalty_total") != 52
        or report.get("return_td_total") != 1
        or report.get("other_unmatched_total") != 2
        or report.get("verified_penalty_first_snap") != PENALTY_COUNT
        or report.get("verified_penalty_games") != 43
        or report.get("no_2026_data") is not True
        or report.get("not_a_future_penalty_probability") is not True
    ):
        raise ValueError("Penalty continuation metadata has changed or is unverified.")
    raw = pd.read_csv(penalty_path)
    needed = {
        "game_id", "play_id", "week", "outcome_group", "outcome_reason",
        "posteam_kick", "posteam_snap", "game_seconds_remaining",
        "game_seconds_remaining_snap", "elapsed_seconds",
        "yardline_100_snap", "next_receiving_own_yardline",
        "verified_penalty_first_snap", "return_td_flag", "penalty_flag",
    }
    if missing := needed.difference(raw.columns):
        raise ValueError(f"Penalty kickoff evidence missing fields: {sorted(missing)}")
    if len(raw) != 55 or raw.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Historical exceptional kickoff denominator changed.")
    selected = raw.loc[raw["verified_penalty_first_snap"].eq(True)].copy()
    if (
        len(selected) != PENALTY_COUNT
        or selected["game_id"].nunique() != 43
        or not selected["outcome_group"].eq("penalty_or_rekick").all()
        or not selected["outcome_reason"].eq("verified_penalty_first_snap").all()
        or not selected["posteam_kick"].eq(selected["posteam_snap"]).all()
        or not selected["penalty_flag"].eq(True).all()
        or selected["return_td_flag"].any()
        or not selected["next_receiving_own_yardline"].between(1, 99).all()
        or not selected["elapsed_seconds"].between(0, 35).all()
        or not selected["week"].between(1, 18).all()
        or not np.allclose(
            selected["elapsed_seconds"].to_numpy(float),
            selected["game_seconds_remaining"].to_numpy(float)
            - selected["game_seconds_remaining_snap"].to_numpy(float),
        )
        or not np.allclose(
            selected["next_receiving_own_yardline"].to_numpy(float),
            100 - selected["yardline_100_snap"].to_numpy(float),
        )
    ):
        raise ValueError("Penalty first-snap records fail eligibility checks.")
    penalty = pd.DataFrame({
        "season": 2025,
        "game_id": selected["game_id"].astype(str).to_numpy(),
        "week": selected["week"].to_numpy(int),
        "kick_play_id": selected["play_id"].to_numpy(float),
        "quarter": 4,
        "transition_type": NEW_TYPE,
        "own_yardline": selected["next_receiving_own_yardline"].to_numpy(float),
        "elapsed_seconds": selected["elapsed_seconds"].to_numpy(float),
    }, columns=ORDINARY_COLUMNS)
    combined = pd.concat([ordinary, penalty], ignore_index=True)
    return validate_resolved_pool(combined)


def validate_resolved_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on unverified rows, duplicate kicks or old/new schema drift."""
    if missing := set(ORDINARY_COLUMNS).difference(pool.columns):
        raise ValueError(f"Resolved kickoff pool missing columns: {sorted(missing)}")
    if (
        len(pool) != TOTAL_COUNT
        or pool.duplicated(["game_id", "kick_play_id"]).any()
        or not pool["season"].eq(2025).all()
        or not pool["quarter"].eq(4).all()
        or not pool["week"].between(1, 18).all()
        or not pool["transition_type"].isin(ALLOWED_TYPES).all()
        or not pool["own_yardline"].between(1, 99).all()
        or not pool["elapsed_seconds"].between(0, 35).all()
    ):
        raise ValueError("Frozen 2025 combined kickoff pool not verified.")
    ordinary = pool.loc[pool["transition_type"].ne(NEW_TYPE)].copy()
    penalty = pool.loc[pool["transition_type"].eq(NEW_TYPE)].copy()
    if len(ordinary) != ORDINARY_COUNT or len(penalty) != PENALTY_COUNT:
        raise ValueError("Resolved kickoff pool has the wrong event mixture.")
    validate_frozen_pool(ordinary)
    if penalty["game_id"].nunique() != 43:
        raise ValueError("Penalty kickoff sample must span 43 distinct games.")
    return pool.copy()


def resolve_post_score_q4_resolved(
    states: list[dict], pool: pd.DataFrame, *,
    rng: np.random.Generator, rules_year: int = 2026,
) -> list[dict]:
    """Sample one entire ordinary/penalty first-snap transition per kickoff."""
    if rules_year < 2025:
        raise ValueError("Historical 2025 kickoff states need 2025+ rules.")
    validate_resolved_pool(pool)
    output = [dict(x) for x in states]
    eligible = [
        i for i, state in enumerate(output)
        if (
            int(state.get("qtr", 0)) == 4
            and float(state.get("is_kickoff", 0)) > .5
            and abs(float(state.get("_score_change_original", 0))) > 0
            and float(state.get("game_seconds_remaining", 0)) > 0
        )
    ]
    if not eligible:
        return output
    donors = pool.iloc[rng.integers(0, len(pool), size=len(eligible))]
    # Special kick from midfield has different penalty conditions. There
    # are no verified 2025 midfield penalty donors in this sample, so
    # always resample those states from ordinary donors only. The existing
    # rule-specific own-20 touchback correction is preserved.
    ordinary_only = pool.loc[pool["transition_type"].ne(NEW_TYPE)].reset_index(drop=True)
    for idx, (_, donor) in zip(eligible, donors.iterrows()):
        state = output[idx]
        if (
            rules_year >= 2026
            and bool(state.get("_kickoff_from_50", False))
            and str(donor["transition_type"]) == NEW_TYPE
        ):
            donor = ordinary_only.iloc[
                int(rng.integers(0, len(ordinary_only)))
            ]
        own = float(donor["own_yardline"])
        elapsed = float(donor["elapsed_seconds"])
        transition = str(donor["transition_type"])
        if (
            rules_year >= 2026
            and bool(state.get("_kickoff_from_50", False))
            and transition == "touchback"
        ):
            own = 20.
        clock = max(0., float(state["game_seconds_remaining"]) - elapsed)
        state.update({
            "is_kickoff": 0.0,
            "yardline_100": 100.0 - own,
            "down": 1.0,
            "ydstogo": min(10., 100.0 - own),
            "goal_to_go": float((100.0 - own) <= 10.0),
            "game_seconds_remaining": clock,
            "half_seconds_remaining": max(
                0., float(state["half_seconds_remaining"]) - elapsed
            ),
            "final_five_minutes": float(clock <= 300.),
            "_kickoff_transition_type": transition,
        })
    return output


@contextmanager
def use_resolved_2025_kickoffs(
    engine: Mapping, pool: pd.DataFrame, *,
    seed: int = 2026, rules_year: int = 2026,
) -> Iterator[None]:
    """Temporarily replace evaluator only in an isolated engine namespace."""
    validate_resolved_pool(pool)
    original = engine["evaluate_post_play_states"]
    namespace = original.__globals__
    if namespace.get("evaluate_post_play_states") is not original:
        raise RuntimeError("An evaluator override is already active.")
    rng = np.random.default_rng(seed)

    def evaluator(base, states):
        if int(base["qtr"]) != 4:
            return original(base, states)
        mapped = resolve_post_score_q4_resolved(
            states, pool, rng=rng, rules_year=rules_year,
        )
        return original(base, mapped)

    namespace["evaluate_post_play_states"] = evaluator
    try:
        yield
    finally:
        namespace["evaluate_post_play_states"] = original
