"""Step 76: audit exceptional 2025 Q4 kickoffs before modeling them.

The Step 73 denominator is fixed: 584 standard-35 Q4 REG kickoffs with
>40s, of which 52 are penalty/re-kick and one is a return touchdown.
Pair penalties to the FIRST subsequent REAL scrimmage snap before any
intervening kickoff, reject score changes, possession changes and invalid
elapsed time; never pretend all penalty-flagged kickoffs were re-kicked.

Freeze actual first-snap observations plus explicit unsupported reasons.
Return-TD is ONE real occurrence, not a calibrated frequency or generic
post-score branch. No 2026 game data or production writes.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kickoff_2025_empirical import load_frozen_pool

STEP73 = ROOT / "src/73_audit_2025_kickoff_coverage.py"
OUT_JSON = ROOT / "data/kickoff_2025_penalty_return_coverage.json"
OUT_CSV = ROOT / "data/kickoff_2025_penalty_first_snaps.csv"
SPECIAL_CATEGORIES = ("penalty_or_rekick", "kickoff_return_td", "other_unmatched")
REAL_SNAP_TYPES = {"run", "pass", "qb_kneel", "qb_spike"}
MAX_TRANSITION_SECONDS = 35.0
REQUIRED = {
    "season", "game_id", "play_id", "play_type", "qtr", "posteam",
    "game_seconds_remaining", "yardline_100", "home_score", "away_score",
    "down", "season_type", "week", "desc", "return_touchdown", "penalty",
}


def coverage_helper():
    spec = importlib.util.spec_from_file_location("frozen_kickoff_coverage_73", STEP73)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing fixed 2025 kickoff classification")
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def flagged_2025(raw: pd.DataFrame, ordinary: pd.DataFrame, helper) -> pd.DataFrame:
    if raw.empty or not raw["season"].eq(2025).all():
        raise ValueError("Only 2025 PBP is permitted.")
    if missing := REQUIRED.difference(raw.columns):
        raise ValueError(f"Missing PBP fields {sorted(missing)}")
    tagged = helper.mark_outcomes(helper.candidate_kickoffs(raw), ordinary)
    result = tagged.loc[tagged["outcome_group"].isin(SPECIAL_CATEGORIES)].copy()
    counts = result["outcome_group"].value_counts()
    if {k: int(counts.get(k, 0)) for k in SPECIAL_CATEGORIES} != {
        "penalty_or_rekick": 52, "kickoff_return_td": 1,
        "other_unmatched": 2,
    }:
        raise RuntimeError("2025 frozen special-kickoff taxonomy changed.")
    if len(result) != 55 or result.duplicated(["game_id", "play_id"]).any():
        raise RuntimeError("Unverifiable 2025 exceptional kickoff IDs.")
    if result["onside_flag"].any():
        raise RuntimeError("Onside attempts belong in separate frozen Step 74.")
    return result.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def next_scrimmage_and_rekick(raw: pd.DataFrame, special: pd.DataFrame) -> pd.DataFrame:
    """Join first genuine scrimmage snap after each kickoff, before a re-kick.

    This never crosses kickoff boundaries, games, or Q4 to overtime.
    """
    if raw.empty or not raw["season"].eq(2025).all():
        raise ValueError("Only 2025 completed play-by-play accepted.")
    if missing := REQUIRED.difference(raw.columns):
        raise ValueError(f"Missing snap fields: {sorted(missing)}")
    df = raw.sort_values(["game_id", "play_id"]).reset_index(drop=True).copy()
    if df.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate PBP events prevent valid attribution.")
    df["_kick_seq"] = df["play_type"].eq("kickoff").groupby(df["game_id"]).cumsum()
    keys = ["game_id", "_kick_seq"]
    kicks = df.loc[df["play_type"].eq("kickoff"), keys + [
        "play_id", "posteam", "qtr", "game_seconds_remaining",
        "home_score", "away_score",
    ]].rename(columns={
        "play_id": "play_id_kick",
        "posteam": "posteam_kick",
        "qtr": "qtr_kick",
        "game_seconds_remaining": "game_seconds_remaining_kick",
        "home_score": "home_score_kick",
        "away_score": "away_score_kick",
    }).copy()
    # First legitimate pre-snap real offensive play; administrative
    # no_play events are not an observed first scrimmage state.
    snaps = df.loc[
        df["play_type"].isin(REAL_SNAP_TYPES)
        & df["posteam"].notna()
        & df["down"].notna()
        & df["yardline_100"].notna(),
        keys + ["play_id", "posteam", "qtr", "yardline_100",
                "game_seconds_remaining", "home_score", "away_score"]
    ].drop_duplicates(keys).rename(columns={
        "play_id": "play_id_snap",
        "posteam": "posteam_snap",
        "qtr": "qtr_snap",
        "yardline_100": "yardline_100_snap",
        "game_seconds_remaining": "game_seconds_remaining_snap",
        "home_score": "home_score_snap",
        "away_score": "away_score_snap",
    })
    next_kicks = kicks[["game_id", "_kick_seq", "play_id_kick",
                        "game_seconds_remaining_kick", "qtr_kick"]].copy()
    for col in ("play_id_kick", "game_seconds_remaining_kick", "qtr_kick"):
        next_kicks["next_"+col] = next_kicks.groupby("game_id")[col].shift(-1)
    k = kicks.merge(snaps, on=keys, how="left", validate="one_to_one")
    k = k.merge(
        next_kicks[["game_id", "_kick_seq", "next_play_id_kick",
                    "next_game_seconds_remaining_kick", "next_qtr_kick"]],
        on=keys, how="left", validate="one_to_one",
    )
    if k.duplicated(["game_id", "play_id_kick"]).any():
        raise RuntimeError("Kickoff-to-first-snap association is not unique.")
    matched = special.merge(
        k.drop(columns=["_kick_seq"]),
        left_on=["game_id", "play_id"],
        right_on=["game_id", "play_id_kick"],
        how="left", validate="one_to_one",
    )
    if not matched["play_id_kick"].notna().all():
        raise RuntimeError("Exceptional kickoff disappeared from PBP.")
    elapsed = pd.to_numeric(
        matched["game_seconds_remaining_kick"], errors="coerce"
    ) - pd.to_numeric(matched["game_seconds_remaining_snap"], errors="coerce")
    same_score = (
        np.isclose(
            pd.to_numeric(matched["home_score_kick"], errors="coerce"),
            pd.to_numeric(matched["home_score_snap"], errors="coerce"),
            rtol=0, atol=.01,
        )
        & np.isclose(
            pd.to_numeric(matched["away_score_kick"], errors="coerce"),
            pd.to_numeric(matched["away_score_snap"], errors="coerce"),
            rtol=0, atol=.01,
        )
    )
    has_valid_snap = (
        matched["play_id_snap"].gt(matched["play_id_kick"])
        & matched["qtr_snap"].eq(4)
        & matched["yardline_100_snap"].between(1, 99)
        & elapsed.between(0, MAX_TRANSITION_SECONDS)
        & same_score
        & matched["posteam_snap"].notna()
        & matched["posteam_kick"].eq(matched["posteam_snap"])
    )
    # Re-kick occurs before another real scrimmage play. It has no
    # eligible first snap inside the CURRENT kick group.
    quick_rekick = (
        ~matched["play_id_snap"].notna()
        & matched["next_play_id_kick"].gt(matched["play_id_kick"])
        & matched["next_qtr_kick"].eq(4)
        & (
            matched["game_seconds_remaining_kick"]
            - matched["next_game_seconds_remaining_kick"]
        ).between(0, 15)
    )
    penalty = matched["outcome_group"].eq("penalty_or_rekick")
    clean = penalty & has_valid_snap & ~quick_rekick
    reason = np.select(
        [
            matched["outcome_group"].eq("kickoff_return_td"),
            matched["outcome_group"].eq("other_unmatched"),
            penalty & quick_rekick,
            penalty & clean,
            penalty & matched["play_id_snap"].notna() & ~same_score,
            penalty & matched["play_id_snap"].notna()
                & matched["posteam_kick"].ne(matched["posteam_snap"]),
        ],
        [
            "return_touchdown_requires_post_score",
            "unknown_nonordinary_kickoff",
            "re_kick_before_first_snap",
            "verified_penalty_first_snap",
            "score_changed_before_first_snap",
            "unexpected_possession_change",
        ],
        default="not_verified",
    )
    matched["elapsed_seconds"] = elapsed
    matched["verified_penalty_first_snap"] = clean.to_numpy(bool)
    matched["outcome_reason"] = reason
    matched["next_receiving_own_yardline"] = (
        100 - pd.to_numeric(matched["yardline_100_snap"], errors="coerce")
    )
    return matched.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def summarize(special: pd.DataFrame, matched: pd.DataFrame) -> dict:
    if len(special) != 55 or len(matched) != 55:
        raise ValueError("Lost exceptional kickoffs.")
    penalty = matched.loc[matched["outcome_group"].eq("penalty_or_rekick")]
    if len(penalty) != 52:
        raise RuntimeError("Penalty donor denominator was changed.")
    clean = penalty.loc[penalty["verified_penalty_first_snap"]]
    if clean[["game_id", "play_id"]].duplicated().any():
        raise RuntimeError("Duplicate verified penalty donor.")
    if not clean.empty and (
        ~clean["next_receiving_own_yardline"].between(1, 99)
    ).any():
        raise ValueError("Invalid penalty next-possession field position.")
    td = matched.loc[matched["outcome_group"].eq("kickoff_return_td")]
    if len(td) != 1 or not td["return_td_flag"].all():
        raise RuntimeError("Expected exactly one known 2025 Q4 return TD.")
    labels = {
        k: int(v)
        for k, v in matched["outcome_reason"].value_counts().items()
    }
    if sum(labels.values()) != 55:
        raise RuntimeError("Outcome-reason accounting is not exhaustive.")
    return {
        "year": 2025,
        "scope": "REG Q4 standard-35 kickoffs with >40s remaining",
        "special_total": 55,
        "penalty_total": 52,
        "return_td_total": 1,
        "other_unmatched_total": 2,
        "verified_penalty_first_snap": int(len(clean)),
        "verified_penalty_games": int(clean["game_id"].nunique()),
        "verified_penalty_avg_own_yardline": (
            float(clean["next_receiving_own_yardline"].mean())
            if len(clean) else None
        ),
        "verified_penalty_avg_elapsed_seconds": (
            float(clean["elapsed_seconds"].mean()) if len(clean) else None
        ),
        "outcome_reason_counts": labels,
        "historical_only": True,
        "not_a_future_penalty_probability": True,
        "not_a_2026_return_td_rate": True,
        "no_2026_data": True,
        "return_touchdown_examples": td[[
            "game_id", "play_id", "week", "game_seconds_remaining",
            "posteam", "desc", "return_td_flag",
        ]].to_dict(orient="records"),
    }


def main() -> None:
    os.chdir(ROOT)
    print("=== STEP 76: PENALTIES / RE-KICKS / RETURN TOUCHDOWN ===", flush=True)
    raw = nfl.load_pbp([2025]).to_pandas()
    pool = load_frozen_pool()
    special = flagged_2025(raw, pool, coverage_helper())
    matched = next_scrimmage_and_rekick(raw, special)
    result = summarize(special, matched)
    print(f"Penalties/re-kicks {result['penalty_total']} -- "
          f"verified next snaps {result['verified_penalty_first_snap']} "
          f"across {result['verified_penalty_games']} games", flush=True)
    print("Classification:", result["outcome_reason_counts"], flush=True)
    print("2025 return TD:", result["return_touchdown_examples"], flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str)+"\n")
    cols = [
        "game_id", "play_id", "week", "outcome_group", "outcome_reason",
        "posteam_kick", "posteam_snap", "game_seconds_remaining",
        "game_seconds_remaining_snap", "elapsed_seconds",
        "yardline_100_snap", "next_receiving_own_yardline",
        "verified_penalty_first_snap", "return_td_flag", "penalty_flag",
        "next_play_id_kick", "desc",
    ]
    matched[cols].to_csv(OUT_CSV, index=False, float_format="%.3f")
    print("FROZEN:", OUT_JSON.relative_to(ROOT), OUT_CSV.relative_to(ROOT), flush=True)
    print("No production change; excluded events are not probability forecasts.")


if __name__ == "__main__":
    main()
