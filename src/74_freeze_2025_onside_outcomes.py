"""Step 74: audit observed 2025 NFL onside kick recovery and possession.

This is a retrospective, read-only analysis of 2025 REG Q4 standard-own-35
kickoffs with >40 seconds on the clock. Reconcile EXACTLY to Step 73's
41 onside events. The official own_kickoff_recovery column gives outcomes;
matching to the next snap gives independently checkable possession/spot/clock.

Do not assume a 2025 onside attempt/recovery rate is predictive of a specific
game, that a failed onside follows an ordinary return distribution, or that
unmatched onside plays can be labeled failures. No 2026 data is read.
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

SOURCE = ROOT / "src/73_audit_2025_kickoff_coverage.py"
OUTPUT = ROOT / "data/kickoff_2025_onside_outcomes.json"
SAMPLES = ROOT / "data/kickoff_2025_onside_first_snaps.csv"
REQUIRED = {
    "season", "game_id", "play_id", "season_type", "qtr", "play_type",
    "posteam", "yardline_100", "game_seconds_remaining",
    "home_score", "away_score", "down", "own_kickoff_recovery", "desc",
}
SCRIMMAGE = {"run", "pass", "qb_kneel", "qb_spike", "no_play"}
MAX_SNAP_ELAPSED = 35.0


def load_step73():
    spec = importlib.util.spec_from_file_location("historical_kickoff_coverage_73", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Step 73 coverage audit unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_onside(raw: pd.DataFrame, donors: pd.DataFrame, step73) -> pd.DataFrame:
    """Share the EXACT scope and taxonomy of the frozen Step 73 denominator."""
    if raw.empty or not raw["season"].eq(2025).all():
        raise ValueError("2025-only historical PBP required")
    if missing := REQUIRED.difference(raw.columns):
        raise ValueError(f"Onside outcome fields missing: {sorted(missing)}")
    kicks = step73.mark_outcomes(step73.candidate_kickoffs(raw), donors)
    onside = kicks.loc[kicks["outcome_group"].eq("onside_attempt")].copy()
    if len(onside) != 41:
        raise RuntimeError(
            f"Frozen Step 73 onside denominator changed: {len(onside)} vs 41"
        )
    vals = pd.to_numeric(onside["own_kickoff_recovery"], errors="coerce")
    if not vals.dropna().isin([0, 1]).all():
        raise ValueError("Official onside recovery flag must be 0/1/missing.")
    onside["official_kicker_recovered"] = vals
    return onside.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def pair_first_snaps(raw: pd.DataFrame, onside: pd.DataFrame) -> pd.DataFrame:
    """Pair to the first plausible Q4 scrimmage snap BEFORE next kickoff.

    The receiving team is posteam on an ordinary kickoff. Successful onside
    kicks must have the OPPOSITE possession at the next snap. Never fill in
    missing next snaps or treat penalty-driven re-kicks as possession events.
    """
    if missing := REQUIRED.difference(raw.columns):
        raise ValueError(f"Missing first-snap fields: {sorted(missing)}")
    if raw.empty or not raw["season"].eq(2025).all():
        raise ValueError("No other season is allowed")
    if onside.empty:
        raise ValueError("No eligible onside kicks")
    df = raw.sort_values(["game_id", "play_id"]).reset_index(drop=True).copy()
    if df.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate play IDs")
    df["kick_seq"] = df["play_type"].eq("kickoff").groupby(df["game_id"]).cumsum()
    keys = ["game_id", "kick_seq"]
    kick = df.loc[df["play_type"].eq("kickoff"), keys + [
        "play_id", "posteam", "qtr", "game_seconds_remaining",
        "home_score", "away_score",
    ]].copy().rename(columns={
        "play_id": "play_id_kick",
        "posteam": "posteam_kick",
        "qtr": "qtr_kick",
        "game_seconds_remaining": "game_seconds_remaining_kick",
        "home_score": "home_score_kick",
        "away_score": "away_score_kick",
    })
    snaps = df.loc[
        df["play_type"].isin(SCRIMMAGE)
        & df["posteam"].notna()
        & df["down"].notna()
        & df["yardline_100"].notna(),
        keys + [
            "play_id", "posteam", "qtr", "yardline_100",
            "game_seconds_remaining", "home_score", "away_score",
        ]
    ].drop_duplicates(keys).rename(columns={
        "play_id": "play_id_snap",
        "posteam": "posteam_snap",
        "qtr": "qtr_snap",
        "yardline_100": "yardline_100_snap",
        "game_seconds_remaining": "game_seconds_remaining_snap",
        "home_score": "home_score_snap",
        "away_score": "away_score_snap",
    })
    paired = kick.merge(snaps, on=keys, how="left", validate="one_to_one")
    matched = onside.merge(
        paired[[
            "game_id", "play_id_kick", "posteam_kick", "posteam_snap",
            "play_id_snap", "qtr_snap", "yardline_100_snap",
            "game_seconds_remaining_snap",
            "home_score_kick", "home_score_snap",
            "away_score_kick", "away_score_snap",
        ]],
        left_on=["game_id", "play_id"],
        right_on=["game_id", "play_id_kick"],
        how="left", validate="one_to_one",
    )
    elapsed = (
        pd.to_numeric(matched["game_seconds_remaining"], errors="coerce")
        - pd.to_numeric(matched["game_seconds_remaining_snap"], errors="coerce")
    )
    score_stable = (
        np.isclose(
            pd.to_numeric(matched["home_score_kick"], errors="coerce"),
            pd.to_numeric(matched["home_score_snap"], errors="coerce"),
            atol=.01, rtol=0,
        )
        & np.isclose(
            pd.to_numeric(matched["away_score_kick"], errors="coerce"),
            pd.to_numeric(matched["away_score_snap"], errors="coerce"),
            atol=.01, rtol=0,
        )
    )
    has_next = (
        matched["play_id_snap"].gt(matched["play_id"])
        & matched["qtr_snap"].eq(4)
        & elapsed.between(0, MAX_SNAP_ELAPSED)
        & matched["yardline_100_snap"].between(1, 99)
        & score_stable
        & matched["posteam_kick"].notna()
        & matched["posteam_snap"].notna()
    )
    inferred = (
        matched["posteam_snap"].ne(matched["posteam_kick"]).astype(bool)
    )
    official = matched["official_kicker_recovered"]
    flag_known = official.notna()
    no_penalty = ~matched["penalty_flag"] & ~matched["return_td_flag"]
    consistent = has_next & flag_known & (
        inferred.to_numpy(int) == official.fillna(-1).to_numpy(int)
    )
    reliable = consistent & no_penalty
    matched["kickoff_to_snap_elapsed"] = elapsed
    matched["post_snap_kicker_recovered"] = inferred
    matched["matched_reliable"] = reliable.astype(bool)
    matched["unmatched_reason"] = np.select(
        [
            matched["penalty_flag"].to_numpy(bool),
            matched["return_td_flag"].to_numpy(bool),
            ~flag_known.to_numpy(bool),
            ~has_next,
            (~consistent & has_next & flag_known),
        ],
        [
            "penalty_on_onside_attempt", "scoring_return",
            "unknown_recovery_flag", "no_clean_next_snap",
            "official_flag_vs_first_snap_disagreement",
        ],
        default="verified",
    )
    return matched.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def summarize(onside: pd.DataFrame, matched: pd.DataFrame) -> dict:
    if len(onside) != len(matched) or len(onside) != 41:
        raise ValueError("Onside audit lost denominator rows")
    recovery = onside["official_kicker_recovered"]
    known = recovery.notna()
    confirmed = int(recovery.eq(1).sum())
    reliable = matched.loc[matched["matched_reliable"]].copy()
    checked = reliable["official_kicker_recovered"].astype(int)
    distinct_groups = {
        label: {
            "n": int((checked == result).sum()),
            "mean_elapsed": (
                float(reliable.loc[checked == result, "kickoff_to_snap_elapsed"].mean())
                if (checked == result).any() else None
            ),
            "mean_next_own_yardline": (
                float((100 - reliable.loc[
                    checked == result, "yardline_100_snap"
                ]).mean())
                if (checked == result).any() else None
            ),
        }
        for label, result in (("kicker_recovers", 1), ("receiver_recovers", 0))
    }
    if sum(int(n) for n in matched["unmatched_reason"].value_counts()) != len(onside):
        raise RuntimeError("Every eligible attempt must have one classification")
    return {
        "year": 2025,
        "all_on_side_eligible": len(onside),
        "games": int(onside["game_id"].nunique()),
        "scope": "2025 REG Q4 standard-35 onside attempts, >40sec left",
        "flag_known": int(known.sum()),
        "flag_missing": int((~known).sum()),
        "official_kicker_recoveries": confirmed,
        "official_recovery_share_of_known": (
            float(confirmed / known.sum()) if known.any() else None
        ),
        "matched_reliable": int(matched["matched_reliable"].sum()),
        "exclusion_reason_counts": {
            str(label): int(count)
            for label, count in matched["unmatched_reason"].value_counts().items()
        },
        "reliable_first_snap": distinct_groups,
        "historical_only": True,
        "not_a_policy_probability": True,
        "no_2026_data": True,
    }


def main():
    os.chdir(ROOT)
    print("=== STEP 74: 2025 ONSIDE RECOVERY / POSSESSION AUDIT ===", flush=True)
    pbp = nfl.load_pbp([2025]).to_pandas()
    pool = load_frozen_pool()
    step73 = load_step73()
    eligible = classify_onside(pbp, pool, step73)
    paired = pair_first_snaps(pbp, eligible)
    results = summarize(eligible, paired)
    print("All eligible onside attempts:", results["all_on_side_eligible"], flush=True)
    print("Known recovery flags:", results["flag_known"], flush=True)
    print("Kicking team recovered:", results["official_kicker_recoveries"], flush=True)
    print("Verified matching next snaps:", results["matched_reliable"], flush=True)
    print("Other cases:", results["exclusion_reason_counts"], flush=True)
    print("Conditional next-snap observations:", results["reliable_first_snap"], flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2)+"\n")
    cols = [
        "game_id", "play_id", "week", "posteam_kick", "posteam_snap",
        "official_kicker_recovered", "game_seconds_remaining",
        "game_seconds_remaining_snap", "kickoff_to_snap_elapsed",
        "yardline_100_snap", "penalty_flag", "return_td_flag",
        "matched_reliable", "unmatched_reason",
    ]
    paired[cols].to_csv(SAMPLES, index=False, float_format="%.3f")
    print("Frozen:", OUTPUT.relative_to(ROOT), SAMPLES.relative_to(ROOT), flush=True)
    print("No production changes. No attempt probabilities or causal policy claims.")


if __name__ == "__main__":
    main()
