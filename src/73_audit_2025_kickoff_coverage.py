"""Step 73: expose what the 2025 ordinary Q4 kickoff preview EXCLUDES.

Read actual public 2025 REG play-by-play, freeze an auditable denominator for
the existing 488 empirical donor plays, and enumerate excluded events.
No numerical probability for an unmodeled outcome is inferred from this
descriptive inventory. No 2026 data. No production model changes.

A touchdown or onside attempt is NOT interchangeable with a normal kickoff:
these outcomes need a different score/possession state. In particular,
do not sample a regular return in place of an onside recovery.
"""

from __future__ import annotations

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

REPORT = ROOT / "data/kickoff_2025_q4_coverage.json"
REPORT_YEAR = 2025
CLOCK_FLOOR = 40.0
WEEKS = range(1, 19)
COLS = {
    "season", "season_type", "game_id", "play_id", "play_type", "qtr",
    "week", "yardline_100", "game_seconds_remaining", "desc",
    "return_touchdown", "penalty", "posteam_score", "defteam_score",
    "posteam",
}


def candidate_kickoffs(raw: pd.DataFrame) -> pd.DataFrame:
    """All Q4 standard-35 kickoffs with >40sec; identical scope to donor pool."""
    missing = COLS.difference(raw.columns)
    if missing:
        raise ValueError(f"Cannot determine kickoff exclusions, fields missing: {sorted(missing)}")
    if raw.empty or not raw["season"].eq(REPORT_YEAR).all():
        raise ValueError("Only 2025 NFL play-by-play is allowed.")
    df = raw.loc[
        raw["season_type"].eq("REG")
        & raw["play_type"].eq("kickoff")
        & pd.to_numeric(raw["qtr"], errors="coerce").eq(4)
        & pd.to_numeric(raw["yardline_100"], errors="coerce").eq(35)
        & pd.to_numeric(raw["game_seconds_remaining"], errors="coerce").gt(CLOCK_FLOOR)
        & pd.to_numeric(raw["week"], errors="coerce").between(1, 18)
    ].copy()
    if df.empty:
        raise RuntimeError("No Q4 kickoffs meet the frozen donor scope.")
    if df[["game_id", "play_id"]].isna().any().any():
        raise ValueError("Missing event identifiers.")
    if df.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate 2025 kickoff IDs.")
    return df


def mark_outcomes(kicks: pd.DataFrame, donors: pd.DataFrame) -> pd.DataFrame:
    """Check all 488 frozen donors exist, then classify unmatched plays.

    Onside, return TD, penalty, and unmatched are mutually exclusive for
    reporting, applied in that order; overlaps are reported separately.
    """
    if kicks.empty or donors.empty:
        raise ValueError("Need all kickoffs and the reviewed 2025 donor snapshot.")
    key = ["game_id", "play_id"]
    verified = donors.rename(columns={"kick_play_id": "play_id"})
    ordinary_keys = verified[key].copy()
    ordinary_keys["game_id"] = ordinary_keys["game_id"].astype(str)
    ordinary_keys["play_id"] = pd.to_numeric(ordinary_keys["play_id"])
    if ordinary_keys.duplicated(key).any():
        raise ValueError("Frozen donors contain duplicate kickoff IDs.")

    frame = kicks.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["play_id"] = pd.to_numeric(frame["play_id"], errors="raise")
    annotated = frame.merge(
        ordinary_keys.assign(_in_donor_pool=True), on=key, how="left",
        validate="one_to_one",
    )
    annotated["modeled_ordinary"] = annotated["_in_donor_pool"].fillna(False).astype(bool)
    if int(annotated["modeled_ordinary"].sum()) != len(donors):
        raise RuntimeError(
            "Frozen 2025 donor IDs do not match the full 2025 PBP universe; "
            "never publish an incorrect coverage fraction."
        )

    desc = annotated["desc"].fillna("").astype(str)
    onside = desc.str.contains(r"\bonside\b", case=False, regex=True)
    return_td = pd.to_numeric(
        annotated["return_touchdown"], errors="coerce"
    ).fillna(0).eq(1)
    penalty = (
        pd.to_numeric(annotated["penalty"], errors="coerce")
        .fillna(0).eq(1)
        | desc.str.contains("penalty", case=False, regex=False)
    )
    flags = pd.DataFrame({
        "onside_flag": onside.to_numpy(bool),
        "return_td_flag": return_td.to_numpy(bool),
        "penalty_flag": penalty.to_numpy(bool),
    }, index=annotated.index)
    annotated = pd.concat([annotated, flags], axis=1)
    ordinary = annotated["modeled_ordinary"]
    if annotated.loc[ordinary, ["onside_flag", "return_td_flag", "penalty_flag"]].any().any():
        raise RuntimeError("Reviewed ordinary donor was classified as a special kickoff.")
    annotated["outcome_group"] = np.select(
        [
            ordinary.to_numpy(),
            onside.to_numpy(),
            return_td.to_numpy(),
            penalty.to_numpy(),
        ],
        [
            "modeled_ordinary",
            "onside_attempt",
            "kickoff_return_td",
            "penalty_or_rekick",
        ],
        default="other_unmatched",
    )
    annotated["receiver_margin"] = (
        pd.to_numeric(annotated["posteam_score"], errors="coerce")
        - pd.to_numeric(annotated["defteam_score"], errors="coerce")
    )
    return annotated


def cluster_coverage_interval(rows: pd.DataFrame, *,
                              repetitions: int = 1500, seed: int = 73
                              ) -> list[float] | None:
    """Bootstrap whole games, conditional on the observed source population."""
    if len(rows) < 30 or rows["game_id"].nunique() < 10:
        return None
    games, inv = np.unique(rows["game_id"].astype(str), return_inverse=True)
    modeled = np.bincount(
        inv, weights=rows["modeled_ordinary"].to_numpy(float),
        minlength=len(games),
    )
    total = np.bincount(inv, minlength=len(games))
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(games), size=(repetitions, len(games)))
    sampled = modeled[selections].sum(axis=1) / total[selections].sum(axis=1)
    return [float(x) for x in np.quantile(sampled, [0.025, 0.975])]


def summarize_group(name: str, rows: pd.DataFrame) -> dict:
    total = len(rows)
    groups = rows["outcome_group"].value_counts()
    counts = {
        label: int(groups.get(label, 0))
        for label in (
            "modeled_ordinary", "onside_attempt", "kickoff_return_td",
            "penalty_or_rekick", "other_unmatched"
        )
    }
    if sum(counts.values()) != total:
        raise RuntimeError("Kickoff taxonomy failed to account for every play.")
    return {
        "name": name,
        "total_kickoffs": total,
        "games": int(rows["game_id"].nunique()),
        "counts": counts,
        "ordinary_share": (
            float(counts["modeled_ordinary"] / total) if total else None
        ),
        "ordinary_share_game_cluster_ci95": cluster_coverage_interval(rows),
        "low_support": bool(total < 30),
        "special_flag_overlaps": int(
            (rows[["onside_flag", "return_td_flag", "penalty_flag"]]
             .sum(axis=1) > 1).sum()
        ) if total else 0,
    }


def build_report(raw: pd.DataFrame, donors: pd.DataFrame) -> dict:
    universe = candidate_kickoffs(raw)
    classified = mark_outcomes(universe, donors)
    close_clock = pd.to_numeric(
        classified["game_seconds_remaining"], errors="coerce"
    ).between(40, 300, inclusive="both")
    receiver_leads = classified["receiver_margin"].between(1, 3)
    groups = [
        ("2025 Q4 STANDARD 35, >40 SEC", classified),
        ("2025 Q4 FINAL 5 MIN, >40 SEC", classified.loc[close_clock]),
        ("Q4 FINAL 5, RECEIVER LEADS 1-3", classified.loc[close_clock & receiver_leads]),
        ("Q4 FINAL 5, RECEIVER TRAILS 1-3",
         classified.loc[close_clock & classified["receiver_margin"].between(-3, -1)]),
    ]
    report = {
        "data_year": REPORT_YEAR,
        "regulation_only": True,
        "week_range": [1, 18],
        "kickoff_spot_yardline_100": 35,
        "kickoff_seconds_floor_exclusive": CLOCK_FLOOR,
        "model": "2025 frozen ordinary kickoff-to-first-snap donor preview",
        "date_context": "2025 previously inspected; not a fresh holdout",
        "scope": (
            "Historical selected Q4 standard-35 kickoffs with >40 seconds. "
            "Excludes special kickoff spots, non-Q4, and endgame <=40 sec."
        ),
        "classification_note": (
            "Exclusive priority: matched ordinary, onside description, "
            "official return TD, penalty, other unmatched. Other unmatched "
            "may include clock expiry, missing first snap, turnover, penalties "
            "not captured by text/flag, or unusual kicks. No invented resolution."
        ),
        "not_a_forecast": True,
        "no_2026_outcomes": True,
        "groups": [summarize_group(name, group) for name, group in groups],
    }
    if report["groups"][0]["counts"]["modeled_ordinary"] != len(donors):
        raise RuntimeError("Denominator and frozen empirical pool disagree.")
    return report


def main() -> None:
    os.chdir(ROOT)
    print("=== STEP 73: 2025 Q4 KICKOFF COVERAGE AUDIT ===", flush=True)
    print("Read-only modeling: no production changes, no 2026 data.", flush=True)
    pbp = nfl.load_pbp([REPORT_YEAR]).to_pandas()
    pool = load_frozen_pool()
    report = build_report(pbp, pool)
    for group in report["groups"]:
        print(f"\n{group['name']}: n={group['total_kickoffs']} "
              f"games={group['games']} "
              f"modeled={group['counts']['modeled_ordinary']} "
              f"share={group['ordinary_share']}", flush=True)
        print("  exclusive categories:", group["counts"], flush=True)
        print("  game-cluster 95% CI:", group["ordinary_share_game_cluster_ci95"],
              flush=True)
        if group["low_support"]:
            print("  LOW SUPPORT: not enough kickoffs for inference.", flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2)+"\n")
    print(f"\nFROZEN DESCRIPTIVE COVERAGE: {REPORT.relative_to(ROOT)}", flush=True)
    print("Remaining unsupported kicks CANNOT be silently treated as ordinary.")
    print("No onside/recovery/return-TD outcome predictions have been fitted.")


if __name__ == "__main__":
    main()
