"""Read-only 2025 kickoff HOLDOUT audit against frozen 2024 donor assumptions.

Run from the repository root:
  ./.venv/bin/python src/55_audit_2025_kickoffs.py

This script MUST NOT write datasets, fit models, select a production
kickoff touchback rate, or modify any win-probability estimates. Its pairing
helper also accepts 2026 for independent evaluation of 2025-trained models. 2025
is evaluation-only. Ordinary kickoff transitions are observational;
onside kicks, unusual penalties, return touchdowns and recovered kicks
may appear under 'other/unmatched' rather than in the ordinary sample.
"""

from __future__ import annotations

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kickoff_transition import build_2024_pool  # noqa: E402

REQUIRED = {
    "season", "game_id", "play_id", "play_type", "qtr",
    "game_seconds_remaining", "posteam", "yardline_100", "down",
    "touchback", "desc",
}
ORDINARY_SNAP_TYPES = {"run", "pass", "no_play", "qb_kneel", "qb_spike"}


def audit_kickoff_pairs(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """Classify every kickoff, retaining unmatched and excluded rows for coverage.

    The initial kickoff's posteam is the receiving team, consistent with
    nflverse play-by-play. Each candidate is matched to the FIRST valid
    subsequent offensive snap BEFORE the next kickoff in its own game.
    """
    if season not in (2024, 2025, 2026):
        raise ValueError("Audit supports 2024 development, 2025 context training, and 2026 evaluation only.")
    missing = REQUIRED.difference(pbp.columns)
    if missing:
        raise ValueError(f"Missing required PBP columns: {sorted(missing)}")
    if pbp.empty or not pd.to_numeric(pbp["season"], errors="coerce").eq(season).all():
        raise ValueError("PBP must contain exactly one requested season.")

    df = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True).copy()
    df["kick_seq"] = df["play_type"].eq("kickoff").groupby(df["game_id"]).cumsum()

    kickoff = df.loc[df["play_type"].eq("kickoff"), [
        "game_id", "play_id", "kick_seq", "qtr", "game_seconds_remaining",
        "posteam", "touchback", "desc",
    ]].rename(columns={
        "play_id": "kick_play_id", "qtr": "kick_qtr",
        "game_seconds_remaining": "kick_seconds", "posteam": "receiver",
    })

    snap = df.loc[
        df["play_type"].isin(ORDINARY_SNAP_TYPES)
        & df["down"].notna() & df["yardline_100"].notna()
        & df["posteam"].notna(),
        ["game_id", "play_id", "kick_seq", "qtr", "game_seconds_remaining",
         "posteam", "yardline_100"],
    ].drop_duplicates(["game_id", "kick_seq"]).rename(columns={
        "play_id": "snap_play_id", "qtr": "snap_qtr",
        "game_seconds_remaining": "snap_seconds", "posteam": "snap_team",
        "yardline_100": "snap_yardline",
    })

    pairs = kickoff.merge(snap, on=["game_id", "kick_seq"], how="left",
                          validate="one_to_one")
    pairs["elapsed"] = pairs["kick_seconds"] - pairs["snap_seconds"]
    pairs["own"] = 100.0 - pairs["snap_yardline"]

    is_onside = pairs["desc"].fillna("").str.contains(
        "onside", case=False, regex=False)
    valid_snap = (
        pairs["snap_play_id"].gt(pairs["kick_play_id"])
        & pairs["kick_qtr"].eq(pairs["snap_qtr"])
        & pairs["elapsed"].between(0, 35)
        & pairs["own"].between(1, 99)
        & pairs["receiver"].notna()
        & pairs["receiver"].eq(pairs["snap_team"])
    )
    tb_flag = pd.to_numeric(pairs["touchback"], errors="coerce").fillna(0).eq(1)
    tb_own = (20.0, 30.0) if season == 2024 else (20.0, 35.0)
    touchback = (valid_snap & tb_flag & pairs["own"].isin(tb_own)).fillna(False)
    returned = (valid_snap & ~tb_flag & pairs["elapsed"].between(1, 35)).fillna(False)

    pairs["status"] = np.select(
        [is_onside, touchback, returned],
        ["onside", "touchback", "return"],
        default="other/unmatched",
    )
    pairs["season"] = season
    return pairs[[
        "season", "game_id", "kick_play_id", "kick_qtr", "kick_seconds",
        "status", "own", "elapsed", "touchback", "desc",
    ]].reset_index(drop=True)


def _q4_counts_match_frozen_2024(pbp: pd.DataFrame, audited: pd.DataFrame) -> None:
    """Guard against accidental drift from the existing 2024 donor extractor."""
    reference = build_2024_pool(pbp)
    seen = audited.loc[audited["status"].isin(["touchback", "return"])]
    columns = ["game_id", "quarter", "transition_type", "own_yardline",
               "elapsed_seconds"]
    a = seen.rename(columns={
        "kick_qtr": "quarter", "status": "transition_type",
        "own": "own_yardline", "elapsed": "elapsed_seconds",
    })[columns].sort_values(columns).reset_index(drop=True)
    b = reference[columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False,
                                  obj="Audit must exactly match frozen 2024 pool")


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Approximate 95% Wilson interval for the observed ordinary touchback rate."""
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    d = 1 + z*z/total
    center = (p + z*z/(2*total))/d
    offset = z/d * math.sqrt(p*(1-p)/total + z*z/(4*total*total))
    return center-offset, center+offset


def metrics(pairs: pd.DataFrame) -> dict:
    normal = pairs.loc[~pairs["status"].eq("onside")]
    valid = normal.loc[normal["status"].isin(["touchback", "return"])]
    tb = valid.loc[valid["status"].eq("touchback")]
    ret = valid.loc[valid["status"].eq("return")]
    p = (len(tb)/len(valid)) if len(valid) else float("nan")
    low, high = wilson_interval(len(tb), len(valid))
    return {
        "kickoffs": len(pairs),
        "onside": int(pairs["status"].eq("onside").sum()),
        "ordinary": len(valid),
        "other_unmatched": int(normal["status"].eq("other/unmatched").sum()),
        "coverage_pct": 100*len(valid)/len(normal) if len(normal) else float("nan"),
        "touchbacks": len(tb),
        "returns": len(ret),
        "touchback_pct": 100*p,
        "touchback_ci_low": 100*low,
        "touchback_ci_high": 100*high,
        "tb_20": int(tb["own"].eq(20).sum()),
        "tb_30": int(tb["own"].eq(30).sum()),
        "tb_35": int(tb["own"].eq(35).sum()),
        "return_own_mean": ret["own"].mean(),
        "return_own_p10": ret["own"].quantile(0.10),
        "return_own_p90": ret["own"].quantile(0.90),
        "return_elapsed_mean": ret["elapsed"].mean(),
        "return_elapsed_median": ret["elapsed"].median(),
        "all_own_mean": valid["own"].mean(),
        "all_elapsed_mean": valid["elapsed"].mean(),
    }


def _format_metric(value, digits=2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> None:
    import nflreadpy as nfl  # Deferred so synthetic unit tests need no NFL download.

    print("Loading 2024 development and 2025 HOLDOUT NFL play-by-play...")
    audits = {}
    for year in (2024, 2025):
        frame = nfl.load_pbp([year]).sort(["game_id", "play_id"]).to_pandas()
        print(f"{year} PBP rows: {len(frame):,}")
        audits[year] = audit_kickoff_pairs(frame, year)
        if year == 2024:
            _q4_counts_match_frozen_2024(frame, audits[year])
            print("2024 parity check against frozen donor extractor: PASS")

    for name, include in (
        ("ALL QUARTERS", lambda d: d),
        ("REGULATION Q4", lambda d: d.loc[d["kick_qtr"].eq(4)]),
        ("Q4 LAST 5 MINUTES", lambda d: d.loc[
            d["kick_qtr"].eq(4) & d["kick_seconds"].le(300)]),
    ):
        print(f"\n=== {name} ===")
        for year in (2024, 2025):
            subset = include(audits[year])
            m = metrics(subset)
            print(
                f"{year}: kickoff_rows={m['kickoffs']} onside={m['onside']} "
                f"ordinary={m['ordinary']} other/unmatched={m['other_unmatched']} "
                f"coverage={_format_metric(m['coverage_pct'])}%"
            )
            print(
                f"      TB={m['touchbacks']} returns={m['returns']} "
                f"TB rate={_format_metric(m['touchback_pct'])}% "
                f"(approx 95% interval {_format_metric(m['touchback_ci_low'])}-"
                f"{_format_metric(m['touchback_ci_high'])}%) "
                f"TB positions own20/30/35={m['tb_20']}/{m['tb_30']}/{m['tb_35']}"
            )
            print(
                f"      Return start avg={_format_metric(m['return_own_mean'])}, "
                f"p10/p90={_format_metric(m['return_own_p10'])}/"
                f"{_format_metric(m['return_own_p90'])}; "
                f"return clock mean/median={_format_metric(m['return_elapsed_mean'])}/"
                f"{_format_metric(m['return_elapsed_median'])} sec"
            )
            print(
                f"      Overall receiving start avg={_format_metric(m['all_own_mean'])}; "
                f"clock mean={_format_metric(m['all_elapsed_mean'])} sec"
            )

    q24 = metrics(audits[2024].loc[audits[2024]["kick_qtr"].eq(4)])
    q25 = metrics(audits[2025].loc[audits[2025]["kick_qtr"].eq(4)])
    if q24["returns"] and q24["touchbacks"] and q25["ordinary"]:
        # Not a prospective forecast: the 2025 observed TB rate is supplied
        # deliberately to isolate errors in RETURN and TB20 conditional outcomes.
        p25 = q25["touchbacks"]/q25["ordinary"]
        p20_24 = q24["tb_20"]/q24["touchbacks"]
        expected_own = (
            p25 * (p20_24*20 + (1-p20_24)*35)
            + (1-p25)*q24["return_own_mean"]
        )
        expected_clock = (1-p25)*q24["return_elapsed_mean"]
        print("\n=== DIAGNOSTIC USING OBSERVED 2025 Q4 TOUCHBACK RATE ===")
        print("This is NOT a forecast or a fitted production input.")
        print("2024 donors + actual 2025 TB rate -> expected start own",
              _format_metric(expected_own),
              "; actual 2025 own", _format_metric(q25["all_own_mean"]))
        print("2024 donors + actual 2025 TB rate -> expected clock",
              _format_metric(expected_clock),
              "; actual 2025 clock", _format_metric(q25["all_elapsed_mean"]))

    for year in (2024, 2025):
        odd = audits[year].loc[
            audits[year]["kick_qtr"].eq(4)
            & audits[year]["status"].eq("other/unmatched")
        ]
        print(f"\n=== {year} Q4 UNMATCHED / EXCEPTIONAL EXAMPLES (up to 5) ===")
        for _, row in odd.head(5).iterrows():
            print(f"game={row['game_id']} play={row['kick_play_id']} "
                  f"desc={str(row['desc'])[:140]}")

    print("\nREAD ONLY. No data/models updated; 2025 remains held out.")
    print("Ordinary-kickoff samples exclude rare special outcomes and may not")
    print("represent post-score kickoffs conditional on game context.")


if __name__ == "__main__":
    main()
