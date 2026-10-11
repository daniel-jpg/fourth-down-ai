"""Read-only kickoff-score-context diagnostics on frozen 2024 / held-out 2025 PBP.

Run from repository root:
  ./.venv/bin/python src/56_audit_2025_kickoff_context.py

Descriptive holdout evaluation ONLY. Do not fit a production touchback rate,
change a model, or write an artifact based on these results.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_receiver_margin(pairs: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Use pre-kickoff scoreboard from the RECEIVING team's perspective.

    Check score_differential against posteam_score-defteam_score where both
    are present; if the feed disagrees, fail instead of drawing a false conclusion.
    """
    required = {"game_id", "play_id", "play_type", "posteam"}
    if not required.issubset(pbp.columns):
        raise ValueError("PBP missing game/play/team identifiers")
    if "score_differential" not in pbp and not {"posteam_score", "defteam_score"}.issubset(pbp.columns):
        raise ValueError("PBP missing receiving-team score differential")
    kick = pbp.loc[pbp["play_type"].eq("kickoff")].copy()
    if "score_differential" in kick:
        margin = pd.to_numeric(kick["score_differential"], errors="coerce")
    else:
        margin = pd.Series(np.nan, index=kick.index)
    if {"posteam_score", "defteam_score"}.issubset(kick.columns):
        score_margin = (
            pd.to_numeric(kick["posteam_score"], errors="coerce")
            - pd.to_numeric(kick["defteam_score"], errors="coerce")
        )
        disagreement = margin.notna() & score_margin.notna() & (margin-score_margin).abs().gt(0.01)
        if disagreement.any():
            raise ValueError(
                f"Score differential disagrees with receiving-team scores on "
                f"{int(disagreement.sum())} kickoff rows; cannot stratify safely"
            )
        margin = margin.fillna(score_margin)

    key = kick[["game_id", "play_id"]].rename(columns={"play_id": "kick_play_id"}).copy()
    key["receiver_margin"] = margin.to_numpy()
    if key.duplicated(["game_id", "kick_play_id"]).any():
        raise ValueError("Kickoff identifiers are not unique")
    joined = pairs.merge(key, on=["game_id", "kick_play_id"], how="left", validate="one_to_one")
    if joined["receiver_margin"].isna().any():
        print(f"WARNING: score unavailable for {int(joined['receiver_margin'].isna().sum())} kickoffs")
    return joined


def margin_label(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value <= -4:
        return "receiver trailing 4+"
    if value < 0:
        return "receiver trailing 1-3"
    if value == 0:
        return "tied"
    if value <= 3:
        return "receiver leading 1-3"
    return "receiver leading 4+"


def context_stats(subset: pd.DataFrame) -> pd.DataFrame:
    """Touchback rate conditional on ordinary kickoffs, with exclusions visible."""
    d = subset.copy()
    d["group"] = d["receiver_margin"].map(margin_label)
    normal = d["status"].isin(["touchback", "return"])
    d["ordinary"] = normal
    d["tb"] = d["status"].eq("touchback")
    d["return"] = d["status"].eq("return")
    d["onside"] = d["status"].eq("onside")
    d["other"] = d["status"].eq("other/unmatched")
    rows = []
    for label in ("receiver leading 1-3", "receiver trailing 1-3",
                  "receiver leading 4+", "receiver trailing 4+", "tied", "unknown"):
        x = d.loc[d["group"].eq(label)]
        if x.empty:
            continue
        n = int(x["ordinary"].sum())
        tb = int(x["tb"].sum())
        ret = int(x["return"].sum())
        if n:
            z = 1.959963984540054
            prop = tb/n
            scale = 1 + z*z/n
            center = (prop + z*z/(2*n))/scale
            radius = z/scale * math.sqrt(prop*(1-prop)/n + z*z/(4*n*n))
            interval = (max(0.0, center-radius), min(1.0, center+radius))
        else:
            interval = (float("nan"), float("nan"))
        rows.append({
            "receiving_team_score": label,
            "kickoffs": len(x),
            "ordinary": n,
            "touchbacks": tb,
            "returns": ret,
            "onside": int(x["onside"].sum()),
            "other": int(x["other"].sum()),
            "touchback_rate": tb/n if n else float("nan"),
            "ci_low": interval[0],
            "ci_high": interval[1],
            "avg_start": x.loc[x["ordinary"], "own"].mean(),
            "avg_clock": x.loc[x["ordinary"], "elapsed"].mean(),
            "low_support": n < 30,
        })
    return pd.DataFrame(rows)


def report(title: str, rows: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    for rec in rows.to_dict("records"):
        rate = f"{rec['touchback_rate']:.1%}" if pd.notna(rec["touchback_rate"]) else "n/a"
        ci = (f"[{rec['ci_low']:.0%},{rec['ci_high']:.0%}]"
              if pd.notna(rec["ci_low"]) else "n/a")
        avg = f"{rec['avg_start']:.2f}" if pd.notna(rec["avg_start"]) else "n/a"
        clock = f"{rec['avg_clock']:.2f}" if pd.notna(rec["avg_clock"]) else "n/a"
        warning = " LOW SAMPLE" if rec["low_support"] else ""
        print(f"{rec['receiving_team_score']:24s} "
              f"ordinary={rec['ordinary']:3d} TB={rec['touchbacks']:3d} "
              f"returns={rec['returns']:3d} TB-rate={rate:>6s} CI={ci:>9s} "
              f"own={avg:>5s} clock={clock:>5s}s "
              f"onside={rec['onside']:2d} other={rec['other']:2d}{warning}")


def main() -> None:
    import nflreadpy as nfl

    audit_script = runpy.run_path(str(ROOT / "src/55_audit_2025_kickoffs.py"))
    pairing = audit_script["audit_kickoff_pairs"]
    for year in (2024, 2025):
        print(f"\nLoading {year} {'holdout' if year == 2025 else 'development'} play-by-play...")
        pbp = nfl.load_pbp([year]).sort(["game_id", "play_id"]).to_pandas()
        joined = add_receiver_margin(pairing(pbp, year), pbp)
        q4 = joined.loc[joined["kick_qtr"].eq(4)]
        last5 = q4.loc[q4["kick_seconds"].le(300)]
        report(f"{year} Q4 BY RECEIVING TEAM SCORE", context_stats(q4))
        report(f"{year} Q4 LAST FIVE MINUTES BY RECEIVING TEAM SCORE", context_stats(last5))

    print("\nHOLDOUT DIAGNOSTIC ONLY. No 2025 values fitted or promoted to production.")
    print("Receiver leading by 1-3 corresponds to the Rams made-FG continuation;")
    print("receiver trailing by 1-3 corresponds to many Rams TD+XP continuations.")
    print("Onside/exceptions are shown but NOT modeled by ordinary kickoff sampler.")


if __name__ == "__main__":
    main()
