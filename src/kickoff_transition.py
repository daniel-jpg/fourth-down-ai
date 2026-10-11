"""Experimental, development-only kickoff continuation sampler.

2024 is the sole donor season. Its *conditional* return outcomes are not a
calibrated forecast of 2025+ tactics. Callers MUST supply a touchback rate;
this module must not silently change production fourth-down decisions.

Scope: ordinary Q4 kickoffs after a score, with the receiving team taking the
next first down. Excludes onside recoveries, return TDs, and unusual penalties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


POOL_COLUMNS = [
    "season", "game_id", "quarter", "transition_type",
    "own_yardline", "elapsed_seconds",
]


def build_2024_pool(pbp: pd.DataFrame) -> pd.DataFrame:
    """Extract ordinary 2024 kickoff -> first receiving-team snap transitions.

    Both kickoff rows and subsequent scrimmage rows use `posteam` to mean the
    *receiving team* for ordinary kickoffs. No future game data is used to
    predict an outcome: this constructs the empirical donor distribution.
    """
    required = {
        "game_id", "play_id", "play_type", "qtr", "season",
        "game_seconds_remaining", "posteam", "yardline_100", "down",
        "touchback", "desc",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"Missing NFL PBP columns: {sorted(missing)}")

    df = pbp.copy()
    if not df["season"].dropna().eq(2024).all() or df["season"].isna().any():
        raise ValueError("Development kickoff pool must use 2024 only.")

    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    df["kick_seq"] = df["play_type"].eq("kickoff").groupby(df["game_id"]).cumsum()

    kick = df.loc[df["play_type"].eq("kickoff"), [
        "game_id", "kick_seq", "qtr", "game_seconds_remaining",
        "posteam", "touchback", "desc",
    ]].rename(columns={
        "qtr": "kick_qtr", "game_seconds_remaining": "kick_clock",
        "posteam": "receiver",
    })
    kick = kick.loc[
        ~kick["desc"].fillna("").str.contains("onside", case=False, regex=False)
    ]

    scrimmage = df.loc[
        df["play_type"].isin(["run", "pass", "no_play", "qb_kneel", "qb_spike"])
        & df["down"].notna() & df["yardline_100"].notna()
        & df["posteam"].notna(),
        ["game_id", "kick_seq", "qtr", "game_seconds_remaining",
         "posteam", "yardline_100"],
    ].drop_duplicates(["game_id", "kick_seq"]).rename(columns={
        "qtr": "snap_qtr", "game_seconds_remaining": "snap_clock",
        "posteam": "snap_team", "yardline_100": "snap_yardline",
    })

    joined = kick.merge(scrimmage, on=["game_id", "kick_seq"], how="inner")
    joined["elapsed"] = joined["kick_clock"] - joined["snap_clock"]
    joined["own"] = 100.0 - joined["snap_yardline"]
    joined = joined.loc[
        joined["kick_qtr"].eq(joined["snap_qtr"])
        & joined["elapsed"].between(0, 35)
        & joined["own"].between(1, 99)
        & joined["receiver"].notna()
        & joined["receiver"].eq(joined["snap_team"])
    ].copy()

    tb = pd.to_numeric(joined["touchback"], errors="coerce").fillna(0).eq(1)
    valid_tb = tb & joined["own"].isin([20.0, 30.0])
    valid_return = ~tb & joined["elapsed"].between(1, 35)
    joined = joined.loc[valid_tb | valid_return].copy()
    joined["transition_type"] = np.where(
        valid_tb.loc[joined.index], "touchback", "return"
    )

    return pd.DataFrame({
        "season": 2024,
        "game_id": joined["game_id"].to_numpy(),
        "quarter": joined["kick_qtr"].to_numpy(dtype=int),
        "transition_type": joined["transition_type"].to_numpy(),
        "own_yardline": joined["own"].to_numpy(dtype=float),
        "elapsed_seconds": joined["elapsed"].to_numpy(dtype=float),
    }, columns=POOL_COLUMNS)


def _validated_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if set(POOL_COLUMNS).difference(pool.columns):
        raise ValueError("Pool is missing required columns.")
    if pool.empty or not pool["season"].eq(2024).all():
        raise ValueError("Only nonempty 2024 development donor pools are supported.")
    if not pool["transition_type"].isin(["return", "touchback"]).all():
        raise ValueError("Unexpected kickoff transition type.")
    if not pool["own_yardline"].between(1, 99).all():
        raise ValueError("Invalid receiving-team yardline in kickoff pool.")
    if not pool["elapsed_seconds"].between(0, 35).all():
        raise ValueError("Invalid kickoff clock runoff in pool.")
    if not pool.loc[
        pool["transition_type"].eq("touchback"), "own_yardline"
    ].isin([20.0, 30.0]).all():
        raise ValueError("2024 touchback donors must end at own 20 or 30.")
    return pool


def resolve_q4_kickoffs(
    states: list[dict],
    pool: pd.DataFrame,
    *,
    touchback_rate: float,
    rng: np.random.Generator,
    rules_year: int = 2026,
) -> list[dict]:
    """Return copied states, sampling ordinary post-score Q4 kickoff continuations.

    No action is taken outside regulation Q4, for non-scoring kickoff states,
    or when the quarter is already at 0:00. Touchback rate is an explicit
    *scenario assumption*, never learned from the 2025 holdout.
    """
    if not np.isfinite(touchback_rate) or not 0 <= touchback_rate <= 1:
        raise ValueError("touchback_rate must be between 0 and 1.")
    if rules_year < 2024:
        raise ValueError("This pool only represents the 2024+ dynamic kickoff.")

    p = _validated_pool(pool)
    q4 = p.loc[p["quarter"].eq(4)]
    if (q4["transition_type"].eq("return").sum() >= 50
            and q4["transition_type"].eq("touchback").sum() >= 20):
        p = q4

    tb = p.loc[p["transition_type"].eq("touchback")]
    returns = p.loc[p["transition_type"].eq("return")]
    if tb.empty or returns.empty:
        raise ValueError("Pool needs both ordinary touchbacks and returns.")

    p_tb20 = float(tb["own_yardline"].eq(20).mean())
    ret_own = returns["own_yardline"].to_numpy(dtype=float)
    ret_clock = returns["elapsed_seconds"].to_numpy(dtype=float)

    output = [dict(state) for state in states]
    eligible = [
        i for i, s in enumerate(output)
        if (int(s.get("qtr", 0)) == 4
            and float(s.get("is_kickoff", 0)) > 0.5
            and abs(float(s.get("_score_change_original", 0))) > 0
            and float(s.get("game_seconds_remaining", 0)) > 0)
    ]
    size = len(eligible)
    if not size:
        return output

    is_tb = rng.random(size) < touchback_rate
    is_low_tb = rng.random(size) < p_tb20
    ret_indices = rng.integers(0, len(ret_own), size=size)

    for j, i in enumerate(eligible):
        s = output[i]
        if is_tb[j]:
            # A landing-zone touchback is at 20. A normal direct end-zone
            # touchback is at 35 in 2025+, versus 30 under 2024 rules.
            # 2026 exception: a kickoff taken from the 50 yields own 20.
            own = (20.0 if is_low_tb[j] or
                   (rules_year >= 2026 and s.get("_kickoff_from_50", False))
                   else (35.0 if rules_year >= 2025 else 30.0))
            elapsed = 0.0
            transition = "touchback"
        else:
            own = float(ret_own[ret_indices[j]])
            elapsed = float(ret_clock[ret_indices[j]])
            transition = "return"

        remaining = max(0.0, float(s["game_seconds_remaining"]) - elapsed)
        s.update({
            "is_kickoff": 0.0,
            "yardline_100": 100.0 - own,
            "down": 1.0,
            "ydstogo": min(10.0, 100.0 - own),
            "goal_to_go": float((100.0 - own) <= 10.0),
            "game_seconds_remaining": remaining,
            "half_seconds_remaining": max(
                0.0, float(s["half_seconds_remaining"]) - elapsed
            ),
            "final_five_minutes": float(remaining <= 300.0),
            "_kickoff_transition_type": transition,
        })

    return output
