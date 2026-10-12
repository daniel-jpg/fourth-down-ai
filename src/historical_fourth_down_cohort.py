"""Historical 2014-25 REG fourth-down pre-snap cohort, research ONLY.

The checked-in labeled PBP has three very different classes of columns:
pre-snap features; the observed action; and final game/post-play labels.
Never feed the latter two into a predictive feature matrix.

This extractor uses explicit whitelists and independently materializes three
frames, keyed by (game_id, play_id). It is HISTORICAL DEV ONLY. It does not
retrieve or accept any 2026/2027 football data, and it never modifies the
2027 prospective synthetic-only cohort or the 2026 kickoff gates.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.fourth_down_policy_study import ACTIONS

START_SEASON, END_SEASON = 2014, 2025
KEY = ["game_id", "play_id"]
# Access to observed plays/terminal scores is permitted ONLY as labels.
SOURCE_COLUMNS = [
    "season", "season_type", "week", "game_id", "play_id",
    "down", "qtr", "game_seconds_remaining", "yardline_100",
    "ydstogo", "goal_to_go", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "posteam", "home_team", "away_team", "location",
    "action", "result",
]
# Feature columns intentionally contain NO action, result, final scores,
# desc, first_down, touchdown, converted, yards_gained, EPA, play_type or
# post-play data. This is the complete certified pre-snap covariate schema.
FEATURE_COLUMNS = (
    "quarter", "game_seconds_remaining", "yardline_100",
    "ydstogo", "goal_to_go", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "site_advantage",
)
PRESNAP_COLUMNS = [
    "game_id", "play_id", "season", "week",
    *FEATURE_COLUMNS, "site", "feature_asof",
]
ACTION_COLUMNS = [*KEY, "observed_action"]
OUTCOME_COLUMNS = [*KEY, "offense_win"]
NORMAL_ACTIONS = frozenset(ACTIONS)


def _finite(value: Any, *, lower: float, upper: float) -> float | None:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and lower <= number <= upper else None


def _integer(value: Any, *, lower: int, upper: int) -> int | None:
    number = _finite(value, lower=lower, upper=upper)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _id(value: Any) -> str | None:
    """Normalize nflverse play_id 243.0 to stable JSON string '243'."""
    number = _integer(value, lower=1, upper=1_000_000_000)
    if number is not None:
        return str(number)
    return _string(value)


def _make_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail if duplicate identified plays could overcount final win labels."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Historical labeled-PBP source must be nonempty")
    missing = set(SOURCE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required labeled-PBP fields: {sorted(missing)}")
    f = frame[SOURCE_COLUMNS].copy()
    years = pd.to_numeric(f["season"], errors="coerce")
    if years.isna().any() or not years.between(START_SEASON, END_SEASON).all():
        raise ValueError("Only historical 2014-2025 seasons are permitted")
    f["game_id"] = f["game_id"].map(_string)
    f["play_id"] = f["play_id"].map(_id)
    if f[KEY].isna().any().any():
        raise ValueError("Missing historical game or play ID; cannot audit joins")
    if f.duplicated(KEY).any():
        raise ValueError("Duplicate game/play key in fourth-down PBP")
    return f.sort_values(KEY, kind="stable").reset_index(drop=True)


def _game_margins(source: pd.DataFrame) -> dict[str, float | None]:
    """Use final result ONLY as outcome, require one consistent game margin."""
    grouped = {}
    for gid, g in source.groupby("game_id", sort=True):
        teams = g[["home_team", "away_team", "season", "week"]].drop_duplicates()
        if len(teams) != 1:
            raise ValueError(f"Conflicting historical game teams/season/week: {gid}")
        values = pd.to_numeric(g["result"], errors="coerce").to_numpy(float)
        seen = np.unique(values[np.isfinite(values)])
        if len(seen) > 1:
            raise ValueError(f"Conflicting final home margins for game {gid}")
        grouped[gid] = float(seen[0]) if len(seen) else None
    return grouped


def _state(row: Any) -> tuple[dict[str, Any] | None, str | None]:
    if row.season_type != "REG":
        return None, "not_regular_season"
    year = _integer(row.season, lower=START_SEASON, upper=END_SEASON)
    week = _integer(row.week, lower=1, upper=18)
    q = _integer(row.qtr, lower=1, upper=5)
    down = _integer(row.down, lower=1, upper=4)
    if year is None or week is None or q is None or down is None:
        return None, "invalid_football_state"
    if q == 5:
        return None, "overtime"
    if down != 4:
        return None, "not_fourth_down"
    if row.action == "NO_PLAY":
        return None, "administrative_no_play"
    if row.action not in NORMAL_ACTIONS:
        return None, "fake_or_unclassified_action"

    offense = _string(row.posteam)
    home = _string(row.home_team)
    away = _string(row.away_team)
    if home is None or away is None or offense not in (home, away) or home == away:
        return None, "missing_or_mismatched_team"

    venue = _string(row.location)
    if venue is None:
        return None, "missing_venue"
    site = "NEUTRAL" if venue == "Neutral" else (
        "HOME" if offense == home else "AWAY"
    )
    quarter_clock = {1: (2700, 3600), 2: (1800, 2700),
                     3: (900, 1800), 4: (0, 900)}
    lo, hi = quarter_clock[q]
    sec = _finite(row.game_seconds_remaining, lower=lo, upper=hi)
    pos = _finite(row.yardline_100, lower=.5, upper=99.5)
    ydstogo = _finite(row.ydstogo, lower=0, upper=100)
    margin = _finite(row.score_differential, lower=-150, upper=150)
    goal = _integer(row.goal_to_go, lower=0, upper=1)
    own_to = _integer(row.posteam_timeouts_remaining, lower=0, upper=3)
    opp_to = _integer(row.defteam_timeouts_remaining, lower=0, upper=3)
    if any(x is None for x in (
        sec, pos, ydstogo, margin, goal, own_to, opp_to
    )):
        return None, "missing_or_invalid_presnap_feature"
    if ydstogo > max(1.0, pos) + 1e-9:
        return None, "impossible_yards_to_go"
    return {
        "game_id": row.game_id, "play_id": row.play_id,
        "season": year, "week": week, "quarter": q,
        "game_seconds_remaining": sec, "yardline_100": pos,
        "ydstogo": ydstogo, "goal_to_go": goal,
        "score_differential": margin,
        "posteam_timeouts_remaining": own_to,
        "defteam_timeouts_remaining": opp_to,
        "site_advantage": 0 if site == "NEUTRAL" else (
            1 if site == "HOME" else -1
        ),
        "site": site, "feature_asof": "pre_snap",
    }, None


def extract_historical_cohort(raw: pd.DataFrame) -> dict[str, Any]:
    """Separate 2014-25 presnap, observed actions, and final-winner labels.

    This uses one historical PBP *source*, so outcomes are not truly
    independent externally; strict whitelists block label access in X.
    An independent source verification remains necessary before 2027.
    """
    source = _make_keys(raw)
    margins = _game_margins(source)
    states, actions, winners, excluded = [], [], [], []
    for row in source.itertuples(index=False):
        state, reason = _state(row)
        if reason is None:
            margin = margins[row.game_id]
            if margin is None:
                reason = "missing_final_result"
            elif margin == 0:
                reason = "tied_final_game"
        if reason:
            excluded.append({
                "game_id": row.game_id, "play_id": row.play_id,
                "reason": reason,
            })
            continue
        states.append(state)
        actions.append({
            "game_id": row.game_id, "play_id": row.play_id,
            "observed_action": row.action,
        })
        offense_home = state["site_advantage"] == 1 or (
            state["site_advantage"] == 0 and row.posteam == row.home_team
        )
        # Neutral venue still has a designated HOME team on the scoreboard.
        y = int(margin > 0) if offense_home else int(margin < 0)
        winners.append({
            "game_id": row.game_id, "play_id": row.play_id,
            "offense_win": y,
        })

    def frame(records, columns):
        return pd.DataFrame.from_records(records, columns=columns)

    p = frame(states, PRESNAP_COLUMNS)
    a = frame(actions, ACTION_COLUMNS)
    y = frame(winners, OUTCOME_COLUMNS)
    x_keys = set(map(tuple, p[KEY].itertuples(index=False, name=None)))
    for other in (a, y):
        other_keys = set(map(tuple, other[KEY].itertuples(index=False, name=None)))
        if other_keys != x_keys:
            raise RuntimeError("Pre-snap/action/outcome keys were misaligned")
    exclusions = dict(sorted(Counter(e["reason"] for e in excluded).items()))
    if len(p) + len(excluded) != len(source):
        raise RuntimeError("Historical candidate/exclusion accounting error")
    # A histogram of underrepresented actions is an AUDIT ONLY, never
    # a pre-snap feature or post-hoc eligibility override.
    return {
        "pre_snap": p, "observed_actions": a, "final_labels": y,
        "excluded": frame(excluded, ["game_id", "play_id", "reason"]),
        "audit": {
            "source_years": "2014-2025",
            "source_candidates": len(source),
            "included_decisions": len(p),
            "excluded_decisions": len(excluded),
            "exclusion_counts": exclusions,
            "observed_action_counts": dict(sorted(
                Counter(a["observed_action"]).items()
            )),
            "distinct_games": int(p["game_id"].nunique()),
            "source_definition": "historical labeled PBP, no 2026/2027 games",
            "feature_policy": "whitelisted_pre_snap_only",
            "external_final_score_verification": "NOT PERFORMED",
        },
    }


def join_training_labels(cohort: dict[str, Any]) -> pd.DataFrame:
    """Join labels for FITTING only. Do not pass returned object as X."""
    pre = cohort["pre_snap"]
    factual = cohort["observed_actions"]
    winner = cohort["final_labels"]
    result = pre.merge(factual, on=KEY, how="inner", validate="one_to_one")
    result = result.merge(winner, on=KEY, how="inner", validate="one_to_one")
    if len(result) != len(pre):
        raise ValueError("Historical training join dropped some decisions")
    return result.sort_values(KEY, kind="stable").reset_index(drop=True)


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """Reproducible row SHA-256 for the explicit whitelisted table."""
    if frame.empty:
        return hashlib.sha256(b"[]").hexdigest()
    ordered = frame.sort_values(KEY, kind="stable")
    rows = ordered.replace({np.nan: None}).to_dict(orient="records")
    value = json.dumps(
        rows, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()
