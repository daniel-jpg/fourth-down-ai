"""Offline nflreadpy-shaped fourth-down intake for a FUTURE retrospective study.

The original project downloads play-by-play through nflreadpy.load_pbp().
This module consumes an already supplied in-memory DataFrame and never
fetches games. It accepts either the unchanged 2014-25 development archive
or clearly named synthetic_ 2027 fixtures, NEVER real 2026/2027 outcomes.

CRITICAL: Completed-game PBP provides reconstructed pre-play-looking fields,
not independently timestamped before-snap evidence. Actions and final-score
claims are projected separately from the decision-state projection.
No production forecasts, nuisance fitting or holdout scoring occur here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SOURCE = ROOT / "data/fourth_downs_raw.parquet"
HISTORICAL_GIT_BLOB = "41ae23de12913f0bf06ac608ad4716ae48d75df8"
FROZEN_POLICY = "31176d50af25db9dd42dae07e9b17cb5c2b930b4"
MODE_HISTORICAL = "historical_2014_2025_development"
MODE_SYNTHETIC = "fabricated_2027_only"

# All future source versions must be reviewed against this exact projection.
# No downstream action/result columns are allowed in preplay_features.
PREPLAY_SOURCE_FIELDS = (
    "season", "season_type", "week", "game_id", "play_id", "down",
    "qtr", "game_seconds_remaining", "yardline_100", "ydstogo",
    "goal_to_go", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "posteam", "home_team", "away_team", "location", "roof",
)
LABEL_SOURCE_FIELDS = (
    "play_type", "qb_scramble", "desc", "aborted_play", "safety",
    "result",
)
KEY = ("game_id", "play_id")
PREPLAY_OUTPUT_FIELDS = (
    "game_id", "play_id", "season", "week", "quarter", "down",
    "offense_team", "home_team", "away_team", "site",
    "site_advantage", "game_seconds_remaining", "yardline_100",
    "ydstogo", "goal_to_go", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "roof",
    "model_context_gaps", "model_context_complete",
    "feature_asof", "frozen_policy_commit",
)
EXCLUDED_FIELDS = ("game_id", "play_id", "reason")
OBSERVED_FIELDS = ("game_id", "play_id", "observed_action", "classification")
GAME_LABEL_FIELDS = ("game_id", "source_claimed_home_margin", "verification_status")
NORMAL_ACTIONS = {"NORMAL_GO_RUN", "NORMAL_GO_PASS", "PUNT", "FIELD_GOAL"}
ROOFS = {"outdoors", "open", "closed", "dome", "indoors"}
FAKE_PUNT = re.compile(r"punt formation|fake punt|lined up to punt.*fake", re.I)
FAKE_FG = re.compile(r"field.goal formation|fake field goal|fake fg", re.I)


def _number(value: Any, low: float, high: float) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None
    return x if math.isfinite(x) and low <= x <= high else None


def _integer(value: Any, low: int, high: int) -> int | None:
    x = _number(value, low, high)
    return int(x) if x is not None and x.is_integer() else None


def _identifier(value: Any) -> str | None:
    if not isinstance(value, bool):
        n = _integer(value, 1, 1_000_000_000)
        if n is not None:
            return str(n)
    if isinstance(value, str) and value and value == value.strip() and len(value) <= 140:
        if all(c not in value for c in ("\n", "\r", " ", "/")):
            return value
    return None


def _frame(raw: Any) -> pd.DataFrame:
    """Accept pandas or the Polars DataFrame returned by nflreadpy.load_pbp."""
    if isinstance(raw, pd.DataFrame):
        return raw
    if type(raw).__module__.startswith("polars.") and hasattr(raw, "to_pandas"):
        return raw.to_pandas()
    raise TypeError("Supply a pandas or nflreadpy Polars play-by-play DataFrame")


def _candidate_rows(raw: Any, mode: str, *, labels: bool) -> pd.DataFrame:
    if mode not in (MODE_HISTORICAL, MODE_SYNTHETIC):
        raise ValueError("Only frozen historical development or fabricated 2027 allowed")
    frame = _frame(raw)
    required = set(PREPLAY_SOURCE_FIELDS)
    if labels:
        required |= set(LABEL_SOURCE_FIELDS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing nflverse source columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("nflverse source frame must not be empty")
    # Validate ALL input rows before filtering to down 4. Never just drop an
    # unapproved future season present in a mixed multi-year download.
    if any(
        (_integer(y, 2014, 2025) is None if mode == MODE_HISTORICAL
         else _integer(y, 2027, 2027) is None)
        for y in frame["season"]
    ):
        raise ValueError("Source contains a forbidden season (2026/real 2027)")
    if mode == MODE_SYNTHETIC and any(
        not isinstance(g, str) or not g.startswith("synthetic_")
        for g in frame["game_id"]
    ):
        raise ValueError("All 2027 fixture game IDs must start synthetic_")
    projected = frame.loc[:, list(required)].copy()
    # The initial down field is the ONLY fourth-down census selector: no
    # post-play action, outcome, future down or play description can select it.
    selector = projected["down"].map(lambda v: _integer(v, 4, 4) == 4)
    source = projected.loc[selector].copy()
    if source.empty:
        raise ValueError("No source-reported fourth-down opportunities")
    source["game_id"] = source["game_id"].map(_identifier)
    source["play_id"] = source["play_id"].map(_identifier)
    if source[list(KEY)].isna().any().any():
        raise ValueError("Fourth-down source has missing/invalid game or play ID")
    if source.duplicated(list(KEY)).any():
        raise ValueError("Duplicate fourth-down game/play key")
    return source.sort_values(list(KEY), kind="stable").reset_index(drop=True)


def _preplay_state(row: Any) -> tuple[dict | None, str | None]:
    if row.season_type != "REG":
        return None, "NOT_REGULAR_SEASON"
    week = _integer(row.week, 1, 18)
    q = _integer(row.qtr, 1, 5)
    if week is None or q is None:
        return None, "INVALID_WEEK_OR_QUARTER"
    if q == 5:
        return None, "OVERTIME"
    offense, home, away = row.posteam, row.home_team, row.away_team
    if (not all(isinstance(x, str) and x.strip() for x in (offense, home, away))
            or home == away or offense not in (home, away)):
        return None, "MISSING_OR_INCONSISTENT_TEAMS"
    if row.location not in ("Home", "Neutral"):
        return None, "UNKNOWN_PHYSICAL_VENUE"
    lo, hi = {1: (2700, 3600), 2: (1800, 2700),
              3: (900, 1800), 4: (0, 900)}[q]
    sec = _number(row.game_seconds_remaining, lo, hi)
    pos = _number(row.yardline_100, .5, 99.5)
    togo = _number(row.ydstogo, 0, 100)
    diff = _number(row.score_differential, -150, 150)
    goal = _integer(row.goal_to_go, 0, 1)
    own_to = _integer(row.posteam_timeouts_remaining, 0, 3)
    opp_to = _integer(row.defteam_timeouts_remaining, 0, 3)
    if any(x is None for x in (sec, pos, togo, diff, goal, own_to, opp_to)):
        return None, "MISSING_OR_INVALID_PREPLAY_STATE"
    if togo > max(1., pos) + 1e-9:
        return None, "IMPOSSIBLE_YARDS_TO_GO"
    site = "NEUTRAL" if row.location == "Neutral" else (
        "HOME" if offense == home else "AWAY"
    )
    roof = row.roof if isinstance(row.roof, str) and row.roof in ROOFS else None
    gaps = []
    if roof is None:
        gaps.append("ROOF_UNAVAILABLE_OR_NOT_REVIEWED")
    # nflverse does not contain a certified first-half coin-toss receiver.
    # Q2 may not silently infer it from the future Q3 kickoff.
    if q == 2:
        gaps.append("Q2_SECOND_HALF_RECEIVER_UNAVAILABLE")
    return {
        "game_id": row.game_id, "play_id": row.play_id,
        "season": int(row.season), "week": week, "quarter": q, "down": 4,
        "offense_team": offense, "home_team": home, "away_team": away,
        "site": site, "site_advantage": {"HOME": 1, "AWAY": -1, "NEUTRAL": 0}[site],
        "game_seconds_remaining": sec, "yardline_100": pos,
        "ydstogo": togo, "goal_to_go": goal, "score_differential": diff,
        "posteam_timeouts_remaining": own_to,
        "defteam_timeouts_remaining": opp_to, "roof": roof,
        "model_context_gaps": ";".join(gaps),
        "model_context_complete": not gaps,
        "feature_asof": "POSTGAME_RECONSTRUCTED_NOT_WITNESSED",
        "frozen_policy_commit": FROZEN_POLICY,
    }, None


def reconstruct_source_preplay(raw: Any, *, mode: str) -> dict[str, Any]:
    """Phase 1: isolate PREPLAY-looking nflverse fields, never labels.

    The returned "model_context_complete" checks *field presence only*.
    No real-world as-of timing, provenance, or model forecast is certified.
    """
    candidates = _candidate_rows(raw, mode, labels=False)
    features, excluded = [], []
    for row in candidates.itertuples(index=False):
        state, reason = _preplay_state(row)
        if reason:
            excluded.append(dict(zip(EXCLUDED_FIELDS, (row.game_id, row.play_id, reason))))
        else:
            features.append(state)
    p = pd.DataFrame.from_records(features, columns=PREPLAY_OUTPUT_FIELDS)
    e = pd.DataFrame.from_records(excluded, columns=EXCLUDED_FIELDS)
    if len(p) + len(e) != len(candidates):
        raise RuntimeError("Missing source fourth down in preplay audit")
    reasons = dict(sorted(Counter(e["reason"]).items()))
    gap_counts = dict(sorted(Counter(
        reason for value in p["model_context_gaps"] if value
        for reason in value.split(";")
    ).items()))
    return {
        "mode": mode,
        "preplay_features": p,
        "excluded": e,
        "audit": {
            "source_fourth_down_candidates": len(candidates),
            "reconstructed_core_states": len(p),
            "excluded_states": len(e),
            "exclusion_counts": reasons,
            "missing_model_context_counts": gap_counts,
            "context_fields_present": int(p["model_context_complete"].sum()),
            "source_labels_accessed": False,
            "preplay_independently_timestamped": False,
            "real_2027_outcomes_accessed": False,
            "frozen_policy_commit": FROZEN_POLICY,
            "interpretation": (
                "Retrospective state reconstruction; NOT authenticated "
                "decision-time data or a model evaluation"
            ),
        },
    }


def _observed_call(row: Any) -> tuple[str, str]:
    """Phase 2: outcomes may label what happened, NEVER input to phase 1."""
    play = row.play_type
    if not isinstance(play, str):
        return "UNCLASSIFIED", "NO_PLAY_TYPE"
    desc = row.desc if isinstance(row.desc, str) else ""
    formation = bool(FAKE_PUNT.search(desc) or FAKE_FG.search(desc))
    if play == "no_play":
        return "NO_PLAY", "ADMINISTRATIVE_NO_PLAY"
    if play in ("run", "pass"):
        if formation:
            return "FAKE_OR_SPECIAL_TEAMS", "FAKE_EXCLUDED"
        if _integer(row.aborted_play, 0, 1) == 1:
            return "AMBIGUOUS_ABORTED", "MANUAL_REVIEW_REQUIRED"
        if _integer(row.safety, 0, 1) == 1:
            return "AMBIGUOUS_SAFETY", "MANUAL_REVIEW_REQUIRED"
        if play == "pass" or _integer(row.qb_scramble, 0, 1) == 1:
            return "NORMAL_GO_PASS", "NORMAL_CONFIDENT"
        return "NORMAL_GO_RUN", "NORMAL_CONFIDENT"
    if play in ("punt", "field_goal"):
        if "fake" in desc.lower():
            return "FAKE_OR_SPECIAL_TEAMS", "FAKE_EXCLUDED"
        if _integer(row.aborted_play, 0, 1) == 1:
            return "AMBIGUOUS_ABORTED", "MANUAL_REVIEW_REQUIRED"
        return ("PUNT" if play == "punt" else "FIELD_GOAL"), "NORMAL_CONFIDENT"
    return "UNCLASSIFIED", "UNSUPPORTED_PLAY_TYPE"


def project_postplay_labels(raw: Any, *, mode: str) -> dict[str, Any]:
    """Phase 2 EXPLICITLY SEPARATE: observational action and margin claims.

    Final margins here are unverified source CLAIMS, not completed-game
    crosschecks. Never send them to frozen model, policy, or nuisance X.
    """
    candidates = _candidate_rows(raw, mode, labels=True)
    actions = []
    for row in candidates.itertuples(index=False):
        call, classification = _observed_call(row)
        actions.append({
            "game_id": row.game_id, "play_id": row.play_id,
            "observed_action": call, "classification": classification,
        })
    games = []
    for game_id, rows in candidates.groupby("game_id", sort=True):
        margins = [
            _number(value, -150, 150) for value in rows["result"]
        ]
        found = {v for v in margins if v is not None}
        if len(found) > 1:
            raise ValueError(f"Conflicting source final margins for game {game_id}")
        margin = next(iter(found)) if found else None
        games.append({
            "game_id": game_id, "source_claimed_home_margin": margin,
            "verification_status": "UNVERIFIED_SOURCE_CLAIM",
        })
    observed = pd.DataFrame.from_records(actions, columns=OBSERVED_FIELDS)
    margins = pd.DataFrame.from_records(games, columns=GAME_LABEL_FIELDS)
    return {
        "observed_actions": observed,
        "source_game_result_claims": margins,
        "audit": {
            "source_fourth_down_candidates": len(candidates),
            "distinct_games": len(games),
            "classified_normal_actions": int(
                observed["observed_action"].isin(NORMAL_ACTIONS).sum()
            ),
            "normal_action_labels_verified_before_snap": False,
            "final_game_results_independently_verified": False,
            "real_2027_outcomes_accessed": False,
            "evaluation_or_model_refit_performed": False,
        },
    }


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def audit_pinned_historical_archive() -> dict[str, Any]:
    """Development replay ONLY. Verify file bytes BEFORE opening Parquet."""
    if (not HISTORICAL_SOURCE.is_file() or HISTORICAL_SOURCE.is_symlink()
            or HISTORICAL_SOURCE.resolve() != (ROOT / "data/fourth_downs_raw.parquet").resolve()):
        raise ValueError("Only original pinned 2014-25 Parquet archive allowed")
    blob = HISTORICAL_SOURCE.read_bytes()
    if _git_blob_sha1(blob) != HISTORICAL_GIT_BLOB:
        raise ValueError("Historical archive changed; refuse Parquet read")
    raw = pd.read_parquet(HISTORICAL_SOURCE, columns=list(PREPLAY_SOURCE_FIELDS))
    result = reconstruct_source_preplay(raw, mode=MODE_HISTORICAL)
    return {
        "source": "PINNED_2014_2025_NFLVERSE_RAW_PBP_ONLY",
        "source_blob_sha1": HISTORICAL_GIT_BLOB,
        "source_sha256": hashlib.sha256(blob).hexdigest(),
        **result["audit"],
    }


def main() -> None:
    """Local historical-only census; no NFL downloads or future scoring."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-audit-only", action="store_true")
    args = parser.parse_args()
    if not args.historical_audit_only:
        parser.error("Only --historical-audit-only is permitted")
    print(json.dumps(audit_pinned_historical_archive(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
