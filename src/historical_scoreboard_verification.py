"""Independent ESPN final-score cross-check for 2014–2025 PBP labels.

This module is pure: no network requests, model loads/fits or 2026+ data.
ESPN observations are separate POST-GAME outcomes, never decision-time features.

Important: a successful historical SCORE check does NOT certify roof,
second-half receiving team, pre-snap timestamps, fake-play intent, or a
counterfactual coaching advantage. Those require separate review.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

FIRST, LAST = 2014, 2025
GAME_COLUMNS = (
    "season", "season_type", "week", "game_id",
    "home_team", "away_team", "result",
)
# ESPN may use current franchise names even when NFLverse still carries old
# historical city abbreviations. Only explicit same-franchise aliases allowed.
ALIASES = {
    "JAC": "JAX", "JAX": "JAX",
    "STL": "LAR", "LA": "LAR", "LAR": "LAR",
    "SD": "LAC", "SDG": "LAC", "LAC": "LAC",
    "OAK": "LV", "LV": "LV",
    "WAS": "WSH", "WSH": "WSH",
    "ARZ": "ARI", "AZ": "ARI", "ARI": "ARI",
    "BLT": "BAL", "BAL": "BAL",
    "HST": "HOU", "HOU": "HOU",
    "GNB": "GB", "GB": "GB",
    "KAN": "KC", "KC": "KC",
    "NWE": "NE", "NE": "NE",
    "NOR": "NO", "NO": "NO",
    "SFO": "SF", "SF": "SF",
    "TAM": "TB", "TB": "TB",
}
STANDARD = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WSH",
})
SOURCE_KEY = ("season", "week", "home_team", "away_team")
RECORD_KEYS = (
    "season", "week", "game_id", "home", "away",
    "home_score", "away_score", "margin", "event_id",
    "neutral_site",
)


def season_int(value: Any) -> int:
    if type(value) is not int or not FIRST <= value <= LAST:
        raise ValueError("Only archived 2014–2025 NFL seasons allowed")
    return value


def week_int(value: Any, season: int) -> int:
    high = 17 if season <= 2020 else 18
    if type(value) is not int or not 1 <= value <= high:
        raise ValueError(f"Week must be 1–{high} for season {season}")
    return value


def team_code(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Missing team abbreviation")
    abbr = value.strip().upper()
    canonical = ALIASES.get(abbr, abbr)
    if canonical not in STANDARD:
        raise ValueError(f"Unrecognized NFL franchise abbreviation: {abbr!r}")
    return canonical


def _integer(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, str):
        if not value.isdigit():
            raise ValueError(f"{label} must be a nonnegative integer")
        n = int(value)
    elif isinstance(value, (int, np.integer)):
        n = int(value)
    elif isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        n = int(value)
    else:
        raise ValueError(f"{label} must be a nonnegative integer")
    if not low <= n <= high:
        raise ValueError(f"{label} outside [{low}, {high}]")
    return n


def _key(year: int, week: int, home: str, away: str) -> tuple:
    key = (year, week, team_code(home), team_code(away))
    if key[2] == key[3]:
        raise ValueError("Home and away franchise cannot be the same")
    return key


def parse_espn_week(payload: Mapping[str, Any], season: int, week: int) -> list[dict]:
    """Strictly extract final REG scores, never intermediate boxscore values."""
    season_int(season)
    week_int(week, season)
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("ESPN scoreboard response has no event list")
    events = payload["events"]
    if not 1 <= len(events) <= 16:
        raise ValueError("ESPN historical scoreboard week is incomplete or oversized")
    result, seen = [], set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Malformed ESPN event")
        meta = event.get("season", {})
        em = event.get("week", {})
        if (not isinstance(meta, dict) or
                _integer(meta.get("year"), "ESPN season", FIRST, LAST) != season or
                _integer(meta.get("type"), "season type", 1, 3) != 2 or
                not isinstance(em, dict) or
                _integer(em.get("number"), "ESPN week", 1, 18) != week):
            raise ValueError("Wrong ESPN year, week or non-REG game")
        comp = event.get("competitions")
        if not isinstance(comp, list) or len(comp) != 1 or not isinstance(comp[0], dict):
            raise ValueError("ESPN event requires exactly one competition")
        competition = comp[0]
        status = event.get("status") or competition.get("status")
        if not isinstance(status, dict) or not isinstance(status.get("type"), dict):
            raise ValueError("Missing final game-status provenance")
        if status["type"].get("completed") is not True:
            raise ValueError("Historical ESPN game is not final")
        team_rows = competition.get("competitors")
        if not isinstance(team_rows, list) or len(team_rows) != 2:
            raise ValueError("ESPN competition must have two competitors")
        sides = {}
        for team in team_rows:
            if not isinstance(team, dict) or not isinstance(team.get("team"), dict):
                raise ValueError("Malformed ESPN competitor")
            side = team.get("homeAway")
            if side not in {"home", "away"} or side in sides:
                raise ValueError("Exactly one ESPN home and one away required")
            sides[side] = {
                "team": team_code(team["team"].get("abbreviation")),
                "score": _integer(team.get("score"), "ESPN final score", 0, 150),
            }
        if set(sides) != {"home", "away"}:
            raise ValueError("Missing ESPN home or away side")
        key = _key(season, week, sides["home"]["team"], sides["away"]["team"])
        if key in seen:
            raise ValueError("Duplicate ESPN matchup in same week")
        seen.add(key)
        event_id = str(event.get("id", "")).strip()
        if not event_id.isdigit():
            raise ValueError("Missing ESPN unique event id")
        home_score = sides["home"]["score"]
        away_score = sides["away"]["score"]
        neutral = competition.get("neutralSite")
        if neutral is not None and type(neutral) is not bool:
            raise ValueError("ESPN neutralSite should be boolean when supplied")
        result.append(dict(zip(RECORD_KEYS, (
            season, week, str(event_id),
            sides["home"]["team"], sides["away"]["team"],
            home_score, away_score, home_score-away_score, event_id, neutral
        ))))
    return sorted(result, key=lambda r: (r["season"], r["week"], r["home"], r["away"]))


def pbp_game_census(frame: pd.DataFrame) -> pd.DataFrame:
    """Game-level source census. No play outcomes besides final margin."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Need historical labeled fourth-down source")
    missing = set(GAME_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing PBP fields: {sorted(missing)}")
    f = frame[list(GAME_COLUMNS)].copy()
    seasons = pd.to_numeric(f["season"], errors="coerce")
    if seasons.isna().any() or not seasons.between(FIRST, LAST).all():
        raise ValueError("2026+ or unapproved source seasons are forbidden")
    if f["game_id"].isna().any() or f["game_id"].duplicated().all():
        raise ValueError("Missing or corrupt PBP game ids")
    f = f[f["season_type"].eq("REG")].copy()
    if f.empty:
        raise ValueError("No regular-season game data")
    records = []
    for game_id, game in f.groupby("game_id", sort=True):
        year_values = game["season"].drop_duplicates().tolist()
        weeks = game["week"].drop_duplicates().tolist()
        home = game["home_team"].drop_duplicates().tolist()
        away = game["away_team"].drop_duplicates().tolist()
        if not all(len(x)==1 for x in (year_values, weeks, home, away)):
            raise ValueError(f"Inconsistent PBP game teams/week: {game_id}")
        year = season_int(int(year_values[0]))
        week = week_int(int(weeks[0]), year)
        key = _key(year, week, home[0], away[0])
        margin = pd.to_numeric(game["result"], errors="coerce")
        unique = margin[margin.notna() & np.isfinite(margin)].unique()
        if len(unique) > 1:
            raise ValueError(f"Conflicting PBP final margin for game {game_id}")
        if len(unique) == 1 and not float(unique[0]).is_integer():
            raise ValueError("PBP final point margin is nonintegral")
        records.append({
            "game_id": str(game_id), "season": year, "week": week,
            "home": key[2], "away": key[3],
            "pbp_home_margin": int(unique[0]) if len(unique) else None,
            "pbp_fourth_down_rows": int(len(game)),
        })
    result = pd.DataFrame(records).sort_values("game_id").reset_index(drop=True)
    if result.duplicated(["season","week","home","away"]).any():
        raise ValueError("Duplicate normalized PBP game matchup key")
    return result


def verify_independent_margins(
    games: pd.DataFrame, external: Sequence[Mapping[str, Any]],
    *, expected_seasons: Sequence[int],
) -> dict[str, Any]:
    """Audit every PBP game in selected historical seasons; reject partial matches.

    Matching on (season, week, normalized home/away) rather than unrelated
    PBP vs ESPN numeric game IDs ensures robust, auditable correspondence.
    Every game must match and have an equal final home point margin.
    """
    expected = sorted({season_int(x) for x in expected_seasons})
    if not expected:
        raise ValueError("Explicit selected historical seasons required")
    if not isinstance(games, pd.DataFrame) or not set({
        "game_id", "season", "week", "home", "away",
        "pbp_home_margin", "pbp_fourth_down_rows",
    }).issubset(games):
        raise ValueError("Invalid PBP game census")
    considered = games[games["season"].isin(expected)].copy()
    if considered.empty or set(considered["season"]) != set(expected):
        raise ValueError("Missing selected PBP historical season")
    ref = {}
    for item in external:
        if not isinstance(item, dict) or set(item) != set(RECORD_KEYS):
            raise ValueError("Malformed independent ESPN final-game record")
        year = season_int(item["season"])
        week = week_int(item["week"], year)
        key = _key(year, week, item["home"], item["away"])
        if key in ref:
            raise ValueError("Repeated independent ESPN game key")
        score_h = _integer(item["home_score"], "home score", 0, 150)
        score_a = _integer(item["away_score"], "away score", 0, 150)
        if item["margin"] != score_h-score_a:
            raise ValueError("Internally inconsistent ESPN final margin")
        ref[key] = item
    issues, matched, unknown_source = [], [], []
    for row in considered.itertuples(index=False):
        key = _key(row.season, row.week, row.home, row.away)
        observed = ref.get(key)
        if observed is None:
            issues.append({
                "game_id": row.game_id, "reason": "NO_INDEPENDENT_GAME_MATCH",
                "year": row.season, "week": row.week,
                "home": row.home, "away": row.away,
            })
        elif row.pbp_home_margin is None or pd.isna(row.pbp_home_margin):
            unknown_source.append({
                "game_id": row.game_id, "reason": "PBP_FINAL_MARGIN_MISSING",
                "espn_event_id": observed["event_id"],
            })
        elif int(row.pbp_home_margin) != observed["margin"]:
            issues.append({
                "game_id": row.game_id, "reason": "FINAL_MARGIN_MISMATCH",
                "pbp_margin": int(row.pbp_home_margin),
                "espn_margin": observed["margin"],
                "espn_event_id": observed["event_id"],
            })
        else:
            matched.append({
                "game_id": row.game_id,
                "season": row.season,
                "week": row.week,
                "espn_event_id": observed["event_id"],
                "home_margin": int(row.pbp_home_margin),
                "n_fourth_down_rows": int(row.pbp_fourth_down_rows),
            })
    all_game_count = len(considered)
    compared = len(matched)+sum(x["reason"] == "FINAL_MARGIN_MISMATCH" for x in issues)
    status = (
        "ALL_MATCHED"
        if len(matched) == all_game_count and not issues and not unknown_source
        else "BLOCKED_INCOMPLETE_OR_MISMATCHED"
    )
    checksum = hashlib.sha256(json.dumps(
        sorted(matched, key=lambda m:m["game_id"]),
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "status": status,
        "selected_seasons": expected,
        "pbp_games": all_game_count,
        "matched_final_scores": len(matched),
        "compared_final_scores": compared,
        "total_fourth_down_rows_in_verified_games": sum(
            m["n_fourth_down_rows"] for m in matched
        ),
        "missing_pbp_scores": unknown_source,
        "issues": issues,
        "reason_counts": dict(sorted(Counter(i["reason"] for i in issues).items())),
        "matched_games_sha256": checksum,
        "independent_score_source": "ESPN historical final scoreboards (not nflverse)",
        "not_verified": [
            "pre-snap timestamps", "roof as of decision",
            "second-half kickoff receiver", "fake formation intent",
            "causal action value", "future 2026/2027 game results",
        ],
    }
