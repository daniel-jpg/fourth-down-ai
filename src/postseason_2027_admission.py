"""Future NFLverse 2027 REG season admission contract: FABRICATED-ONLY rehearsal.

Read-only, offline verifier of a COMPLETE 272-game source season against a
separately supplied scoreboard-shaped 272-game final roster. Neither source
is downloaded here. Synthetic ESPN-shaped rows are NOT independently
authenticated real ESPN records. Current code UNCONDITIONALLY refuses genuine
2027/2026 data, even if the caller claims the season is finished.

After the 2027 season an independently reviewed, separately versioned
real-data loader and release gate will be required. Never modify the 2026
once-only experiments to bypass this rule.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from src.fourth_down_policy_study import POLICY_COMMIT
from src.historical_scoreboard_verification import STANDARD

MODE = "fabricated_2027_only"
YEAR = 2027
WEEKS = frozenset(range(1, 19))
EXPECTED_GAMES = 272
EXPECTED_TEAMS = 32
GAMES_PER_TEAM = 17
FINAL_WEEK_GAMES = 16
SOURCE_FIELDS = frozenset({
    "game_id", "season", "week", "season_type", "home_team",
    "away_team", "pbp_home_margin",
})
EXTERNAL_FIELDS = frozenset({
    "event_id", "season", "week", "game_type", "status",
    "home_team", "away_team", "home_score_final", "away_score_final",
    "provider",
})
FAKE_PROVIDER = "FABRICATED_ESPN_SCOREBOARD_FIXTURE"
ID_RE = re.compile(r"synthetic_[a-zA-Z0-9_]{1,120}\Z")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def refuse_future_data(mode: Any) -> None:
    """Never inspect caller data before selecting the synthetic-only mode."""
    if mode != MODE:
        raise PermissionError(
            "Genuine 2027 holdout admission is disabled. A completed regular "
            "season, externally reviewed scoreboard, original-source version "
            "hashes, separate precommitted release review and once-only "
            "authorization are required. A caller-supplied final flag "
            "cannot unlock this fabricated-only code."
        )


def _exact(record: Any, keys: frozenset[str], label: str) -> None:
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError(f"{label} must have exact approved fields")


def _whole(value: Any, low: int, high: int, field: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low},{high}]")
    return value


def _teams(record: Mapping[str, Any]) -> tuple[str, str]:
    home, away = record["home_team"], record["away_team"]
    if (not isinstance(home, str) or not isinstance(away, str)
            or home not in STANDARD or away not in STANDARD or home == away):
        raise ValueError("Unknown or duplicate NFL team code")
    return home, away


def _score_source(rows: Sequence[Mapping[str, Any]]) -> dict:
    if not isinstance(rows, list) or len(rows) != EXPECTED_GAMES:
        raise ValueError("PBP roster must contain exactly 272 complete REG games")
    registered = {}
    ids = set()
    for row in rows:
        _exact(row, SOURCE_FIELDS, "nflverse source game")
        if row["season_type"] != "REG":
            raise ValueError("Only REG game census allowed")
        _whole(row["season"], YEAR, YEAR, "season")
        week = _whole(row["week"], 1, 18, "week")
        home, away = _teams(row)
        key = (week, home, away)
        game_id = row["game_id"]
        if (not isinstance(game_id, str) or not ID_RE.fullmatch(game_id)
                or game_id in ids or key in registered):
            raise ValueError("Duplicate/unauthorized PBP game ID or matchup")
        ids.add(game_id)
        margin = _whole(row["pbp_home_margin"], -150, 150, "PBP final margin")
        registered[key] = (game_id, margin)
    return registered


def _score_reference(rows: Sequence[Mapping[str, Any]]) -> dict:
    if not isinstance(rows, list) or len(rows) != EXPECTED_GAMES:
        raise ValueError("Independent scoreboard roster must contain exactly 272 finals")
    by_matchup = {}
    events = set()
    for row in rows:
        _exact(row, EXTERNAL_FIELDS, "independent shaped scoreboard")
        _whole(row["season"], YEAR, YEAR, "season")
        week = _whole(row["week"], 1, 18, "week")
        home, away = _teams(row)
        if row["game_type"] != "REG" or row["status"] != "final":
            raise ValueError("Independently sourced game is not final REG")
        if row["provider"] != FAKE_PROVIDER:
            raise ValueError("Only intentionally fabricated ESPN-shaped records allowed")
        event_id = row["event_id"]
        if (not isinstance(event_id, str) or not event_id.startswith("synthetic_")
                or event_id in events):
            raise ValueError("Duplicate or nonfabricated event ID")
        events.add(event_id)
        key = (week, home, away)
        if key in by_matchup:
            raise ValueError("Repeated independent scoreboard matchup")
        home_points = _whole(row["home_score_final"], 0, 150, "home final score")
        away_points = _whole(row["away_score_final"], 0, 150, "away final score")
        by_matchup[key] = (home_points, away_points)
    return by_matchup


def _complete_roster(keys: Sequence[tuple[int, str, str]]) -> dict:
    weeks = Counter()
    games_per_team = Counter()
    team_week = set()
    for week, home, away in keys:
        weeks[week] += 1
        for team in (home, away):
            pair = (week, team)
            if pair in team_week:
                raise ValueError("NFL team plays twice in a single week")
            team_week.add(pair)
            games_per_team[team] += 1
    if (set(weeks) != WEEKS or set(games_per_team) != set(STANDARD)
            or len(games_per_team) != EXPECTED_TEAMS
            or weeks[18] != FINAL_WEEK_GAMES
            or any(c < 1 or c > 16 for c in weeks.values())
            or any(c != GAMES_PER_TEAM for c in games_per_team.values())):
        raise ValueError(
            "2027 season incomplete: need all weeks 1-18, 272 games, "
            "32 teams with 17 games each and 16 final-week games"
        )
    return dict(sorted(weeks.items()))


def verify_fabricated_complete_season(
    source_games: Any, reference_games: Any, *, mode: Any,
) -> dict[str, Any]:
    """Structural independence check. NEVER certification of a real provider."""
    refuse_future_data(mode)
    pbp = _score_source(source_games)
    ref = _score_reference(reference_games)
    if set(pbp) != set(ref):
        raise ValueError("Independent final roster does not match complete PBP schedule")
    weekly = _complete_roster(sorted(pbp))
    final_games = []
    for key in sorted(pbp):
        game_id, source_margin = pbp[key]
        home_points, away_points = ref[key]
        if source_margin != home_points - away_points:
            raise ValueError(f"Independent final-score mismatch for {key!r}")
        week, home, away = key
        final_games.append({
            "game_id": game_id, "season": YEAR, "week": week,
            "home_team": home, "away_team": away,
            "home_score_final": home_points,
            "away_score_final": away_points,
            "status": "final", "game_type": "REG",
        })
    return {
        "status": "FABRICATED_FULL_SEASON_SCORE_CROSSCHECK_PASSED",
        "final_games": final_games,
        "audit": {
            "season": YEAR,
            "season_completeness": "272_OF_272_STRUCTURALLY_CHECKED",
            "games_per_team": GAMES_PER_TEAM,
            "source_games": len(pbp),
            "source_weeks": weekly,
            "score_matches": len(final_games),
            "nflverse_source_sha256": _canonical_hash(source_games),
            "independent_shaped_score_sha256": _canonical_hash(reference_games),
            "normalized_finals_sha256": _canonical_hash(final_games),
            "frozen_policy_commit": POLICY_COMMIT,
            "authentic_espn_response_verified": False,
            "external_source_independence_certified": False,
            "2027_season_currently_complete": False,
            "real_2027_data_accessed": False,
            "future_evaluation_authorized": False,
            "interpretation": (
                "All 272 fake source scores match an invented, separately "
                "constructed ESPN-shaped final roster. NOT independently "
                "authenticated real NFL source data."
            ),
        },
    }
