"""2026 NFL scheduled INTERNATIONAL venue exceptions; no game outcomes.

A team name is not a physical stadium. For the nine NFL-official
international fixtures, any attempt to derive stadium coordinates or
offense wind from that team's regular domestic home venue must FAIL.

For any OTHER fixture, a nonmatch does NOT prove home-site validity
(neutral/alternate/relocated domestic games still possible).
No future-season assumption, no data download or real game scoring.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

from .nfl_stadium_weather_geometry import EXPECTED_TEAMS

FILE = Path(__file__).resolve().parent / "2026_nfl_international_venue_schedule.json"
OFFICIAL_SOURCE = (
    "https://operations.nfl.com/programs-initiatives/"
    "international-growth/nfl-international-games"
)


def load_international_schedule(*, season: Any = 2026) -> dict:
    if type(season) is not int or season != 2026:
        raise PermissionError("Only the pinned 2026 international venue schedule is supported")
    if not FILE.is_file() or FILE.is_symlink():
        raise ValueError("Missing pinned official 2026 game-to-venue exception list")
    data = json.loads(FILE.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema") !=
        "2026_NFL_OFFICIAL_SCHEDULED_INTERNATIONAL_VENUES_V1"
        or data.get("season") != 2026
        or data.get("source") != OFFICIAL_SOURCE
        or not isinstance(data.get("events"), list)
        or len(data["events"]) != 9
    ):
        raise ValueError("2026 international fixture source is invalid or incomplete")
    distinct_dates_pairs = set()
    venue_ids = set()
    for event in data["events"]:
        if not isinstance(event, dict):
            raise ValueError("Malformed 2026 international fixture")
        dt = _parse_date(event.get("nfl_schedule_date_et"))
        if not isinstance(event.get("participants"), list):
            raise ValueError("Missing international fixture team identities")
        teams = event["participants"]
        if (len(teams) != 2 or type(teams[0]) is not str
                or type(teams[1]) is not str
                or len(set(teams)) != 2
                or any(team not in EXPECTED_TEAMS for team in teams)):
            raise ValueError("Ambiguous international fixture team identities")
        key = (str(dt), tuple(sorted(teams)))
        if key in distinct_dates_pairs:
            raise ValueError("Duplicate international fixture")
        distinct_dates_pairs.add(key)
        if (not isinstance(event.get("venue_name"), str)
                or not event["venue_name"].strip()
                or not isinstance(event.get("external_venue_id"), str)
                or not event["external_venue_id"].strip()):
            raise ValueError("Missing published international game venue")
        if (
            event.get("stadium_coordinates") is not None
            or event.get("playing_field_axis") is not None
            or event.get("actual_game_played_verified") is not False
            or event.get("opposite_direction_at_snap_verified") is not False
            or event.get("roster_or_outcome_loaded") is not False
        ):
            raise ValueError("Unverified future weather geometry or NFL outcome claim")
        venue_ids.add(event["external_venue_id"])
    if len(venue_ids) != 8:
        raise ValueError("Expected NFL's nine 2026 games in eight international venues")
    return data


def _parse_date(value: Any) -> date:
    if type(value) is not str or len(value) != 10:
        raise ValueError("Expected NFL published YYYY-MM-DD schedule date (ET)")
    try:
        dt = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid official schedule date") from exc
    if dt.year != 2026 or dt.isoformat() != value:
        raise PermissionError("Cannot look up games outside the 2026 schedule")
    return dt


def get_international_fixture(
    *, game_date_et: Any, team_a: Any, team_b: Any,
) -> dict | None:
    """Exact DATE + UNORDERED MATCHUP. No site inference on a nonmatch."""
    dt = _parse_date(game_date_et)
    if (type(team_a) is not str or type(team_b) is not str
            or team_a not in EXPECTED_TEAMS or team_b not in EXPECTED_TEAMS
            or team_a == team_b):
        raise ValueError("Explicit distinct 2026 NFL team identities are required")
    for event in load_international_schedule()["events"]:
        if (event["nfl_schedule_date_et"] == dt.isoformat()
                and set(event["participants"]) == {team_a, team_b}):
            return dict(event)
    return None


def guard_team_home_venue_for_game(
    *, game_date_et: Any, venue_team: Any, opponent_team: Any,
) -> dict:
    """Refuse home-venue substitution for ALL 9 known overseas games.

    The lookup has no outcome labels and no international stadium bearings.
    Nonmatching dates/opponents remain explicitly UNVERIFIED (not OK).
    """
    match = get_international_fixture(
        game_date_et=game_date_et,
        team_a=venue_team,
        team_b=opponent_team,
    )
    if match is not None:
        raise ValueError(
            "2026_SCHEDULED_INTERNATIONAL_VENUE_CONFLICT: "
            f"{game_date_et} {venue_team}-{opponent_team} takes place at "
            f"{match['venue_name']}, {match['city_country']}; "
            "the domestic team home-stadium coordinates/field bearing "
            "must NOT be used. International geometry not verified."
        )
    return {
        "status": "NOT_ON_KNOWN_INTERNATIONAL_LIST_ACTUAL_SITE_UNVERIFIED",
        "game_date_et": game_date_et,
        "venue_team": venue_team,
        "opponent_team": opponent_team,
        "domestic_home_venue_confirmed": False,
        "other_neutral_or_relocated_venue_excluded": False,
        "field_attack_direction_verified": False,
        "official_model_unchanged": True,
        "protected_2026_game_outcomes_accessed": False,
    }
