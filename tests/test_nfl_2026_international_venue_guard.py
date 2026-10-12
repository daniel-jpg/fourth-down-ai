"""2026 scheduled overseas fixtures are NOT US team home venues.

Only official 2026 SCHEDULE, no real game results, 2025 reserved holdouts,
2026 protected feature outcomes, live weather or coach action downloads.
"""
from __future__ import annotations

from copy import deepcopy
import json
import pytest

from src import nfl_2026_international_venue_guard as international
from scripts import capture_nfl_venue_awc_context as collector


def test_official_nine_international_games_eight_distinct_venues():
    data = international.load_international_schedule()
    games = data["events"]
    assert len(games) == 9
    assert len({x["external_venue_id"] for x in games}) == 8
    assert {x["nfl_schedule_date_et"] for x in games} == {
        "2026-09-10", "2026-09-27", "2026-10-04",
        "2026-10-11", "2026-10-18", "2026-10-25",
        "2026-11-08", "2026-11-15", "2026-11-22",
    }
    assert len({x["venue_name"] for x in games}) == 8
    assert "2026" in data["schema"]
    assert "operations.nfl.com" in data["source"]
    assert all(g["stadium_coordinates"] is None for g in games)
    assert all(g["playing_field_axis"] is None for g in games)
    assert all(g["actual_game_played_verified"] is False for g in games)
    assert all(g["roster_or_outcome_loaded"] is False for g in games)


@pytest.mark.parametrize("date,a,b,venue", [
    ("2026-09-10", "SF", "LAR", "Melbourne Cricket Ground"),
    ("2026-09-27", "BAL", "DAL", "Maracanã Stadium"),
    ("2026-10-04", "IND", "WAS", "Tottenham Hotspur Stadium"),
    ("2026-10-11", "PHI", "JAX", "Tottenham Hotspur Stadium"),
    ("2026-10-18", "HOU", "JAX", "Wembley Stadium"),
    ("2026-10-25", "PIT", "NO", "Stade de France"),
    ("2026-11-08", "CIN", "ATL", "Bernabéu Stadium"),
    ("2026-11-15", "NE", "DET", "FC Bayern Munich Arena"),
    ("2026-11-22", "MIN", "SF", "Estadio Banorte"),
])
def test_every_published_fixture_refuses_regular_team_home_geometry(
    date, a, b, venue,
):
    f = international.get_international_fixture(
        game_date_et=date, team_a=a, team_b=b,
    )
    assert f["venue_name"] == venue
    assert international.get_international_fixture(
        game_date_et=date, team_a=b, team_b=a,
    ) == f
    for home, away in ((a, b), (b, a)):
        with pytest.raises(ValueError, match="INTERNATIONAL_VENUE_CONFLICT") as err:
            international.guard_team_home_venue_for_game(
                game_date_et=date, venue_team=home, opponent_team=away,
            )
        assert venue in str(err.value)
        assert "domestic" in str(err.value)


def test_nonmatch_is_explicitly_unknown_not_claimed_home_venue():
    result = international.guard_team_home_venue_for_game(
        game_date_et="2026-10-12", venue_team="SEA", opponent_team="GB",
    )
    assert result["status"] == (
        "NOT_ON_KNOWN_INTERNATIONAL_LIST_ACTUAL_SITE_UNVERIFIED"
    )
    assert result["domestic_home_venue_confirmed"] is False
    assert result["other_neutral_or_relocated_venue_excluded"] is False
    assert result["official_model_unchanged"] is True
    assert result["protected_2026_game_outcomes_accessed"] is False
    # The JAX second London date cannot be conflated with the first.
    assert international.get_international_fixture(
        game_date_et="2026-10-11", team_a="HOU", team_b="JAX",
    ) is None


@pytest.mark.parametrize("date", [
    "2027-01-01", "2025-10-12", "bad-date", "2026-02-30", None,
    "2026/10/12", 20261012, True,
])
def test_invalid_dates_fail_before_football_or_weather_lookup(date):
    with pytest.raises((ValueError, PermissionError)):
        international.get_international_fixture(
            game_date_et=date, team_a="SF", team_b="LAR",
        )


@pytest.mark.parametrize("teams", [
    ("SEA", "SEA"), ("LAR", "OAK"), ("?", "SF"),
    (None, "SF"), ("SF", 123), ("", ""),
])
def test_wrong_or_same_teams_rejected(teams):
    with pytest.raises(ValueError, match="team"):
        international.get_international_fixture(
            game_date_et="2026-09-10", team_a=teams[0], team_b=teams[1],
        )


def test_bad_or_early_source_data_fails_closed(tmp_path, monkeypatch):
    original = international.load_international_schedule()
    target = tmp_path / "official-schedule-copy.json"
    monkeypatch.setattr(international, "FILE", target)
    changed = deepcopy(original)
    changed["events"][0]["stadium_coordinates"] = [35.0, -87.0]
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Unverified"):
        international.load_international_schedule()
    changed = deepcopy(original)
    changed["events"][0]["actual_game_played_verified"] = True
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Unverified"):
        international.load_international_schedule()
    changed = deepcopy(original)
    changed["events"].pop()
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="incomplete"):
        international.load_international_schedule()


def test_noa_a_venue_wrapper_cannot_call_weather_for_international_match(
    monkeypatch,
):
    calls = []
    def no_noaa(station):
        calls.append(station)
        raise AssertionError("NOAA traffic must be rejected before I/O")
    monkeypatch.setattr(collector, "fetch_recent_awc_metar", no_noaa)
    # Pure wrapper: identical impossible game, international physical venue.
    with pytest.raises(ValueError, match="INTERNATIONAL_VENUE_CONFLICT"):
        collector.capture_venue_weather(
            venue_team="LAR", season=2026,
            attack_end="TOWARD_AXIS_BEARING_A",
            station="KSEA", decision_utc="2026-09-10T20:00:00Z",
            roof="outdoors", source_bytes=b"[]",
            captured_utc="2026-09-10T20:01:00Z",
            fetched_from_awc_https=True,
            game_date_et="2026-09-10", opponent_team="SF",
        )
    assert calls == []


def test_live_weather_requires_dated_matchup_even_for_nonoverseas_games():
    with pytest.raises(ValueError, match="requires a dated matchup"):
        collector.capture_venue_weather(
            venue_team="SEA", season=2026,
            attack_end="TOWARD_AXIS_BEARING_A",
            station="KSEA", decision_utc="2026-10-12T20:00:00Z",
            roof="outdoors", source_bytes=b"[]",
            captured_utc="2026-10-12T20:01:00Z",
            fetched_from_awc_https=True,
        )
    with pytest.raises(ValueError, match="supplied together"):
        collector.capture_venue_weather(
            venue_team="SEA", season=2026,
            attack_end="TOWARD_AXIS_BEARING_A",
            station="KSEA", decision_utc="2026-10-12T20:00:00Z",
            roof="outdoors", source_bytes=b"[]",
            captured_utc="2026-10-12T20:01:00Z",
            fetched_from_awc_https=False,
            game_date_et="2026-10-12",
        )


def test_future_season_schedule_abstains_without_opening_files(monkeypatch):
    class Trap:
        def __getattribute__(self, field):
            raise AssertionError("Future football schedule must not be read")
    monkeypatch.setattr(international, "FILE", Trap())
    with pytest.raises(PermissionError, match="Only the pinned 2026"):
        international.load_international_schedule(season=2027)
