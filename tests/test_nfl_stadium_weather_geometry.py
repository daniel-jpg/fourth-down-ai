"""NFL venue weather geometry: no guessing attack-end / relocated stadium."""
from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from src import nfl_stadium_weather_geometry as venue
from scripts import capture_nfl_venue_awc_context as collector
from src.metar_decision_weather import validate_metar_receipt_for_display

OBS_UTC = "2026-10-12T18:00:00Z"
RECEIPT_UTC = "2026-10-12T18:04:00Z"
DECISION_UTC = "2026-10-12T18:35:00Z"
RETRIEVAL_UTC = "2026-10-12T18:45:00Z"


@pytest.fixture()
def artificial_kbfi_metar_bytes():
    """Fictitious station-level weather with real-looking geography."""
    from datetime import datetime, timezone
    samples = [{
        "icaoId": "KBFI",
        "obsTime": int(datetime(
            2026, 10, 12, 18, 0, tzinfo=timezone.utc,
        ).timestamp()),
        "receiptTime": RECEIPT_UTC,
        "lat": 47.53,
        "lon": -122.302,
        "wdir": 90,
        "wspd": 12,
        "rawOb": (
            "METAR KBFI 121800Z 09012KT 10SM -RA "
            "BKN015 12/08 A2996 RMK AO2"
        ),
    }]
    return json.dumps(samples, sort_keys=True).encode("utf-8")


def test_full_2026_team_list_not_fake_official_field_azimuth():
    data = venue.load_stadium_reference(season=2026)
    assert data["schema_version"] == "2026_unofficial_field_axis_lookup_v1"
    assert set(data["venues"]) == venue.EXPECTED_TEAMS
    assert len(data["venues"]) == 32
    assert len([x for x in data["venues"].values() if x["lat"] is not None]) == 31
    assert data["source"]["blob_sha1"] == (
        "12b15041d31b181fe7a2ef06551e046042141158"
    )
    assert "unofficial" in data["schema_version"]
    assert "roof_type" not in data["venues"]["ATL"]
    assert data["venues"]["ATL"]["physical_game_roof_known"] is False
    assert data["venues"]["LV"]["venue_name"] == "Allegiant Stadium"
    assert "OAK" not in data["venues"]


def test_shared_stadiums_and_actual_2026_tennessee_site():
    rows = venue.load_stadium_reference()["venues"]
    for a, b in (("NYG", "NYJ"), ("LAC", "LAR")):
        assert rows[a]["lat"] == rows[b]["lat"]
        assert rows[a]["lon"] == rows[b]["lon"]
        assert rows[a]["venue_id"] == rows[b]["venue_id"]
    assert "Nissan Stadium" in rows["TEN"]["venue_name"]
    assert "2027" in rows["TEN"]["note"]
    assert rows["BUF"]["lat"] is None
    assert "new 2026 venue not yet mapped" in venue.stadium_menu()["BUF"]


@pytest.mark.parametrize("team", ["SEA", "GB", "LV", "NYJ", "LAC", "TEN"])
def test_both_explicit_attack_directions_differ_by_exactly_180(team):
    a = venue.resolve_stadium_geometry(
        team=team, season=2026, offense_attack_end=venue.ATTACK_ENDS[0],
    )
    b = venue.resolve_stadium_geometry(
        team=team, season=2026, offense_attack_end=venue.ATTACK_ENDS[1],
    )
    assert math.isclose(
        (a["offense_attack_heading_true_deg_approx"] + 180) % 360,
        b["offense_attack_heading_true_deg_approx"],
        abs_tol=1e-9,
    )
    assert a["field_lat"] == b["field_lat"]
    assert a["field_lon"] == b["field_lon"]
    assert a["status"] == "APPROXIMATE_2026_VENUE_GEOMETRY_CONTEXT_ONLY"
    for field in (
        "offense_attack_heading_user_confirmed_at_snap",
        "stadium_coordinates_independently_verified",
        "field_axis_orientation_independently_surveyed",
        "actual_stadium_roof_open_verified", "actual_game_venue_confirmed",
        "wind_from_airport_is_on_field_sensor",
        "future_2027_venue_approved", "real_2027_outcomes_accessed",
        "can_change_official_fourth_down_recommendation",
    ):
        assert a[field] is False


@pytest.mark.parametrize("year", [2014, 2025, 2027, 2028, "2026", True, None])
def test_expired_or_wrong_year_denied_before_file_is_accessed(
    monkeypatch, year,
):
    class Trap:
        def __getattribute__(self, item):
            raise AssertionError("Accessed unapproved football venue file")
    monkeypatch.setattr(venue, "VENUES_FILE", Trap())
    with pytest.raises(PermissionError, match="restricted to 2026"):
        venue.resolve_stadium_geometry(
            team="SEA", season=year,
            offense_attack_end=venue.ATTACK_ENDS[0],
        )
    with pytest.raises(PermissionError, match="restricted to 2026"):
        venue.stadium_menu(season=year)


@pytest.mark.parametrize("team,reason", [
    ("BUF", "Buffalo moved"),
    ("OAK", "explicitly listed"),
    ("LA", "explicitly listed"),
    ("", "explicitly listed"),
    ("random", "explicitly listed"),
])
def test_stale_buffalo_and_unlisted_stadiums_never_guess(team, reason):
    with pytest.raises(ValueError, match=reason):
        venue.resolve_stadium_geometry(
            team=team, season=2026,
            offense_attack_end=venue.ATTACK_ENDS[0],
        )


@pytest.mark.parametrize("end", [
    None, "2026", "A", "B", "HOME", "", 0, 180, True,
])
def test_attacking_end_cannot_be_inferred_from_game_metadata(end):
    with pytest.raises(ValueError, match="explicitly selected"):
        venue.resolve_stadium_geometry(
            team="SEA", season=2026, offense_attack_end=end,
        )


def test_modified_source_cannot_imply_known_roof_or_retired_buffalo_coords(
    monkeypatch, tmp_path,
):
    original = venue.load_stadium_reference()
    changed = deepcopy(original)
    changed["venues"]["ATL"]["physical_game_roof_known"] = True
    file = tmp_path / "unsupported-stadium.json"
    file.write_text(json.dumps(changed))
    monkeypatch.setattr(venue, "VENUES_FILE", file)
    with pytest.raises(ValueError, match="roof"):
        venue.load_stadium_reference()
    changed["venues"]["ATL"]["physical_game_roof_known"] = False
    changed["venues"]["BUF"]["lat"] = 42.7
    file.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Buffalo"):
        venue.load_stadium_reference()


def test_fake_seattle_venue_proximity_and_direction_against_official_metar_format(
    artificial_kbfi_metar_bytes,
):
    outcomes = []
    for end in venue.ATTACK_ENDS:
        result = collector.capture_venue_weather(
            venue_team="SEA", season=2026, attack_end=end,
            station="KBFI", decision_utc=DECISION_UTC,
            roof="outdoors", source_bytes=artificial_kbfi_metar_bytes,
            captured_utc=RETRIEVAL_UTC, fetched_from_awc_https=False,
        )
        outcomes.append(result)
        assert result["station_distance_km"] < 10
        assert result["venue_geometry"]["team_venue_reference"] == "SEA"
        assert result["venue_geometry"]["field_axis_orientation_independently_surveyed"] is False
        assert result["source_downloaded_after_decision"] is True
        assert result["airport_present_weather_code"] == "AIRPORT_RAIN_REPORTED"
        assert result["official_model_unchanged"] is True
        assert result["actual_game_at_this_venue_verified"] is False
        assert result["no_live_NFL_game_or_2027_outcomes_accessed"] is True
        assert result["snap_specific_weather_verified"] is False
        assert result["may_change_official_go_fg_or_punt_model"] is False
        parsed = validate_metar_receipt_for_display(result)
        assert parsed["unsigned_manual_import_not_source_authenticated"] is True
    assert math.isclose(
        outcomes[0]["headwind_mph_signed"],
        -outcomes[1]["headwind_mph_signed"],
        abs_tol=.001,
    )
    assert math.isclose(
        outcomes[0]["crosswind_from_right_mph_signed"],
        -outcomes[1]["crosswind_from_right_mph_signed"],
        abs_tol=.001,
    )


def test_late_or_distant_station_and_invalid_2027_venue_refused(
    artificial_kbfi_metar_bytes,
):
    with pytest.raises(PermissionError, match="restricted to 2026"):
        collector.capture_venue_weather(
            venue_team="SEA", season=2027,
            attack_end=venue.ATTACK_ENDS[0],
            station="KBFI", decision_utc=DECISION_UTC,
            roof="outdoors", source_bytes=artificial_kbfi_metar_bytes,
            captured_utc=RETRIEVAL_UTC, fetched_from_awc_https=False,
        )
    with pytest.raises(ValueError, match="Buffalo moved"):
        collector.capture_venue_weather(
            venue_team="BUF", season=2026,
            attack_end=venue.ATTACK_ENDS[0],
            station="KBFI", decision_utc=DECISION_UTC,
            roof="outdoors", source_bytes=artificial_kbfi_metar_bytes,
            captured_utc=RETRIEVAL_UTC, fetched_from_awc_https=False,
        )
    with pytest.raises(ValueError, match="too far"):
        collector.capture_venue_weather(
            venue_team="GB", season=2026,
            attack_end=venue.ATTACK_ENDS[0],
            station="KBFI", decision_utc=DECISION_UTC,
            roof="outdoors", source_bytes=artificial_kbfi_metar_bytes,
            captured_utc=RETRIEVAL_UTC, fetched_from_awc_https=False,
        )


def test_no_weather_promotion_or_download_side_effect_in_geometry_source():
    from inspect import getsource
    source = getsource(venue)
    for unsupported in (
        "load_pbp(", "requests.get(", "urlopen(", "joblib.dump(",
        "nflreadpy", "websocket", "2027-outcomes",
    ):
        assert unsupported not in source
    assert "field_axis_orientation_independently_surveyed" in source
    assert "actual_stadium_roof_open_verified" in source
