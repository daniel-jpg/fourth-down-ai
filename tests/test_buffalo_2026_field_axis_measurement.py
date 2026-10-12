"""No real NFL play outcomes. Fictional GPS reference points only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from math import isclose

import pytest

from src import buffalo_2026_field_axis_measurement as measure
from scripts import measure_buffalo_field_axis as cli


SOUTH_LAT = 42.772635
SOUTH_LON = -78.792271
NORTH_LAT = 42.773445
NORTH_LON = -78.792109
BYTES_NORTH = b"FICTIONAL_DATA_NOT_AN_ACTUAL_SURVEY_NORTH_01"
BYTES_SOUTH = b"FICTIONAL_DATA_NOT_AN_ACTUAL_SURVEY_SOUTH_02"


def point(side, lat, lon, error=2., kind="goal line"):
    return {
        "latitude": lat,
        "longitude": lon,
        "horizontal_uncertainty_m": error,
        "source_id": side.lower()+"gps01",
        "observer_id": "observer"+side.lower()+"01",
        "measurement_method": "CLAIMED_FIELD_GPS",
        "source_url": "https://example.org/fictional-"+side.lower(),
        "collected_at_utc": "2026-10-12T18:00:00Z",
        "landmark_description": f"{side.lower()} physical {kind} midpoint observed in fabricated fixture",
    }


@pytest.fixture()
def parts():
    return {
        "NORTH": point("NORTH", NORTH_LAT, NORTH_LON),
        "SOUTH": point("SOUTH", SOUTH_LAT, SOUTH_LON),
    }


@pytest.fixture()
def evidence():
    return {
        "northgps01": BYTES_NORTH,
        "southgps01": BYTES_SOUTH,
    }


def make(parts, evidence, *, landmark="GOAL_LINE_MIDPOINTS_100YDS",
         season=2026, team="BUF"):
    return measure.build_buffalo_field_axis_candidate(
        season=season, venue_team=team,
        landmark_type=landmark,
        point_metadata=parts, original_artifacts=evidence,
    )


def test_cannot_turn_new_highmark_stadium_center_into_fake_true_azimuth(
    parts, evidence,
):
    actual = make(parts, evidence)
    assert actual["status"] == measure.CANDIDATE_ONLY
    assert actual["season"] == 2026
    assert actual["venue"] == "NEW_HIGHMARK_STADIUM_BUFFALO"
    assert actual["expected_field_separation_m"] == 91.44
    assert 90 <= actual["wgs84_geometry"]["distance_m"] <= 92
    angle = actual["candidate_northward_azimuth_deg"]
    assert 7 < angle < 10
    assert actual["candidate_southward_azimuth_deg"] == pytest.approx(
        angle+180, abs=.00001,
    )
    assert 0 < actual["candidate_worst_case_angle_error_deg_if_errors_true"] < 4
    assert actual["center_offset_from_2026_osm_stadium_approx_m"] < 1
    assert actual["source_points"]["NORTH"]["original_local_artifact_sha256"] == (
        hashlib.sha256(BYTES_NORTH).hexdigest()
    )
    assert BYTES_NORTH.decode() not in json.dumps(actual)
    assert actual["source_contents_in_receipt"] is False
    for field in (
        "independently_surveyed_exact_field_bearing",
        "coordinate_provenance_independently_authenticated",
        "independent_source_coordinates_crosschecked",
        "two_witnesses_independence_verified",
        "goal_line_landmarks_verified_on_field",
        "attack_end_of_specific_nfl_play_verified",
        "snap_level_weather_source_verified",
        "may_use_candidate_as_trusted_METAR_direction",
        "may_change_official_go_fg_punt_model",
        "protected_2026_nfl_outcomes_examined",
        "real_2027_outcomes_examined",
    ):
        assert actual[field] is False, field


def test_wgs84_real_geographic_angle_not_confusing_lat_lon_distances():
    straight = measure.wgs84_field_vector_m(
        42.772635, -78.79219, 42.773445, -78.79219,
    )
    assert straight["true_azimuth_south_to_north_deg"] == 0
    assert straight["true_azimuth_north_to_south_deg"] == 180
    assert straight["east_m_from_south_to_north"] == 0
    assert 89 < straight["distance_m"] < 91
    diagonal = measure.wgs84_field_vector_m(
        SOUTH_LAT, SOUTH_LON, NORTH_LAT, NORTH_LON,
    )
    assert diagonal["east_m_from_south_to_north"] > 12
    assert diagonal["north_m_from_south_to_north"] > 89


def test_back_end_line_separation_is_120_yards_not_goal_line_100(
    parts, evidence,
):
    changed = deepcopy(parts)
    for side, lat, lon in (
        ("SOUTH", 42.772546, -78.792201),
        ("NORTH", 42.773534, -78.792179),
    ):
        changed[side]["latitude"] = lat
        changed[side]["longitude"] = lon
        changed[side]["landmark_description"] = (
            f"{side.lower()} back end line midpoint in fabricated GPS fixture"
        )
    calibrated = make(
        changed, evidence, landmark="BACK_END_LINE_MIDPOINTS_120YDS",
    )
    assert calibrated["expected_field_separation_m"] == 109.728
    assert 108 < calibrated["wgs84_geometry"]["distance_m"] < 111
    with pytest.raises(ValueError, match="landmark description"):
        make(parts, evidence, landmark="BACK_END_LINE_MIDPOINTS_120YDS")


@pytest.mark.parametrize("badpoint,partname,field", [
    ((42.7750, -78.7921), "NORTH", "separation"),  # roof/bowl not goal line
    ((42.7731, -78.786954), "NORTH", "field geometry"),  # old stadium
])
def test_invalid_field_rectangle_or_old_stadium_fails_without_heading(
    parts, evidence, badpoint, partname, field,
):
    changed = deepcopy(parts)
    changed[partname]["latitude"] = badpoint[0]
    changed[partname]["longitude"] = badpoint[1]
    with pytest.raises(ValueError, match="separation|vicinity|aligned|outside admissible range"):
        make(changed, evidence)


def test_reject_coordinates_near_old_stadium_not_the_2026_new_stadium(
    parts, evidence,
):
    old_stadium = deepcopy(parts)
    for point in old_stadium.values():
        point["longitude"] += .00524
    with pytest.raises(ValueError, match="vicinity|outside admissible range"):
        make(old_stadium, evidence)


def test_east_west_field_alignment_conflicts_with_documentary_axis(
    parts, evidence,
):
    bad = deepcopy(parts)
    bad["SOUTH"]["latitude"] = 42.7730
    bad["NORTH"]["latitude"] = 42.77315
    bad["SOUTH"]["longitude"] = -78.79275
    bad["NORTH"]["longitude"] = -78.79165
    with pytest.raises(ValueError, match="NORTH/SOUTH"):
        make(bad, evidence)


def test_swapped_end_zones_wrong_direction_and_nonfinite_rejected(
    parts, evidence,
):
    changed = deepcopy(parts)
    changed["NORTH"]["latitude"], changed["SOUTH"]["latitude"] = (
        changed["SOUTH"]["latitude"], changed["NORTH"]["latitude"]
    )
    with pytest.raises(ValueError, match="physically north"):
        make(changed, evidence)
    for val in (float("nan"), float("inf"), True, 92):
        changed = deepcopy(parts)
        changed["NORTH"]["latitude"] = val
        with pytest.raises(ValueError):
            make(changed, evidence)


def test_measurement_uncertainty_cannot_be_fake_zero_or_uninformative(
    parts, evidence,
):
    for bad in (-1., 0., .01, 999., float("nan"), True):
        changed = deepcopy(parts)
        changed["NORTH"]["horizontal_uncertainty_m"] = bad
        with pytest.raises(ValueError, match="uncertainty"):
            make(changed, evidence)
    changed = deepcopy(parts)
    changed["NORTH"]["horizontal_uncertainty_m"] = 12.
    changed["SOUTH"]["horizontal_uncertainty_m"] = 12.
    with pytest.raises(ValueError, match="too uncertain"):
        make(changed, evidence)


def test_source_and_observer_are_not_auto_certified_even_if_distinct(
    parts, evidence,
):
    data = make(parts, evidence)
    assert not data["coordinate_provenance_independently_authenticated"]
    assert all(
        rec["source_authenticity_verified"] is False
        and rec["independent_field_survey_certified"] is False
        for rec in data["source_points"].values()
    )
    for key in ("observer_id", "source_url"):
        changed = deepcopy(parts)
        changed["NORTH"][key] = changed["SOUTH"][key]
        with pytest.raises(ValueError, match="Different"):
            make(changed, evidence)
    with pytest.raises(ValueError, match="Different"):
        make(parts, {"northgps01": BYTES_NORTH, "southgps01": BYTES_NORTH})


@pytest.mark.parametrize("year", [2025, 2027, "2026", True, None])
def test_denies_other_seasons_before_opening_stadium_dataset(
    parts, evidence, year, monkeypatch,
):
    def trap(*args, **kwargs):
        raise AssertionError("Never read venue info for future year")
    monkeypatch.setattr(measure, "resolve_stadium_location", trap)
    with pytest.raises(PermissionError, match="2026"):
        make(parts, evidence, season=year)


def test_reject_non_buffalo_non_geo_and_missing_artifacts(parts, evidence):
    with pytest.raises(ValueError, match="Buffalo"):
        make(parts, evidence, team="SEA")
    with pytest.raises(ValueError, match="landmark"):
        make(parts, evidence, landmark="STADIUM_POLYGON")
    with pytest.raises(ValueError, match="original artifacts"):
        make(parts, {"northgps01": BYTES_NORTH})


def test_geography_claim_time_and_provenance_must_be_bounded(parts, evidence):
    for field, value in (
        ("collected_at_utc", "2025-10-12T18:00:00Z"),
        ("collected_at_utc", "2026-10-12T18:00:00"),
        ("measurement_method", "APPROVED_OFFICIAL_GNSS"),
        ("source_url", "http://unverified.test/"),
        ("landmark_description", "Stadium center, not a surveyed point"),
        ("latitude", 0),
        ("longitude", -70),
    ):
        changed = deepcopy(parts)
        changed["NORTH"][field] = value
        with pytest.raises(ValueError):
            make(changed, evidence)


def test_local_cli_honors_original_artifacts_and_never_leaks_payload(
    parts, evidence, tmp_path,
):
    folder = tmp_path/"evidence"
    folder.mkdir()
    p = {
        side: point | {
            "artifact_file": f"{side.lower()}.gps.txt",
        }
        for side, point in parts.items()
    }
    manifest = folder/"points.json"
    manifest.write_text(json.dumps(p))
    for side in ("NORTH", "SOUTH"):
        (folder/f"{side.lower()}.gps.txt").write_bytes(
            evidence[p[side]["source_id"]]
        )
    data = cli.build_local_candidate(
        manifest_path=manifest, landmark_type="GOAL_LINE_MIDPOINTS_100YDS",
    )
    assert data["source_contents_in_receipt"] is False
    output = tmp_path/"candidate.json"
    cli.write_weather_receipt(output, data)
    assert output.is_file()
    assert BYTES_NORTH not in output.read_bytes()
    assert BYTES_SOUTH not in output.read_bytes()
    assert json.loads(output.read_text())["candidate_northward_azimuth_deg"] > 0
    with pytest.raises(FileExistsError):
        cli.write_weather_receipt(output, data)


def test_local_cli_refuses_absolute_path_symlink_and_directory_traversal(
    parts, evidence, tmp_path,
):
    folder = tmp_path/"evidence"
    folder.mkdir()
    (folder/"north.gps.txt").write_bytes(BYTES_NORTH)
    (folder/"south.gps.txt").write_bytes(BYTES_SOUTH)
    rows = {
        side: point | {"artifact_file": f"{side.lower()}.gps.txt"}
        for side, point in parts.items()
    }
    manifest = folder/"points.json"
    manifest.write_text(json.dumps(rows))
    for bad in ("../private/secret.json", "/tmp/personal_file", "."):
        altered = deepcopy(rows)
        altered["NORTH"]["artifact_file"] = bad
        manifest.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="safe relative|inaccessible"):
            cli.build_local_candidate(
                manifest_path=manifest,
                landmark_type="GOAL_LINE_MIDPOINTS_100YDS",
            )
    altered = deepcopy(rows)
    (folder/"north.gps.txt").unlink()
    (folder/"north.gps.txt").symlink_to(folder/"south.gps.txt")
    manifest.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="inaccessible"):
        cli.build_local_candidate(
            manifest_path=manifest,
            landmark_type="GOAL_LINE_MIDPOINTS_100YDS",
        )


def test_never_fetches_live_data_2026_results_or_promotes_official_model():
    from inspect import getsource
    for obj in (measure, cli):
        code = getsource(obj)
        for prohibited in (
            "nflreadpy", "load_pbp(", "urlopen(", "requests.get(",
            "predict_wp(", "simulate_go(", "joblib.dump(",
        ):
            assert prohibited not in code
