"""Source-hashed Buffalo 2026 snap-attack-end witness research tests.

ONLY fictitious game/clock observations. Never retrieve 2026 protected
football actions/outcomes or any real NFL result.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from src import nfl_stadium_weather_geometry as venue
from src import nfl_2026_snap_direction_evidence as direction
from scripts import capture_buffalo_snap_direction_claim as cli

ARGS = {
    "season": 2026,
    "venue_team": "BUF",
    "opponent_team": "NE",
    "offense_team": "BUF",
    "game_date_et": "2026-10-12",
    "play_key": "fictitiousplay00459",
    "quarter": 4,
    "game_clock_mmss": "04:59",
    "decision_at_utc": "2026-10-12T20:30:00Z",
}
A = b"FICTIONAL_SOURCE_VIDEO_NORTH_ENDZONE_HASH_ONLY_1"
B = b"FICTIONAL_DIFFERENT_ONSITE_WITNESS_SOURCE_2"


def claim(source_id="fakevideo001", witness_id="observerone",
          end="NORTH", observed="2026-10-12T20:29:30Z",
          available="2026-10-12T20:31:00Z", source_url=None):
    return {
        "source_id": source_id,
        "witness_id": witness_id,
        "source_type": "BROADCAST_FRAME",
        "artifact_observed_at_utc": observed,
        "artifact_available_at_utc": available,
        "attack_end_claim": end,
        "location_landmark_observed": (
            "human says offense was facing north end zone from north upper deck"
        ),
        "source_url": source_url or f"https://example.org/{source_id}",
    }


def adjudicate(rows=None, artifacts=None, **overrides):
    rows = rows if rows is not None else [claim()]
    artifacts = artifacts if artifacts is not None else {
        rows[0]["source_id"]: A,
    }
    return direction.assess_2026_buffalo_snap_direction(
        **(ARGS | overrides),
        witness_metadata=rows,
        original_artifacts=artifacts,
    )


def test_confirmed_north_south_stadium_axis_but_no_false_numeric_azimuth():
    data = venue.load_stadium_reference(season=2026)
    src = data["venues"]["BUF"]["qualitative_field_axis"]
    assert src["orientation"] == "NORTH_SOUTH_REPORTED_NOT_SURVEYED"
    assert src["end_zone_names"] == ["north", "south"]
    assert src["exact_azimuth_degrees"] is None
    assert src["nominal_true_north_axis_azimuth_degrees"] is None
    assert src["maximum_true_north_deviation_degrees_known"] is False
    assert src["attack_end_for_any_given_snap_known"] is False
    assert src["numerical_wind_projection_authorized"] is False
    assert len(src["documentary_sources"]) >= 2
    sources = " ".join(x["url"] for x in src["documentary_sources"])
    assert "populous.com/highmarkstadium" in sources
    assert "washingtonpost.com/sports/nfl/2026/08/07" in sources
    location = venue.resolve_stadium_location(team="BUF", season=2026)
    assert location["physical_endzone_labels"] == ["north", "south"]
    assert location["qualitative_field_orientation"] == (
        "REPORTED_APPROX_NORTH_SOUTH_NOT_SURVEYED"
    )
    assert location["field_axis_degrees_approx"] is None
    assert location["surveyed_numeric_headwind_projection_approved"] is False
    with pytest.raises(ValueError, match="axis"):
        venue.resolve_stadium_geometry(
            team="BUF", season=2026,
            offense_attack_end=venue.ATTACK_ENDS[0],
        )


def test_single_late_attestation_is_retrospective_and_unverified():
    out = adjudicate()
    assert out["status"] == direction.RECEIPT_STATUS
    assert out["claimed_attack_end"] == "NORTH"
    assert out["source_count"] == 1
    assert out["claim_consistency"] == "ONE_UNCORROBORATED_USER_ASSERTION"
    assert out["retrospective_source_present"] is True
    assert out["all_sources_asserted_available_predecision"] is False
    assert out["numeric_true_north_attack_azimuth_degrees"] is None
    assert out["numeric_headwind_crosswind_permitted"] is False
    assert out["direction_of_attack_independently_certified"] is False
    assert out["trusted_live_forecast_provenance"] is False
    assert out["may_change_frozen_go_fg_punt_probabilities"] is False
    assert out["protected_2026_fourth_down_outcomes_evaluated"] is False
    assert out["real_2027_outcomes_accessed"] is False
    assert out["source_bytes_in_receipt"] is False
    assert A.decode() not in json.dumps(out)
    assert out["witnesses"][0]["original_local_artifact_sha256"] == hashlib.sha256(A).hexdigest()
    assert out["known_international_venue_guard"]["domestic_home_venue_confirmed"] is False
    assert out["source_citations_user_claimed_not_fetched"] is True


def test_two_distinct_agreeing_sources_still_are_not_authenticated():
    rows = [
        claim(available="2026-10-12T20:29:52Z"),
        claim("onsite002", "observertwo", observed="2026-10-12T20:29:40Z",
              available="2026-10-12T20:29:55Z"),
    ]
    out = adjudicate(rows, {rows[0]["source_id"]: A, rows[1]["source_id"]: B})
    assert out["claim_consistency"] == "TWO_USER_ASSERTED_DISTINCT_WITNESSES"
    assert out["all_sources_asserted_available_predecision"] is True
    assert out["retrospective_source_present"] is False
    assert out["direction_of_attack_independently_certified"] is False
    assert out["trusted_live_forecast_provenance"] is False
    assert out["numeric_headwind_crosswind_permitted"] is False
    assert out["source_count"] == 2
    assert len({w["original_local_artifact_sha256"] for w in out["witnesses"]}) == 2


def test_conflicting_north_south_claims_fail_closed():
    rows = [
        claim(),
        claim("onsite002", "observertwo", end="SOUTH",
              observed="2026-10-12T20:29:40Z"),
    ]
    with pytest.raises(ValueError, match="Conflicting"):
        adjudicate(rows, {rows[0]["source_id"]: A, rows[1]["source_id"]: B})


@pytest.mark.parametrize("field,value,fragment", [
    ("witness_id", "observerone", "Duplicate"),
    ("source_id", "fakevideo001", "Duplicate"),
    ("source_url", "https://example.org/fakevideo001", "Duplicate"),
])
def test_independent_witness_must_use_distinct_identity_and_media(field, value, fragment):
    rows = [claim(), claim("onsite002", "observertwo")]
    rows[1][field] = value
    artifacts = {"fakevideo001": A, "onsite002": B}
    if field == "source_id":
        artifacts = {"fakevideo001": A}
    with pytest.raises(ValueError, match="Duplicate|exactly"):
        adjudicate(rows, artifacts)


def test_same_source_bytes_from_two_lookalike_files_is_not_corroboration():
    rows = [claim(), claim("onsite002", "observertwo")]
    with pytest.raises(ValueError, match="Duplicate"):
        adjudicate(rows, {"fakevideo001": A, "onsite002": A})


@pytest.mark.parametrize("update,pattern", [
    ({"season": 2027}, "2026 only"),
    ({"season": 2025}, "2026 only"),
    ({"season": "2026"}, "2026 only"),
    ({"venue_team": "SEA"}, "Buffalo"),
    ({"offense_team": "KC"}, "Offense"),
    ({"opponent_team": "BUF"}, "Offense"),
    ({"quarter": 5}, "regulation quarter"),
    ({"quarter": True}, "regulation quarter"),
    ({"game_clock_mmss": "16:00"}, "Game clock"),
    ({"game_clock_mmss": "4:70"}, "Game clock"),
    ({"decision_at_utc": "2026-10-12T20:30:00"}, "timezone"),
    ({"game_date_et": "2026-10-01"}, "schedule day"),
    ({"play_key": "post snap WIN"}, "Play ID"),
])
def test_fail_closed_bad_play_metadata(update, pattern):
    with pytest.raises((PermissionError, ValueError), match=pattern):
        adjudicate(**update)


@pytest.mark.parametrize("change,match", [
    ({"artifact_observed_at_utc": "2026-10-12T20:31:00Z"}, "after/too far"),
    ({"artifact_observed_at_utc": "2026-10-12T20:20:00Z"}, "after/too far"),
    ({"artifact_available_at_utc": "2026-10-12T20:28:00Z"}, "before underlying"),
    ({"attack_end_claim": "EAST"}, "NORTH or SOUTH"),
    ({"source_type": "FROM_POSTPLAY_SCORE"}, "source type"),
    ({"location_landmark_observed": "random"}, "geographic field landmark"),
    ({"source_url": "http://untrusted.example/"}, "HTTPS"),
    ({"source_url": "https://x@malicious.example/"}, "HTTPS"),
    ({"play_type": "FIELD_GOAL"}, "Post-play outcomes"),
])
def test_invalid_evidence_never_enters_wind_direction_record(change, match):
    record = claim()
    record.update(change)
    with pytest.raises((PermissionError, ValueError), match=match):
        adjudicate([record], {"fakevideo001": A})


def test_empty_or_incompatible_local_media_is_rejected():
    with pytest.raises(ValueError, match="bytes are required"):
        adjudicate([claim()], {"fakevideo001": b"x"})
    with pytest.raises(ValueError, match="match source IDs"):
        adjudicate([claim()], {"wrong_id": A})
    with pytest.raises(ValueError, match="One or two"):
        adjudicate([], {})
    with pytest.raises(ValueError, match="One or two"):
        adjudicate([claim()] * 3, {})


def test_does_not_ever_fetch_nfl_seasons_or_issue_live_weather():
    from inspect import getsource
    txt = getsource(direction)
    for prohibited in (
        "load_pbp(", "nflreadpy", "urlopen(", "requests.get(",
        "field_goal_probabilities(", "simulate_go(", "predict_wp(",
        "import joblib", "load_model(", "2026_regular_season_scores",
    ):
        assert prohibited not in txt


def test_cli_reads_evidence_files_hashes_without_copying_content(tmp_path):
    original_dir = tmp_path / "evidence"
    original_dir.mkdir()
    (original_dir / "north.txt").write_bytes(A)
    (original_dir / "second.txt").write_bytes(B)
    records = [
        claim(available="2026-10-12T20:29:52Z")
        | {"artifact_file": "north.txt"},
        claim(
            "onsite002", "observertwo", available="2026-10-12T20:29:56Z"
        ) | {"artifact_file": "second.txt"},
    ]
    manifest = original_dir / "sightings.json"
    manifest.write_text(json.dumps(records))
    data = cli.assemble_direction_record(
        **ARGS, witness_manifest=manifest,
    )
    assert data["source_count"] == 2
    assert data["all_sources_asserted_available_predecision"]
    assert data["claimed_attack_end"] == "NORTH"
    assert "artifact_file" not in json.dumps(data)
    assert str(original_dir) not in json.dumps(data)
    out = tmp_path / "receipt.json"
    cli.write_weather_receipt(out, data)
    assert out.is_file()
    assert A not in out.read_bytes()
    assert B not in out.read_bytes()
    with pytest.raises(FileExistsError):
        cli.write_weather_receipt(out, data)


def test_cli_denies_unsafe_relative_file_and_symlinks(tmp_path):
    from pathlib import Path
    folder = tmp_path / "evidence"
    folder.mkdir()
    (folder / "real.txt").write_bytes(A)
    (folder / "symlink.txt").symlink_to(folder / "real.txt")
    metadata = claim() | {"artifact_file": "../secrets.txt"}
    manifest = folder / "sightings.json"
    manifest.write_text(json.dumps([metadata]))
    with pytest.raises(ValueError, match="local relative"):
        cli.assemble_direction_record(**ARGS, witness_manifest=manifest)
    metadata["artifact_file"] = "symlink.txt"
    manifest.write_text(json.dumps([metadata]))
    with pytest.raises(ValueError, match="safe ordinary"):
        cli.assemble_direction_record(**ARGS, witness_manifest=manifest)


def test_manifest_cannot_smuggle_postplay_fields_into_witness(tmp_path):
    folder = tmp_path / "evidence"
    folder.mkdir()
    (folder / "north.txt").write_bytes(A)
    bad = claim() | {"artifact_file": "north.txt", "result": "scored"}
    path = folder / "manifest.json"
    path.write_text(json.dumps([bad]))
    with pytest.raises(ValueError, match="Malformed"):
        cli.assemble_direction_record(**ARGS, witness_manifest=path)



def test_buffalo_direction_receipt_display_validates_but_never_authenticates():
    data = adjudicate()
    shown = direction.validate_buffalo_direction_receipt_for_display(data)
    assert shown["claimed_attack_end"] == "NORTH"
    assert shown["source_count"] == 1
    assert shown["retrospective_source_present"] is True
    assert shown["numeric_attack_bearing_available"] is False
    assert shown["independently_verified"] is False
    assert shown["may_change_official_forecast"] is False
    assert "NOT independently authenticated" in shown["caveat"]


@pytest.mark.parametrize("field,value", [
    ("status", "OFFICIAL_PROVEN_DIRECTION"),
    ("season", 2027),
    ("claimed_attack_end", "WEST"),
    ("numeric_true_north_attack_azimuth_degrees", 0),
    ("numeric_headwind_crosswind_permitted", True),
    ("trusted_live_forecast_provenance", True),
    ("may_change_frozen_go_fg_punt_probabilities", True),
    ("source_count", 2),
    ("retrospective_source_present", False),
    ("all_sources_asserted_available_predecision", True),
    ("no_post_snap_labels_ingested", False),
])
def test_buffalo_unsigned_display_receipt_cannot_escalate_authority(field, value):
    record = deepcopy(adjudicate())
    record[field] = value
    with pytest.raises(ValueError):
        direction.validate_buffalo_direction_receipt_for_display(record)


def test_buffalo_direction_receipt_rejects_changed_witness_hash_or_end():
    record = adjudicate()
    forged = deepcopy(record)
    forged["witnesses"][0]["original_local_artifact_sha256"] = "not_a_hash"
    with pytest.raises(ValueError, match="metadata"):
        direction.validate_buffalo_direction_receipt_for_display(forged)
    forged = deepcopy(record)
    forged["witnesses"][0]["claim_north_south_end"] = "SOUTH"
    with pytest.raises(ValueError, match="disagree"):
        direction.validate_buffalo_direction_receipt_for_display(forged)
    forged = deepcopy(record)
    forged["witnesses"][0]["artifact_available_at_utc"] = (
        "2026-10-12T20:29:52Z"
    )
    with pytest.raises(ValueError, match="available-time|claims conflict"):
        direction.validate_buffalo_direction_receipt_for_display(forged)
