"""Offline, source-hashed human ATTACK-END evidence for a 2026 BUF home snap.

No protected 2026 outcomes, NFL PBP download, web calls, broadcast viewing,
trusted video timestamp, surveyed numeric bearing or policy alteration.

Architect (Populous) and AP sources establish a *QUALITATIVE* N/S stadium
axis for Buffalo 2026, but which north/south end zone the offense attacks
must be witnessed per snap. Witness/footage metadata is user-declared.
Raw local media bytes can be hashed (never embedded in receipt) for audit,
but two user-declared witnesses do not provide an independent signed stream.

All outputs remain research notes. Sources accessed AFTER the alleged
decision are explicitly retrospective, never a live pre-snap forecast.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
import re
from typing import Any, Mapping

from .metar_decision_weather import utc_time
from .nfl_2026_international_venue_guard import guard_team_home_venue_for_game
from .nfl_stadium_weather_geometry import resolve_stadium_location

TYPES = frozenset({
    "BROADCAST_FRAME", "ON_SITE_CAMERA", "ON_SITE_OBSERVER_NOTE",
    "VENUE_VIDEO_SCREENSHOT",
})
ENDS = frozenset({"NORTH", "SOUTH"})
HEX = re.compile(r"[a-z0-9]{6,60}\Z", re.I)
SOURCE_ID = re.compile(r"[a-zA-Z0-9_-]{3,64}\Z")
MAX_EVIDENCE_BYTES = 12 * 1024 * 1024
MAX_OBSERVATION_LEAD_SECONDS = 150
RECEIPT_STATUS = "USER_PROVIDED_BUFFALO_DIRECTION_EVIDENCE_RESEARCH_ONLY"
FORBIDDEN_FIELDS = frozenset({
    "play_type", "result", "epa", "wp", "wpa",
    "success", "converted", "postplay", "winner",
    "touchdown", "field_goal_result", "home_score",
    "away_score", "scoring_play",
})


def _clock(value: Any) -> int:
    if type(value) is not str or not re.fullmatch(r"\d{1,2}:[0-5]\d", value):
        raise ValueError("Game clock must be MM:SS, not inferred from wall clock")
    minute, second = map(int, value.split(":"))
    if minute > 15:
        raise ValueError("Game clock outside regulation quarter range")
    return minute * 60 + second


def _sighting(
    row: Any, source: bytes, *, decision: datetime,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Snap-direction witness needs explicit metadata")
    if set(row) & FORBIDDEN_FIELDS:
        raise PermissionError("Post-play outcomes cannot enter direction evidence")
    expected = {
        "source_id", "witness_id", "source_type", "artifact_observed_at_utc",
        "artifact_available_at_utc", "attack_end_claim",
        "location_landmark_observed", "source_url",
    }
    if set(row) != expected:
        raise ValueError("Unexpected witness evidence schema")
    sid = row["source_id"]
    witness = row["witness_id"]
    if (
        type(sid) is not str or not SOURCE_ID.fullmatch(sid)
        or type(witness) is not str or not SOURCE_ID.fullmatch(witness)
    ):
        raise ValueError("Distinct nonblank witness/source identifiers required")
    if row["source_type"] not in TYPES or type(row["source_type"]) is not str:
        raise ValueError("Unrecognized observation source type")
    if row["attack_end_claim"] not in ENDS or type(row["attack_end_claim"]) is not str:
        raise ValueError("Witness must explicitly identify NORTH or SOUTH")
    landmark = row["location_landmark_observed"]
    if not isinstance(landmark, str) or not 12 <= len(landmark) <= 400:
        raise ValueError("Witness must identify the geographic field landmark")
    if not any(x in landmark.lower() for x in ("north", "south", "end zone", "endzone")):
        raise ValueError("Witness landmark must physically anchor a north/south end zone")
    url = row["source_url"]
    if (
        type(url) is not str or not url.startswith("https://")
        or len(url) > 800 or any(x in url for x in ("\n", "\r", "@"))
    ):
        raise ValueError("Source citation must use a bounded external HTTPS URL")
    observed = utc_time(row["artifact_observed_at_utc"])
    available = utc_time(row["artifact_available_at_utc"])
    if not (
        observed <= decision and
        0 <= (decision - observed).total_seconds() <= MAX_OBSERVATION_LEAD_SECONDS
    ):
        raise ValueError("Direction witness observation is after/too far from decision")
    if available < observed:
        raise ValueError("Source cannot be available before underlying observation")
    if not isinstance(source, bytes) or not 8 <= len(source) <= MAX_EVIDENCE_BYTES:
        raise ValueError("Original captured source file bytes are required")
    return {
        "source_id": sid,
        "witness_id": witness,
        "source_type": row["source_type"],
        "source_url_user_asserted": url,
        "artifact_observed_at_utc": observed.isoformat().replace("+00:00", "Z"),
        "artifact_available_at_utc": available.isoformat().replace("+00:00", "Z"),
        "claim_north_south_end": row["attack_end_claim"],
        "landmark_user_claim": landmark,
        "original_local_artifact_sha256": hashlib.sha256(source).hexdigest(),
        "artifact_bytes_not_in_receipt": True,
        "source_asserted_available_by_decision": available <= decision,
        "independently_authenticated": False,
        "video_frame_or_observer_location_independently_verified": False,
    }


def assess_2026_buffalo_snap_direction(
    *,
    season: Any, venue_team: Any, opponent_team: Any,
    offense_team: Any, game_date_et: Any, play_key: Any,
    quarter: Any, game_clock_mmss: Any, decision_at_utc: Any,
    witness_metadata: Any, original_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Record 1-2 user-asserted, hashed per-snap direction observations.

    With conflicting witnesses fail closed. Two independent witness IDs
    are REPORTED corroboration only, not verified truth. One source is
    marked uncorroborated. None gives an authenticated real-time or
    numeric weather model feature, regardless of availability.
    """
    if type(season) is not int or season != 2026:
        raise PermissionError("Buffalo snap-direction records support 2026 only")
    if venue_team != "BUF" or type(venue_team) is not str:
        raise ValueError("Only the new 2026 Buffalo field is supported")
    if (
        type(opponent_team) is not str or
        type(offense_team) is not str or
        opponent_team == "BUF" or
        offense_team not in {"BUF", opponent_team}
    ):
        raise ValueError("Offense must be Buffalo or its declared opponent")
    # A scheduled international fixture may never use Buffalo's home venue.
    venue_schedule_check = guard_team_home_venue_for_game(
        game_date_et=game_date_et, venue_team=venue_team,
        opponent_team=opponent_team,
    )
    location = resolve_stadium_location(team="BUF", season=2026)
    if (
        location["qualitative_field_orientation"] !=
        "REPORTED_APPROX_NORTH_SOUTH_NOT_SURVEYED"
        or location["field_axis_degrees_approx"] is not None
        or location["surveyed_numeric_headwind_projection_approved"] is not False
    ):
        raise ValueError("Buffalo exact field azimuth is not surveyed")
    if (
        type(play_key) is not str or not HEX.fullmatch(play_key)
        or not (type(quarter) is int and quarter in (1, 2, 3, 4))
    ):
        raise ValueError("Play ID and regulation quarter must be explicit")
    remaining = _clock(game_clock_mmss)
    decision = utc_time(decision_at_utc)
    if (
        not isinstance(witness_metadata, list)
        or not 1 <= len(witness_metadata) <= 2
        or not isinstance(original_artifacts, Mapping)
    ):
        raise ValueError("One or two local direction evidence artifacts required")
    identities = [row.get("source_id") for row in witness_metadata if isinstance(row, dict)]
    if len(identities) != len(witness_metadata) or set(identities) != set(original_artifacts):
        raise ValueError("Local original artifacts must match source IDs exactly")
    sightings = [
        _sighting(row, original_artifacts[row["source_id"]], decision=decision)
        for row in witness_metadata
    ]
    if (
        len(set(x["source_id"] for x in sightings)) != len(sightings)
        or len(set(x["witness_id"] for x in sightings)) != len(sightings)
        or len(set(x["original_local_artifact_sha256"] for x in sightings))
            != len(sightings)
    ):
        raise ValueError("Duplicate witness, source or file is not corroboration")
    claimed_ends = {x["claim_north_south_end"] for x in sightings}
    if len(claimed_ends) > 1:
        raise ValueError("Conflicting north/south snap direction claims: abstain")
    north_or_south = next(iter(claimed_ends))
    alleged_live = all(x["source_asserted_available_by_decision"] for x in sightings)
    return {
        "status": RECEIPT_STATUS,
        "season": 2026,
        "game_date_et": game_date_et,
        "venue_team": venue_team,
        "opponent_team": opponent_team,
        "offense_team": offense_team,
        "play_id_user_supplied": play_key,
        "quarter_user_supplied": quarter,
        "game_clock_mmss_user_supplied": game_clock_mmss,
        "game_clock_seconds_user_supplied": remaining,
        "decision_at_utc_user_supplied": decision.isoformat().replace("+00:00", "Z"),
        "claimed_attack_end": north_or_south,
        "physical_endzone_orientation_source": (
            "ARCHITECT_POPULOUS_AND_AP_QUALITATIVE_NORTH_SOUTH"
        ),
        "numeric_true_north_attack_azimuth_degrees": None,
        "numeric_headwind_crosswind_permitted": False,
        "claim_consistency": (
            "TWO_USER_ASSERTED_DISTINCT_WITNESSES" if len(sightings) == 2
            else "ONE_UNCORROBORATED_USER_ASSERTION"
        ),
        "source_count": len(sightings),
        "all_sources_asserted_available_predecision": alleged_live,
        "retrospective_source_present": not alleged_live,
        "source_citations_user_claimed_not_fetched": True,
        "source_content_authentication_independent": False,
        "direction_of_attack_independently_certified": False,
        "trusted_live_forecast_provenance": False,
        "no_post_snap_labels_ingested": True,
        "game_home_venue_confirmed_by_trusted_schedule_or_feed": False,
        "known_international_venue_guard": venue_schedule_check,
        "witnesses": sightings,
        "source_bytes_in_receipt": False,
        "protected_2026_fourth_down_outcomes_evaluated": False,
        "real_2027_outcomes_accessed": False,
        "may_change_frozen_go_fg_punt_probabilities": False,
        "caveat": (
            "This is a source-hashed, USER-SUPPLIED descriptive allegation "
            "about an observed Buffalo 2026 play. Witness IDs, local files, "
            "timestamps, game identity, field landmarks and end zone are "
            "NOT independently authenticated. Even two matching reports "
            "do NOT prove snap attack bearing or a live model prediction. "
            "No true azimuth or weather-model feature is output."
        ),
    }
