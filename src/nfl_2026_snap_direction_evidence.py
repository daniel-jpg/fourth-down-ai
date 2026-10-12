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

from datetime import datetime, date
import hashlib
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
    schedule_day = date.fromisoformat(game_date_et)
    if not 0 <= (decision.date()-schedule_day).days <= 1:
        raise ValueError("UTC decision must fall on game schedule day or following UTC day")
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
        or len(set(x["source_url_user_asserted"] for x in sightings))
            != len(sightings)
    ):
        raise ValueError("Duplicate witness, source, URL or file is not corroboration")
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



def validate_buffalo_direction_receipt_for_display(receipt: Any) -> dict[str, Any]:
    """Validate a pasted UNSIGNED direction receipt; never trust its sources.

    Media bytes are NOT in the receipt, so the hashes and witness URLs
    cannot be independently authenticated here. This is a self-consistency
    display validator only, not approval to use the compass heading.
    """
    if not isinstance(receipt, dict) or len(receipt) > 55:
        raise ValueError("Expected a bounded Buffalo research receipt")
    if receipt.get("status") != RECEIPT_STATUS:
        raise ValueError("Only 2026 Buffalo research direction receipts allowed")
    if (
        receipt.get("season") != 2026 or type(receipt.get("season")) is not int
        or receipt.get("venue_team") != "BUF"
        or receipt.get("claimed_attack_end") not in ENDS
        or receipt.get("numeric_true_north_attack_azimuth_degrees") is not None
    ):
        raise ValueError("No surveyed numeric direction or other venue is allowed")
    for field in (
        "numeric_headwind_crosswind_permitted",
        "direction_of_attack_independently_certified",
        "trusted_live_forecast_provenance",
        "source_content_authentication_independent",
        "game_home_venue_confirmed_by_trusted_schedule_or_feed",
        "protected_2026_fourth_down_outcomes_evaluated",
        "real_2027_outcomes_accessed",
        "may_change_frozen_go_fg_punt_probabilities",
        "source_bytes_in_receipt",
    ):
        if receipt.get(field) is not False:
            raise ValueError("Direction receipt falsely claims trusted provenance")
    if receipt.get("no_post_snap_labels_ingested") is not True:
        raise ValueError("Snap evidence cannot use post-play result labels")
    if set(receipt) & FORBIDDEN_FIELDS:
        raise ValueError("Post-play output fields are not direction evidence")
    decision = utc_time(receipt.get("decision_at_utc_user_supplied"))
    witnesses = receipt.get("witnesses")
    if (
        not isinstance(witnesses, list) or not 1 <= len(witnesses) <= 2
        or receipt.get("source_count") != len(witnesses)
    ):
        raise ValueError("Direction witness source count is invalid")
    ids, observers, media_hashes, source_urls = set(), set(), set(), set()
    available_by_decision = []
    for witness in witnesses:
        if not isinstance(witness, dict):
            raise ValueError("Malformed source evidence metadata")
        if witness.get("claim_north_south_end") != receipt["claimed_attack_end"]:
            raise ValueError("Source witnesses disagree about physical direction")
        source_id, observer = witness.get("source_id"), witness.get("witness_id")
        media_hash = witness.get("original_local_artifact_sha256")
        source_url = witness.get("source_url_user_asserted")
        if (
            type(source_id) is not str or not SOURCE_ID.fullmatch(source_id)
            or type(observer) is not str or not SOURCE_ID.fullmatch(observer)
            or type(media_hash) is not str or
                not re.fullmatch(r"[0-9a-f]{64}", media_hash)
            or type(source_url) is not str
            or not source_url.startswith("https://")
            or witness.get("artifact_bytes_not_in_receipt") is not True
            or witness.get("independently_authenticated") is not False
            or witness.get(
                "video_frame_or_observer_location_independently_verified"
            ) is not False
        ):
            raise ValueError("Direction media metadata is not trust-safe")
        obs = utc_time(witness.get("artifact_observed_at_utc"))
        avail = utc_time(witness.get("artifact_available_at_utc"))
        if (
            obs > decision or avail < obs
            or (decision - obs).total_seconds() > MAX_OBSERVATION_LEAD_SECONDS
        ):
            raise ValueError("Video/observer timestamp conflicts with decision")
        is_pre = avail <= decision
        if witness.get("source_asserted_available_by_decision") is not is_pre:
            raise ValueError("Video source available-time claim is inconsistent")
        available_by_decision.append(is_pre)
        ids.add(source_id); observers.add(observer)
        media_hashes.add(media_hash); source_urls.add(source_url)
    if any(len(x) != len(witnesses) for x in (
        ids, observers, media_hashes, source_urls,
    )):
        raise ValueError("Duplicate direction witness metadata")
    is_alleged_pre = all(available_by_decision)
    if (
        receipt.get("all_sources_asserted_available_predecision") is not is_alleged_pre
        or receipt.get("retrospective_source_present") is not (not is_alleged_pre)
        or receipt.get("claim_consistency") != (
            "TWO_USER_ASSERTED_DISTINCT_WITNESSES"
            if len(witnesses) == 2 else "ONE_UNCORROBORATED_USER_ASSERTION"
        )
    ):
        raise ValueError("Unsigned receipt's directional evidence claims conflict")
    return {
        "claimed_attack_end": receipt["claimed_attack_end"],
        "source_count": len(witnesses),
        "claim_consistency": receipt["claim_consistency"],
        "retrospective_source_present": not is_alleged_pre,
        "decision_at_utc_user_supplied": decision.isoformat().replace("+00:00", "Z"),
        "numeric_attack_bearing_available": False,
        "independently_verified": False,
        "may_change_official_forecast": False,
        "caveat": (
            "Source-hash and witness fields are merely self-consistent; "
            "original media, attribution, observed play and availability "
            "are NOT independently authenticated. No numerical headwind."
        ),
    }
