"""One-shot real NOAA AWC integration SMOKE TEST, not an NFL observation.

This is deliberately NOT a trained or official football input. It checks
only whether the actual live NOAA API still returns the JSON/raw METAR and
timestamps expected by our research collector. No NFL data is retrieved.

Run explicitly:
  PYTHONPATH=. python -m scripts.smoke_awc_live_format --real-noaa-only

The field position is deliberately set to the METAR STATION position
and the offensive attacking heading arbitrarily to 0 degrees to verify
the parser/math plumbing. These DO NOT represent any stadium or NFL play.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

from scripts.capture_awc_decision_weather import fetch_recent_awc_metar
from src.metar_decision_weather import (
    NO_REAL_POLICY, build_metar_decision_evidence, utc_time,
)

STATION_ID = "KSEA"


def live_noaa_smoke() -> dict:
    original_bytes, retrieved = fetch_recent_awc_metar(STATION_ID)
    reports = json.loads(original_bytes.decode("utf-8"))
    if not isinstance(reports, list) or not reports:
        raise ValueError("NOAA did not return any METAR observations")
    # This chooses STATION coordinates AS AN ARTIFICIAL reference position.
    # It proves the end-to-end real API schema, NOT airport-to-stadium
    # mapping, any verified football game time, or on-field weather.
    matches = [x for x in reports if isinstance(x, dict)
               and x.get("icaoId") == STATION_ID
               and isinstance(x.get("lat"), (int, float))
               and isinstance(x.get("lon"), (int, float))]
    if not matches:
        raise ValueError("NOAA supplied no usable KSEA station coordinates")
    station = matches[0]
    result = build_metar_decision_evidence(
        observations=reports, source_bytes=original_bytes,
        station_id=STATION_ID,
        decision_at_utc=retrieved.isoformat(),
        captured_at_utc=retrieved.isoformat(),
        field_lat=station["lat"], field_lon=station["lon"],
        offense_attack_heading_true_deg=0,
        roof="outdoors", fetched_from_awc_https=True,
    )
    if (result["status"] != NO_REAL_POLICY
            or result["station_distance_km"] != 0
            or result["observation_age_minutes"] > 90
            or not result["observed_and_received_before_decision"]
            or not result["provider_origin_https_verified_for_this_capture"]
            or result["may_change_official_go_fg_or_punt_model"] is not False
            or result["snap_specific_weather_verified"] is not False
            or result["real_2027_outcomes_accessed"] is not False
            or result["user_supplied_field_direction_independently_verified"] is not False):
        raise AssertionError("Live NOAA plumbing did not satisfy research safety")
    return {
        "status": "LIVE_NOAA_REAL_API_SCHEMA_RESEARCH_SMOKE_PASSED",
        "test_type": "LIVE_NOAA_KSEA_REAL_METAR_NO_NFL_GAME",
        "real_public_noaa_source_used": True,
        "football_stadium_location_used": False,
        "real_offense_attack_heading_used": False,
        "observed_utc": result["observed_at_utc"],
        "noaa_receipt_utc": result["noaa_receipt_at_utc"],
        "retrieved_utc": result["captured_at_utc"],
        "age_minutes": result["observation_age_minutes"],
        "wind_speed_mph": result["wind_speed_mph"],
        "wind_from_true_degrees": result["wind_from_true_degrees"],
        "wind_direction_variable": result["wind_direction_variable"],
        "airport_weather_code": result["airport_present_weather_code"],
        "source_response_sha256": result["source_response_sha256"],
        "airport_metar_sha256": result["raw_metar_sha256"],
        "raw_source_logged_or_published": False,
        "snap_level_conditions_proven": False,
        "validated_weather_prediction": False,
        "official_fourth_down_model_modified": False,
        "heldout_nfl_outcomes_accessed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-noaa-only", action="store_true", required=True)
    args = p.parse_args()
    if not args.real_noaa_only:
        p.error("Explicit real NOAA opt-in required")
    print(json.dumps(live_noaa_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
