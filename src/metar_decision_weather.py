"""NOAA Aviation Weather Center METAR -> timestamped off-field NFL weather evidence.

Research ONLY. A METAR is an *airport* observation, not a field/snap sensor.
No production recommendations, model weights, new training or 2026/2027
football outcome retrieval are authorized. Raw NOAA weather MAY be fetched
explicitly by scripts/capture_awc_decision_weather.py; this module is
pure/offline. AWC receiptTime is used as an as-of gate, not as proof that the
football analyst personally obtained the record in real time.

METAR winds blow FROM a true-north bearing; the offense ATTACKING bearing
must be independently supplied for a head/crosswind projection. Wind speed
in METARs is knots, not mph. Missing/variable direction remains UNKNOWN.
Rain/snow codes describe an airport report, never measured field intensity.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

KNOT_TO_MPH = 1.150779448
MAX_OBSERVATION_AGE_MINUTES = 90
MAX_AIRPORT_TO_FIELD_KM = 30.0
ALLOWED_ROOFS = frozenset({"outdoors", "open"})
STATION = re.compile(r"[A-Z][A-Z0-9]{3}\Z")
WIND = re.compile(
    r"(?P<from>VRB|[0-3][0-9]{2})(?P<speed>[0-9]{2,3})"
    r"(?:G(?P<gust>[0-9]{2,3}))?KT\Z"
)
METAR_TIME = re.compile(r"(?P<day>[0-3][0-9])(?P<hour>[0-2][0-9])"
                        r"(?P<minute>[0-5][0-9])Z\Z")
# An explicit present-weather group before RMK, not a report narrative.
WEATHER = re.compile(
    r"(?:[-+]|VC)?(?:TS|SH|FZ|BL|DR|MI|BC|PR)?"
    r"(?:(?:RA|SN|DZ|PL|SG|GR|GS)){1,3}\Z"
)
NO_REAL_POLICY = "METAR_OFF_FIELD_RESEARCH_CONTEXT_ONLY"


def utc_time(value: Any) -> datetime:
    """Require an explicit timezone; reject false-naive UTC assumptions."""
    if isinstance(value, bool):
        raise ValueError("A real offset-aware UTC observation time is required")
    if type(value) in (int, float):
        if not math.isfinite(value) or not 0 <= value <= 4_102_444_800:
            raise ValueError("Invalid weather observation timestamp")
        return datetime.fromtimestamp(value, timezone.utc)
    if not isinstance(value, str) or len(value) > 45:
        raise ValueError("UTC timestamp must be an ISO string or epoch seconds")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid ISO weather observation time") from exc
    if parsed.tzinfo is None:
        raise ValueError("UTC weather time must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _iso(time: datetime) -> str:
    return time.isoformat(timespec="seconds").replace("+00:00", "Z")


def _number(value: Any, low: float, high: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} is missing or not finite")
    if not low <= value <= high:
        raise ValueError(f"{name} outside allowed physical range")
    return float(value)


def _position(lat: Any, lon: Any, name: str) -> tuple[float, float]:
    return (
        _number(lat, -90, 90, f"{name} latitude"),
        _number(lon, -180, 180, f"{name} longitude"),
    )


def distance_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Great-circle proximity only, not a claim of microclimate equality."""
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    a = (
        math.sin((lat2-lat1)/2)**2
        + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, a)))


def parse_raw_metar(raw: Any) -> dict[str, Any]:
    """Fail closed on ambiguous report headers, unsupported units or wind."""
    if not isinstance(raw, str) or not 15 <= len(raw) <= 1800:
        raise ValueError("A complete bounded raw METAR is required")
    words = raw.upper().split()
    if words[0] in ("METAR", "SPECI"):
        words = words[1:]
    if len(words) < 3 or not STATION.fullmatch(words[0]):
        raise ValueError("METAR must start with a four-character station code")
    if not METAR_TIME.fullmatch(words[1]):
        raise ValueError("METAR observation time missing")
    stamp = METAR_TIME.fullmatch(words[1])
    if int(stamp["day"]) not in range(1, 32) or int(stamp["hour"]) > 23:
        raise ValueError("METAR day/hour is invalid")
    tail = words[2:]
    while tail and tail[0] in ("AUTO", "COR"):
        tail = tail[1:]
    # Wind must be in the next group; don't accidentally parse a remark.
    if not tail or not WIND.fullmatch(tail[0]):
        raise ValueError("No unambiguous METAR wind group in knots")
    w = WIND.fullmatch(tail[0])
    from_code = w["from"]
    speed = float(int(w["speed"]))
    gust = float(int(w["gust"])) if w["gust"] else None
    if speed > 150 or (gust is not None and (gust < speed or gust > 200)):
        raise ValueError("METAR wind/gust speed physically implausible")
    if from_code == "000" and speed != 0:
        raise ValueError("000 direction allowed only for calm wind")
    if from_code != "VRB" and int(from_code) > 360:
        raise ValueError("Invalid METAR wind-from direction")
    if speed == 0 and gust is not None:
        raise ValueError("Calm wind cannot contain a gust")
    from_deg = (
        None if from_code == "VRB"
        else float(int(from_code) % 360)
    )
    # Variable-sector "120V200" means a nominal direction cannot be
    # translated into one credible field-relative head/crosswind.
    variable_sector = any(
        re.fullmatch(r"[0-3][0-9]{2}V[0-3][0-9]{2}", token)
        for token in tail[1:5]
    )
    at_rmk = tail.index("RMK") if "RMK" in tail else len(tail)
    present = tail[1:at_rmk]
    reported_rain = False
    reported_snow = False
    for token in present:
        if not WEATHER.fullmatch(token):
            continue
        reported_rain |= any(x in token for x in ("RA", "DZ"))
        reported_snow |= any(x in token for x in ("SN", "PL", "SG"))
    if reported_rain and reported_snow:
        precip = "AIRPORT_MIXED_PRECIP_REPORTED"
    elif reported_rain:
        precip = "AIRPORT_RAIN_REPORTED"
    elif reported_snow:
        precip = "AIRPORT_SNOW_ICE_REPORTED"
    else:
        precip = "AIRPORT_PRECIP_NOT_REPORTED_UNKNOWN"
    return {
        "station_id": words[0],
        "metar_day": int(stamp["day"]),
        "metar_hour": int(stamp["hour"]),
        "metar_minute": int(stamp["minute"]),
        "wind_from_true_degrees": from_deg,
        "wind_knots": speed,
        "gust_knots": gust,
        "wind_direction_variable": from_code == "VRB" or variable_sector,
        "present_weather_class": precip,
        "field_precipitation_intensity_measured": False,
        "airport_precipitation_is_field_precipitation": False,
    }


def _decoded_wind_consistent(sample: dict[str, Any], parsed: dict[str, Any]) -> None:
    """Treat raw coded weather as primary, refuse contradictory JSON fields."""
    if "wspd" in sample and sample["wspd"] is not None:
        if _number(sample["wspd"], 0, 150, "decoded wind knots") != parsed["wind_knots"]:
            raise ValueError("NOAA decoded wind speed conflicts with raw METAR")
    reported = sample.get("wdir")
    if reported is None or reported == "VRB":
        return
    direction = _number(reported, 0, 360, "decoded wind from degrees") % 360
    if parsed["wind_direction_variable"] and parsed["wind_from_true_degrees"] is None:
        raise ValueError("NOAA decoded wind direction contradicts variable raw METAR")
    if parsed["wind_from_true_degrees"] is not None and (
        direction != parsed["wind_from_true_degrees"]
    ):
        raise ValueError("NOAA decoded wind direction differs from raw METAR")


def _get_report_time(sample: dict[str, Any], parsed: dict[str, Any]) -> datetime:
    observed = utc_time(sample.get("obsTime"))
    if (observed.day != parsed["metar_day"]
            or observed.hour != parsed["metar_hour"]
            or observed.minute != parsed["metar_minute"]):
        raise ValueError("METAR encoded observation time and NOAA UTC timestamp disagree")
    return observed


def _project_wind(
    *, wind_knots: float, wind_from_degrees: float | None,
    wind_direction_variable: bool, offense_heading: float,
) -> dict[str, Any]:
    speed_mph = wind_knots*KNOT_TO_MPH
    if wind_direction_variable:
        return {
            "orientation_status": "VARIABLE_WIND_DIRECTION_UNRESOLVED",
            "headwind_mph_signed": None, "crosswind_from_right_mph_signed": None,
            "tailwind_mph": None,
        }
    if speed_mph == 0:
        return {
            "orientation_status": "CALM_WIND",
            "headwind_mph_signed": 0.0, "crosswind_from_right_mph_signed": 0.0,
            "tailwind_mph": 0.0,
        }
    if wind_from_degrees is None:
        raise ValueError("Noncalm METAR needs a definite wind-from direction")
    theta = math.radians(wind_from_degrees-offense_heading)
    head = speed_mph*math.cos(theta)
    cross = speed_mph*math.sin(theta)
    return {
        "orientation_status": "COMPUTED_FROM_STATION_WIND_AND_SUPPLIED_FIELD_HEADING",
        "headwind_mph_signed": round(head, 3),
        "crosswind_from_right_mph_signed": round(cross, 3),
        "tailwind_mph": round(max(0.0, -head), 3),
    }


def build_metar_decision_evidence(
    *, observations: Any, station_id: str, decision_at_utc: Any,
    captured_at_utc: Any, field_lat: Any, field_lon: Any,
    offense_attack_heading_true_deg: Any, roof: str,
    source_bytes: bytes, fetched_from_awc_https: bool,
) -> dict[str, Any]:
    """Select only an issued-before-decision nearby station observation.

    A backfilled archive can support retrospective weather context, NOT a
    claim that an algorithm saw the station weather before the real snap.
    """
    if roof not in ALLOWED_ROOFS:
        raise ValueError("Only open-roof/outdoor field weather is applicable")
    if not isinstance(station_id, str) or not STATION.fullmatch(station_id):
        raise ValueError("Explicit ICAO station ID required")
    if (type(fetched_from_awc_https) is not bool
            or not isinstance(source_bytes, bytes)
            or not 2 <= len(source_bytes) <= 200_000):
        raise ValueError("Original bounded source JSON bytes and source type required")
    if not isinstance(observations, list) or not 1 <= len(observations) <= 400:
        raise ValueError("NOAA AWC weather result must be a small nonempty list")
    try:
        decoded_original = json.loads(source_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Original NOAA JSON bytes cannot be decoded") from exc
    if decoded_original != observations:
        raise ValueError("Parsed observations do not match hashed source bytes")
    decision = utc_time(decision_at_utc)
    fetched = utc_time(captured_at_utc)
    field = _position(field_lat, field_lon, "football field")
    heading = _number(
        offense_attack_heading_true_deg, 0, 360, "offense attacking true bearing"
    ) % 360
    candidates = []
    for sample in observations:
        if not isinstance(sample, dict):
            raise ValueError("Malformed NOAA METAR JSON observation")
        if sample.get("icaoId") != station_id:
            raise ValueError("NOAA METAR station mismatched requested ICAO")
        parsed = parse_raw_metar(sample.get("rawOb"))
        if parsed["station_id"] != station_id:
            raise ValueError("NOAA raw METAR station does not match request")
        _decoded_wind_consistent(sample, parsed)
        observed = _get_report_time(sample, parsed)
        receipt = utc_time(sample.get("receiptTime"))
        if receipt < observed:
            raise ValueError("NOAA receipt precedes reported METAR time")
        station_pos = _position(sample.get("lat"), sample.get("lon"), "METAR airport")
        km = distance_km(field, station_pos)
        if km > MAX_AIRPORT_TO_FIELD_KM:
            raise ValueError("Selected METAR airport too far from football field")
        # No future observation or report received after decision can be
        # selected. This is checked BEFORE comparing candidate recency.
        age = (decision-observed).total_seconds()/60
        if 0 <= age <= MAX_OBSERVATION_AGE_MINUTES and receipt <= decision:
            candidates.append((observed, receipt, parsed, km, sample))
    if not candidates:
        raise ValueError(
            "No geographically close METAR observation published and "
            "received by NOAA before the football decision (within 90 minutes)"
        )
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    observed, receipt, parsed, km, sample = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == observed:
        other = candidates[1][4]
        if other.get("rawOb") != sample.get("rawOb"):
            raise ValueError("Ambiguous duplicated station weather at same observation time")
    projections = _project_wind(
        wind_knots=parsed["wind_knots"],
        wind_from_degrees=parsed["wind_from_true_degrees"],
        wind_direction_variable=parsed["wind_direction_variable"],
        offense_heading=heading,
    )
    capture_after_decision = fetched > decision
    if fetched < receipt:
        raise ValueError("Claimed source retrieval precedes NOAA receipt time")
    return {
        "status": NO_REAL_POLICY,
        "station_id": station_id,
        "station_distance_km": round(km, 3),
        "offense_attack_heading_true_deg": heading,
        "user_supplied_field_direction_independently_verified": False,
        "observed_at_utc": _iso(observed),
        "noaa_receipt_at_utc": _iso(receipt),
        "decision_at_utc": _iso(decision),
        "captured_at_utc": _iso(fetched),
        "observation_age_minutes": round((decision-observed).total_seconds()/60, 3),
        "observed_and_received_before_decision": True,
        "source_downloaded_after_decision": capture_after_decision,
        "source_received_by_user_before_decision_independently_verified": False,
        "provider_origin_https_verified_for_this_capture": fetched_from_awc_https,
        "raw_metar": sample["rawOb"],
        "raw_metar_sha256": hashlib.sha256(
            sample["rawOb"].encode("utf-8")
        ).hexdigest(),
        "source_response_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_endpoint": (
            f"https://aviationweather.gov/api/data/metar?ids={station_id}&format=json"
        ),
        "wind_speed_knots": parsed["wind_knots"],
        "wind_speed_mph": round(parsed["wind_knots"]*KNOT_TO_MPH, 3),
        "wind_from_true_degrees": parsed["wind_from_true_degrees"],
        "wind_direction_variable": parsed["wind_direction_variable"],
        **projections,
        "airport_present_weather_code": parsed["present_weather_class"],
        "airport_precipitation_is_not_field_intensity": True,
        "precipitation_mm_per_hour_at_field": None,
        "on_field_sensor_verified": False,
        "snap_specific_weather_verified": False,
        "real_time_model_weather_provenance_certified": False,
        "may_change_official_go_fg_or_punt_model": False,
        "real_2027_outcomes_accessed": False,
        "caveats": (
            "Station-level METAR within 30 km and 90 min, not sideline weather. "
            "Field heading/coordinates must be verified independently. "
            "No signed field sensor or rain rate; retrospective downloads "
            "are NOT authentic pre-decision model-issued weather forecasts."
        ),
    }



def validate_metar_receipt_for_display(receipt: Any) -> dict[str, Any]:
    """Validate structure/arithmetic before showing a pasted UNSIGNED receipt.

    Source response authenticity and field heading are still unverified.
    Never use this record as an input to frozen or experimental models.
    """
    if not isinstance(receipt, dict) or len(receipt) > 60:
        raise ValueError("Expected a single bounded weather research receipt")
    if receipt.get("status") != NO_REAL_POLICY:
        raise ValueError("Only non-production station METAR receipts are allowed")
    for key in (
        "user_supplied_field_direction_independently_verified",
        "source_received_by_user_before_decision_independently_verified",
        "on_field_sensor_verified", "snap_specific_weather_verified",
        "real_time_model_weather_provenance_certified",
        "may_change_official_go_fg_or_punt_model",
        "real_2027_outcomes_accessed",
    ):
        if receipt.get(key) is not False:
            raise ValueError("Unapproved METAR authority or model claim")
    station = receipt.get("station_id")
    parsed = parse_raw_metar(receipt.get("raw_metar"))
    if parsed["station_id"] != station:
        raise ValueError("Receipt METAR airport differs from reported airport")
    digest = hashlib.sha256(receipt["raw_metar"].encode("utf-8")).hexdigest()
    if digest != receipt.get("raw_metar_sha256"):
        raise ValueError("Pasted raw METAR was modified since hashing")
    observed = utc_time(receipt.get("observed_at_utc"))
    receipt_time = utc_time(receipt.get("noaa_receipt_at_utc"))
    decision = utc_time(receipt.get("decision_at_utc"))
    captured = utc_time(receipt.get("captured_at_utc"))
    if (observed.day, observed.hour, observed.minute) != (
        parsed["metar_day"], parsed["metar_hour"], parsed["metar_minute"]
    ):
        raise ValueError("Pasted METAR and time do not match")
    if not (
        observed <= receipt_time <= decision
        and 0 <= (decision-observed).total_seconds()/60 <= MAX_OBSERVATION_AGE_MINUTES
        and receipt_time <= captured
    ):
        raise ValueError("METAR time provenance was modified or is stale")
    if receipt.get("source_downloaded_after_decision") is not (
        captured > decision
    ):
        raise ValueError("METAR capture timing contradicts receipt")
    km = _number(
        receipt.get("station_distance_km"),
        0, MAX_AIRPORT_TO_FIELD_KM,
        "distance to stadium",
    )
    if km > MAX_AIRPORT_TO_FIELD_KM:
        raise ValueError("Reported airport outside proximity cap")
    heading = _number(
        receipt.get("offense_attack_heading_true_deg"),
        0, 360, "offense bearing",
    )
    if parsed["wind_knots"] != receipt.get("wind_speed_knots"):
        raise ValueError("Pasted METAR wind knots and raw do not agree")
    if (
        receipt.get("wind_from_true_degrees")
        != parsed["wind_from_true_degrees"]
        or receipt.get("wind_direction_variable")
        is not parsed["wind_direction_variable"]
        or receipt.get("airport_present_weather_code")
        != parsed["present_weather_class"]
    ):
        raise ValueError("Pasted METAR wind/weather conflicts with coded report")
    projected = _project_wind(
        wind_knots=parsed["wind_knots"],
        wind_from_degrees=parsed["wind_from_true_degrees"],
        wind_direction_variable=parsed["wind_direction_variable"],
        offense_heading=heading,
    )
    for field in (
        "orientation_status", "headwind_mph_signed",
        "crosswind_from_right_mph_signed", "tailwind_mph",
    ):
        if receipt.get(field) != projected[field]:
            raise ValueError("Pasted METAR wind-vector arithmetic changed")
    if receipt.get("precipitation_mm_per_hour_at_field") is not None:
        raise ValueError("Airport METAR does not certify field precipitation rate")
    return {
        "airport": station,
        "distance_km": km,
        "observed_at_utc": _iso(observed),
        "age_minutes": round((decision-observed).total_seconds()/60, 1),
        "captured_after_decision": captured > decision,
        "heading_true_deg": heading,
        "wind_speed_mph": round(parsed["wind_knots"]*KNOT_TO_MPH, 2),
        "orientation_status": projected["orientation_status"],
        "headwind_mph_signed": projected["headwind_mph_signed"],
        "crosswind_from_right_mph_signed": projected[
            "crosswind_from_right_mph_signed"
        ],
        "airport_precipitation": parsed["present_weather_class"],
        "unsigned_manual_import_not_source_authenticated": True,
        "field_precipitation_intensity_verified": False,
        "changes_official_recommendation": False,
    }
