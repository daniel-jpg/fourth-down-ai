"""One-shot NOAA Aviation Weather METAR -> off-field weather evidence.

No paid provider, weather model re-training, new NFL outcomes or cron.
Runs only when explicitly invoked, and never writes any official weights.

Use a recent, station-specific AWC HTTPS response (NOAA offers roughly
the preceding 30 days; this command only requests <=6 hours):
    python -m scripts.capture_awc_decision_weather \
      --live-noaa-awc --station KSEA --decision-utc 2026-10-12T18:30:00Z \
      --field-lat 47.595 --field-lon -122.332 \
      --offense-heading-true-deg 0 --roof outdoors \
      --out artifacts/research_metar_receipt.json

The example time/location/field bearing above are NOT a verified NFL snap
or verified field orientation. Replace every one with independently
confirmed values and use an observation already received by NOAA as of the
decision time; otherwise the command refuses to issue an evidence receipt.

Offline test/manual import:
    python -m scripts.capture_awc_decision_weather \
      --source-json fixture.json --captured-utc 2026-10-12T18:40:00Z \
      --station KSEA --decision-utc 2026-10-12T18:30:00Z \
      --field-lat 47.595 --field-lon -122.332 \
      --offense-heading-true-deg 0 --roof outdoors \
      --out artifacts/fixture_only_receipt.json
Offline source provenance is NOT verified as genuine NOAA HTTPS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from src.metar_decision_weather import (
    STATION, build_metar_decision_evidence, utc_time,
)

MAX_SOURCE_BYTES = 200_000
BASE_URL = "https://aviationweather.gov/api/data/metar"


def awc_url(station: str) -> str:
    if not isinstance(station, str) or not STATION.fullmatch(station):
        raise ValueError("Expected a single four-character ICAO station")
    return f"{BASE_URL}?ids={station}&format=json&hours=6"


def fetch_recent_awc_metar(station: str) -> tuple[bytes, datetime]:
    """One HTTPS GET. No event IDs, NFL game data or hourly auto-watches."""
    url = awc_url(station)
    req = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "fourth-down-ai-weather-research/1.0 (manual single request)",
    })
    with urlopen(req, timeout=12) as response:
        if response.geturl() != url:
            raise ValueError("Redirected AWC response is not approved")
        if response.status != 200:
            raise ValueError("AWC returned no usable METAR observations")
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" not in content_type:
            raise ValueError("AWC did not return JSON")
        raw = response.read(MAX_SOURCE_BYTES+1)
    captured = datetime.now(timezone.utc)
    if len(raw) > MAX_SOURCE_BYTES or not raw:
        raise ValueError("AWC response exceeds allowed single-station size")
    return raw, captured


def capture_weather(
    *, source_bytes: bytes, station: str, decision_utc: str,
    captured_utc: str | int | float, field_lat: float, field_lon: float,
    offense_heading_true_deg: float, roof: str,
    fetched_from_awc_https: bool,
) -> dict:
    if not 1 <= len(source_bytes) <= MAX_SOURCE_BYTES:
        raise ValueError("No bounded METAR JSON source")
    samples = json.loads(source_bytes.decode("utf-8"))
    packet = build_metar_decision_evidence(
        observations=samples, source_bytes=source_bytes,
        station_id=station, decision_at_utc=decision_utc,
        captured_at_utc=captured_utc,
        field_lat=field_lat, field_lon=field_lon,
        offense_attack_heading_true_deg=offense_heading_true_deg,
        roof=roof, fetched_from_awc_https=fetched_from_awc_https,
    )
    packet["source_endpoint"] = (
        awc_url(station) if fetched_from_awc_https
        else "OFFLINE_JSON_FIXTURE_NOT_A_VERIFIED_NOAA_FETCH"
    )
    packet["evidence_can_certify_genuine_on_field_weather"] = False
    packet["evidence_can_be_used_for_production_wps"] = False
    return packet


def write_weather_receipt(path: Path, packet: dict) -> None:
    """Never mutate a prior weather-evidence packet (or follow a symlink)."""
    if path.exists() or path.is_symlink():
        raise FileExistsError("A weather evidence receipt already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(packet, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as file:
        file.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--live-noaa-awc", action="store_true")
    choices.add_argument("--source-json", type=Path)
    parser.add_argument("--station", required=True)
    parser.add_argument("--decision-utc", required=True)
    parser.add_argument("--captured-utc")
    parser.add_argument("--field-lat", type=float, required=True)
    parser.add_argument("--field-lon", type=float, required=True)
    parser.add_argument("--offense-heading-true-deg", type=float, required=True)
    parser.add_argument("--roof", choices=["outdoors", "open"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Reject unrelated or duplicate receipts before any optional network GET.
    if args.out.exists() or args.out.is_symlink():
        parser.error("Never overwrite an existing weather evidence receipt")
    if args.source_json and not args.captured_utc:
        parser.error("--source-json requires explicit --captured-utc")
    if args.live_noaa_awc and args.captured_utc:
        parser.error("--live-noaa-awc captures its own retrieval time")
    try:
        awc_url(args.station)
        utc_time(args.decision_utc)
    except ValueError as exc:
        parser.error(str(exc))
    if args.live_noaa_awc:
        source, captured = fetch_recent_awc_metar(args.station)
    else:
        if args.source_json.is_symlink() or not args.source_json.is_file():
            parser.error("Offline METAR JSON must be a regular file")
        if args.source_json.stat().st_size > MAX_SOURCE_BYTES:
            parser.error("Offline source too large")
        source = args.source_json.read_bytes()
        captured = args.captured_utc
    result = capture_weather(
        source_bytes=source,
        station=args.station, decision_utc=args.decision_utc,
        captured_utc=captured,
        field_lat=args.field_lat, field_lon=args.field_lon,
        offense_heading_true_deg=args.offense_heading_true_deg,
        roof=args.roof, fetched_from_awc_https=args.live_noaa_awc,
    )
    write_weather_receipt(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "observation_age_minutes": result["observation_age_minutes"],
        "station_distance_km": result["station_distance_km"],
        "headwind_mph_signed": result["headwind_mph_signed"],
        "crosswind_from_right_mph_signed": result[
            "crosswind_from_right_mph_signed"
        ],
        "airport_present_weather_code": result["airport_present_weather_code"],
        "source_downloaded_after_decision": result[
            "source_downloaded_after_decision"
        ],
        "official_model_unchanged": not result[
            "may_change_official_go_fg_or_punt_model"
        ],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
