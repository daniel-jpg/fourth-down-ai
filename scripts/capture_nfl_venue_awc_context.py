"""2026 NFL venue + nearby NOAA airport METAR: research-only receipt.

Unlike the low-level NOAA CLI, users need not guess stadium coordinates
and can explicitly choose one of two opposite approximate field-axis
directions. The lookup contains no automatic game/quarter/team direction:
the end zone currently attacked MUST be independently checked, and the
venue must be physically where the game is being played.

This is NOT an actual field/snap weather sensor or official policy input.
No 2027 venue map, no NFL play data, no downloaded future results.

Offline example, with a historical *fabricated* KBFI METAR JSON file:
  PYTHONPATH=. python -m scripts.capture_nfl_venue_awc_context \
    --source-json fixtures.json --captured-utc 2026-10-12T18:45:00Z \
    --venue-team SEA --season 2026 --attack-end A --station KBFI \
    --decision-utc 2026-10-12T18:35:00Z --roof outdoors \
    --out artifacts/synthetic_research_receipt.json

Real NOAA airport report, only with user-confirmed venue and direction:
  PYTHONPATH=. python -m scripts.capture_nfl_venue_awc_context \
    --live-noaa-awc --venue-team SEA --season 2026 --attack-end B \
    --station KBFI --decision-utc <recent-UTC-date-time> --roof outdoors \
    --out artifacts/station_context.json

In BOTH modes the community-derived field axis and the physical roof
remain UNVERIFIED; manual team selection never establishes an NFL game.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.capture_awc_decision_weather import (
    MAX_SOURCE_BYTES, capture_weather, fetch_recent_awc_metar,
    write_weather_receipt,
)
from src.metar_decision_weather import utc_time
from src.nfl_stadium_weather_geometry import (
    ATTACK_ENDS, resolve_stadium_geometry,
)


def capture_venue_weather(
    *, venue_team: str, season: int, attack_end: str, station: str,
    decision_utc: str, roof: str, source_bytes: bytes,
    captured_utc: str | int | float,
    fetched_from_awc_https: bool,
) -> dict:
    """No network, filesystem writes, game outcomes or model feature use."""
    geometry = resolve_stadium_geometry(
        team=venue_team, season=season,
        offense_attack_end=attack_end,
    )
    # The stadium geometry is community-sourced; never claim it is a
    # verified snap-specific wind measurement, even for real NOAA HTTPS.
    receipt = capture_weather(
        source_bytes=source_bytes, station=station,
        decision_utc=decision_utc, captured_utc=captured_utc,
        field_lat=geometry["field_lat"], field_lon=geometry["field_lon"],
        offense_heading_true_deg=geometry[
            "offense_attack_heading_true_deg_approx"
        ],
        roof=roof, fetched_from_awc_https=fetched_from_awc_https,
    )
    receipt["venue_geometry"] = geometry
    receipt["stadium_field_axis_community_estimate_only"] = True
    receipt["attack_end_requires_independent_play_confirmation"] = True
    receipt["actual_game_at_this_venue_verified"] = False
    receipt["effective_roof_state_independently_verified"] = False
    receipt["no_live_NFL_game_or_2027_outcomes_accessed"] = True
    receipt["official_model_unchanged"] = True
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    method = p.add_mutually_exclusive_group(required=True)
    method.add_argument("--source-json", type=Path)
    method.add_argument("--live-noaa-awc", action="store_true")
    p.add_argument("--venue-team", required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--attack-end", choices=["A", "B"], required=True)
    p.add_argument("--station", required=True)
    p.add_argument("--decision-utc", required=True)
    p.add_argument("--captured-utc")
    p.add_argument("--roof", choices=["outdoors", "open"], required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # IMPORTANT: validate season, team, field source and optional local
    # destination BEFORE any NOAA call or fixture read.
    try:
        resolve_stadium_geometry(
            team=args.venue_team, season=args.season,
            offense_attack_end=ATTACK_ENDS[0 if args.attack_end == "A" else 1],
        )
        utc_time(args.decision_utc)
    except (ValueError, PermissionError) as exc:
        p.error(str(exc))
    if args.out.exists() or args.out.is_symlink():
        p.error("Never overwrite an existing stadium weather receipt")
    if args.live_noaa_awc and args.captured_utc:
        p.error("Live NOAA mode measures its own response receipt time")
    if args.source_json and not args.captured_utc:
        p.error("Offline JSON requires --captured-utc")
    if args.live_noaa_awc:
        source_bytes, captured = fetch_recent_awc_metar(args.station)
    else:
        if args.source_json.is_symlink() or not args.source_json.is_file():
            p.error("Offline NOAA weather source must be an ordinary file")
        if args.source_json.stat().st_size > MAX_SOURCE_BYTES:
            p.error("Offline NOAA source exceeds single-station size cap")
        source_bytes = args.source_json.read_bytes()
        captured = args.captured_utc
    receipt = capture_venue_weather(
        venue_team=args.venue_team, season=args.season,
        attack_end=ATTACK_ENDS[0 if args.attack_end == "A" else 1],
        station=args.station, decision_utc=args.decision_utc,
        roof=args.roof, source_bytes=source_bytes,
        captured_utc=captured, fetched_from_awc_https=args.live_noaa_awc,
    )
    write_weather_receipt(args.out, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "venue": receipt["venue_geometry"]["stadium_name"],
        "venue_team": args.venue_team,
        "offense_bearing_approx_deg": receipt[
            "offense_attack_heading_true_deg"
        ],
        "station_id": receipt["station_id"],
        "station_distance_km": receipt["station_distance_km"],
        "headwind_mph_signed": receipt["headwind_mph_signed"],
        "crosswind_from_right_mph_signed": receipt[
            "crosswind_from_right_mph_signed"
        ],
        "source_downloaded_after_decision": receipt[
            "source_downloaded_after_decision"
        ],
        "stadium_axis_not_independently_surveyed": True,
        "official_model_unchanged": True,
        "output": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
