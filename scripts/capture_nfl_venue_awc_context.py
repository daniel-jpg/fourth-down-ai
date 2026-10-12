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
    --game-date-et YYYY-MM-DD --opponent-team OPPONENT_CODE \
    --station KBFI --decision-utc <recent-UTC-date-time> --roof outdoors \
    --out artifacts/station_context.json

In BOTH modes the community-derived field axis and the physical roof
remain UNVERIFIED; manual team selection never establishes an NFL game.
In LIVE mode, a 2026 schedule date and opponent are mandatory to reject
known NFL international games before HTTP I/O. A nonmatching fixture
does NOT certify that the actual game used its normal domestic venue.
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
from src.nfl_2026_international_venue_guard import (
    guard_team_home_venue_for_game,
)


def capture_venue_weather(
    *, venue_team: str, season: int, attack_end: str, station: str,
    decision_utc: str, roof: str, source_bytes: bytes,
    captured_utc: str | int | float,
    fetched_from_awc_https: bool,
    game_date_et: str | None = None,
    opponent_team: str | None = None,
) -> dict:
    """No network, filesystem writes, game outcomes or model feature use."""
    if (game_date_et is None) != (opponent_team is None):
        raise ValueError("Game date and opponent must be supplied together")
    if fetched_from_awc_https and game_date_et is None:
        raise ValueError(
            "Live NOAA stadium research requires a dated matchup so "
            "known 2026 international sites cannot use domestic geometry"
        )
    provenance = None
    if game_date_et is not None:
        provenance = guard_team_home_venue_for_game(
            game_date_et=game_date_et,
            venue_team=venue_team,
            opponent_team=opponent_team,
        )
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
    receipt["scheduled_game_venue_preflight"] = provenance
    receipt["scheduled_international_venue_mismatch_checked"] = (
        provenance is not None
    )
    receipt["other_neutral_or_relocated_sites_not_excluded"] = True
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
    p.add_argument("--game-date-et", help="2026 NFL schedule date in Eastern Time")
    p.add_argument("--opponent-team", help="Other team, for rejecting known international venues")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--attack-end", choices=["A", "B"], required=True)
    p.add_argument("--station", required=True)
    p.add_argument("--decision-utc", required=True)
    p.add_argument("--captured-utc")
    p.add_argument("--roof", choices=["outdoors", "open"], required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # IMPORTANT: validate season, team, known game-to-venue exception
    # and destination BEFORE any NOAA network I/O or source file reads.
    if bool(args.game_date_et) != bool(args.opponent_team):
        p.error("--game-date-et and --opponent-team must be provided together")
    if args.live_noaa_awc and not args.game_date_et:
        p.error("Live venue weather lookup REQUIRES --game-date-et and --opponent-team")
    try:
        if args.game_date_et is not None:
            guard_team_home_venue_for_game(
                game_date_et=args.game_date_et,
                venue_team=args.venue_team,
                opponent_team=args.opponent_team,
            )
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
        game_date_et=args.game_date_et, opponent_team=args.opponent_team,
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
