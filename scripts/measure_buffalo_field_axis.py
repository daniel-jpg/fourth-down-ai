"""One-shot Buffalo 2026 field-axis candidate from claimed goal-line GPS fixes.

No live NOAA, no NFL play/outcome access, no field-axis change to stadium
registry, no official model weights. Only local source files are hashed.

Required JSON file (e.g. evidence/buffalo_goal_lines.json) mapping two
physical GOAL LINE MIDPOINT observations, one NORTH and one SOUTH:
{
  "NORTH": {
    "latitude": 42.773445, "longitude": -78.792109,
    "horizontal_uncertainty_m": 2.0,
    "source_id": "northgps01", "observer_id": "observernorth01",
    "measurement_method": "CLAIMED_FIELD_GPS",
    "source_url": "https://example.org/north",
    "collected_at_utc": "2026-10-12T18:00:00Z",
    "landmark_description": "north physical goal line midpoint",
    "artifact_file": "north_gps_raw.txt"
  },
  "SOUTH": {
    "latitude": 42.772635, "longitude": -78.792271,
    "horizontal_uncertainty_m": 2.0,
    "source_id": "southgps01", "observer_id": "observersouth01",
    "measurement_method": "CLAIMED_FIELD_GPS",
    "source_url": "https://example.org/south",
    "collected_at_utc": "2026-10-12T18:00:00Z",
    "landmark_description": "south physical goal line midpoint",
    "artifact_file": "south_gps_raw.txt"
  }
}
THESE COORDINATES ARE INVENTED TEST FIXTURES, NOT ACTUAL FIELD FIXES.
Both local artifact files must be placed adjacent to the manifest.

Then:
  PYTHONPATH=. python -m scripts.measure_buffalo_field_axis \
    --manifest-json evidence/buffalo_goal_lines.json \
    --landmark-type GOAL_LINE_MIDPOINTS_100YDS \
    --out artifacts/Buffalo_untrusted_axis_candidate.json

This DOES NOT prove true NFL end-zone coordinates, source independence,
GPS uncertainty claims, snap attack direction or verified weather.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.capture_awc_decision_weather import write_weather_receipt
from src.buffalo_2026_field_axis_measurement import (
    MAX_SOURCE_BYTES, EXPECTED_METERS, build_buffalo_field_axis_candidate,
)

MAX_MANIFEST_SIZE = 18_000


def build_local_candidate(
    *, manifest_path: Path, landmark_type: str, season: int = 2026,
    venue_team: str = "BUF",
) -> dict:
    # The geometry gate checks protected/future seasons before opening
    # the user-provided files; no 2026 football plays are fetched.
    if type(season) is not int or season != 2026:
        raise PermissionError("Only a 2026 Buffalo measurement is supported")
    if venue_team != "BUF":
        raise ValueError("Only the 2026 Buffalo stadium is supported")
    if landmark_type not in EXPECTED_METERS:
        raise ValueError("Invalid field endpoint landmark type")
    if (
        manifest_path.is_symlink() or not manifest_path.is_file()
        or not 2 <= manifest_path.stat().st_size <= MAX_MANIFEST_SIZE
    ):
        raise ValueError("Expected a small plain two-point local manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Unable to parse local GPS survey claim JSON") from exc
    if type(manifest) is not dict or set(manifest) != {"NORTH", "SOUTH"}:
        raise ValueError("Manifest must list exactly NORTH and SOUTH landmarks")
    root = manifest_path.parent.resolve()
    sources = {}
    metadata = {}
    for side in ("NORTH", "SOUTH"):
        row = manifest[side]
        if type(row) is not dict or "artifact_file" not in row:
            raise ValueError("Missing local original GPS/media artifact path")
        asset = row["artifact_file"]
        if (
            type(asset) is not str or not 1 <= len(asset) <= 160
            or Path(asset).is_absolute() or ".." in Path(asset).parts
        ):
            raise ValueError("Original source file must use a safe relative path")
        file = root / asset
        if (
            file.is_symlink() or not file.is_file()
            or not file.resolve().is_relative_to(root)
            or not 8 <= file.stat().st_size <= MAX_SOURCE_BYTES
        ):
            raise ValueError("Original coordinate source file inaccessible or too large")
        sid = row.get("source_id")
        if type(sid) is not str or sid in sources:
            raise ValueError("Each coordinate must have its own source ID")
        sources[sid] = file.read_bytes()
        metadata[side] = {
            key: val for key, val in row.items() if key != "artifact_file"
        }
    return build_buffalo_field_axis_candidate(
        season=season, venue_team=venue_team,
        landmark_type=landmark_type,
        point_metadata=metadata, original_artifacts=sources,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-json", type=Path, required=True)
    p.add_argument("--landmark-type", choices=sorted(EXPECTED_METERS), required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--venue-team", default="BUF")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or args.out.is_symlink():
        p.error("Never overwrite an earlier field-axis candidate receipt")
    candidate = build_local_candidate(
        manifest_path=args.manifest_json,
        landmark_type=args.landmark_type,
        season=args.season, venue_team=args.venue_team,
    )
    write_weather_receipt(args.out, candidate)
    print(json.dumps({
        "status": candidate["status"],
        "candidate_northward_azimuth_deg": candidate[
            "candidate_northward_azimuth_deg"
        ],
        "candidate_worst_case_angle_error_deg_if_errors_true": candidate[
            "candidate_worst_case_angle_error_deg_if_errors_true"
        ],
        "coordinate_provenance_independently_authenticated": False,
        "may_change_official_go_fg_punt_model": False,
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
