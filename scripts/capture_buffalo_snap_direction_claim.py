"""Record user-supplied BUF 2026 fourth-down ATTACK-END witness evidence.

NO NFL play data is fetched or evaluated, no model weights/policy changed.
The native locally supplied photo/video/note evidence is hashed, never
embedded in the audit receipt or uploaded to the repository.

Example with one or two local independent observation files:
  PYTHONPATH=. python -m scripts.capture_buffalo_snap_direction_claim \
    --season 2026 --opponent-team NE --offense-team BUF \
    --game-date-et 2026-10-12 --play-key sampleplay001 \
    --quarter 4 --game-clock 04:59 \
    --decision-utc 2026-10-12T20:30:00Z \
    --witness-manifest-json evidence/witness_metadata.json \
    --out artifacts/buf_sample_snap_direction_research.json

All times/team/play identifiers above are deliberately illustrative.
The source files must already exist in the directory containing the
manifest. They remain on disk; only SHA256 evidence hashes are exported.
For source records, see src/nfl_2026_snap_direction_evidence.py.

Even two witnesses do NOT prove the field's true compass azimuth, an
independently available live weather forecast or any football outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.nfl_2026_snap_direction_evidence import (
    MAX_EVIDENCE_BYTES, assess_2026_buffalo_snap_direction,
)
from scripts.capture_awc_decision_weather import write_weather_receipt

MAX_MANIFEST_BYTES = 18_000


def assemble_direction_record(
    *,
    season: int, venue_team: str, opponent_team: str,
    offense_team: str, game_date_et: str, play_key: str, quarter: int,
    game_clock_mmss: str, decision_at_utc: str, witness_manifest: Path,
) -> dict:
    if witness_manifest.is_symlink() or not witness_manifest.is_file():
        raise ValueError("Expected a normal local witness manifest JSON file")
    if not 1 <= witness_manifest.stat().st_size <= MAX_MANIFEST_BYTES:
        raise ValueError("Witness manifest exceeds bounded data size")
    try:
        raw = json.loads(witness_manifest.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Invalid local witness JSON") from exc
    if not isinstance(raw, list) or not 1 <= len(raw) <= 2:
        raise ValueError("One or two local witness observations are required")
    sources = {}
    rows = []
    parent = witness_manifest.parent.resolve()
    for witness in raw:
        if not isinstance(witness, dict) or set(witness) != {
            "artifact_file", "source_id", "witness_id", "source_type",
            "artifact_observed_at_utc", "artifact_available_at_utc",
            "attack_end_claim", "location_landmark_observed", "source_url",
        }:
            raise ValueError("Malformed local witness manifest entry")
        artifact_path = witness["artifact_file"]
        if (
            type(artifact_path) is not str or not 1 <= len(artifact_path) <= 128
            or artifact_path.startswith("/") or ".." in Path(artifact_path).parts
        ):
            raise ValueError("Original evidence must use a local relative file")
        file = parent / artifact_path
        if (
            file.is_symlink() or not file.is_file()
            or not file.resolve().is_relative_to(parent)
        ):
            raise ValueError("Original evidence must be a safe ordinary local file")
        if not 8 <= file.stat().st_size <= MAX_EVIDENCE_BYTES:
            raise ValueError("Original evidence media size outside bounds")
        sid = witness["source_id"]
        if sid in sources:
            raise ValueError("Duplicate physical source file identity")
        sources[sid] = file.read_bytes()
        rows.append({
            key: val for key, val in witness.items()
            if key != "artifact_file"
        })
    return assess_2026_buffalo_snap_direction(
        season=season, venue_team=venue_team,
        opponent_team=opponent_team, offense_team=offense_team,
        game_date_et=game_date_et, play_key=play_key,
        quarter=quarter, game_clock_mmss=game_clock_mmss,
        decision_at_utc=decision_at_utc,
        witness_metadata=rows, original_artifacts=sources,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--venue-team", default="BUF")
    parser.add_argument("--opponent-team", required=True)
    parser.add_argument("--offense-team", required=True)
    parser.add_argument("--game-date-et", required=True)
    parser.add_argument("--play-key", required=True)
    parser.add_argument("--quarter", type=int, required=True)
    parser.add_argument("--game-clock", required=True)
    parser.add_argument("--decision-utc", required=True)
    parser.add_argument("--witness-manifest-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Prevent accidental overwrite before reading any witness artifacts.
    if args.out.exists() or args.out.is_symlink():
        parser.error("Never overwrite a prior directional evidence receipt")
    record = assemble_direction_record(
        season=args.season,
        venue_team=args.venue_team,
        opponent_team=args.opponent_team,
        offense_team=args.offense_team,
        game_date_et=args.game_date_et,
        play_key=args.play_key,
        quarter=args.quarter,
        game_clock_mmss=args.game_clock,
        decision_at_utc=args.decision_utc,
        witness_manifest=args.witness_manifest_json,
    )
    write_weather_receipt(args.out, record)
    print(json.dumps({
        "status": record["status"],
        "claimed_attack_end": record["claimed_attack_end"],
        "claim_consistency": record["claim_consistency"],
        "retrospective_source_present": record["retrospective_source_present"],
        "numeric_true_north_attack_azimuth_degrees": None,
        "changes_official_predictions": False,
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
