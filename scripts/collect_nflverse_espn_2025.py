"""Existing NFLverse + ESPN source evidence interface, 2025 HISTORICAL ONLY.

By default no network; reads original 2014-2025 archives already in repo:
  PYTHONPATH=. python -m scripts.collect_nflverse_espn_2025 \
    --historical-offline-only --output artifacts/nflverse_espn_2025.json

OPTIONAL manual 2025-only ESPN raw-response refresh:
  PYTHONPATH=. python -m scripts.collect_nflverse_espn_2025 \
    --historical-espn-online-only --output artifacts/new_espn_2025.json

Both modes REFUSE 2026/2027 and never run model/policy evaluations.
All real 2027 source downloads, scoreboard collection and release locks
remain disabled until a separate human-reviewed post-2027 authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.nflverse_espn_historical_collection import (
    HISTORICAL_YEAR, MODE, save_historical_2025_evidence,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    op = p.add_mutually_exclusive_group(required=True)
    op.add_argument("--historical-offline-only", action="store_true")
    op.add_argument("--historical-espn-online-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        p.error("Refusing to overwrite an existing source-evidence artifact")
    if args.historical_offline_only:
        result = save_historical_2025_evidence(
            args.output, year=HISTORICAL_YEAR, mode=MODE,
        )
        print(json.dumps({
            "status": result["status"],
            "verified_historical_games": result["reg_game_count"],
            "external_score_archive_games": result[
                "archived_independent_scores"
            ]["reference_games_verified"],
            "actual_2027_outcomes_accessed": False,
            "output": str(args.output),
        }, sort_keys=True))
    else:
        # This reuses the EXISTING, historical-only, already-reviewed ESPN
        # collector, which saves all raw weekly response hashes and the
        # independently parsed final-score sidecar in a separate artifact.
        from scripts.audit_independent_espn_historical_scores import run
        result = run([HISTORICAL_YEAR], args.output)
        if result["status"] != "ALL_MATCHED":
            raise RuntimeError("Historical ESPN scoreboard crosscheck failed")


if __name__ == "__main__":
    main()
