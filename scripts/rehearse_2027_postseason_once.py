"""FABRICATED 272-game 2027 REG rehearsal, with no NFL data retrieval.

Run from repository root:
  python -m scripts.rehearse_2027_postseason_once --fabricated-only \
    --outdir artifacts/fabricated_2027_postseason_once

The exclusive output directory is a one-time LOCAL dry-run lock. GitHub
Actions runners are ephemeral, so this is NOT a durable future evaluation
lock across multiple jobs or a real-2027 authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fourth_down_policy_study import POLICY_COMMIT
from src.postseason_2027_admission import (
    FAKE_PROVIDER, MODE, STANDARD, _canonical_hash,
)
from src.postseason_2027_once_only import run_fabricated_postseason_once


def fabricate_whole_season() -> dict:
    """Circle-schedule 32 invented team records, 17 games each, 18 weeks."""
    rotation = sorted(STANDARD)
    sources, independent = [], []
    serial = 0
    for round_index in range(17):
        for index in range(16):
            home, away = rotation[index], rotation[31 - index]
            if round_index % 2:
                home, away = away, home
            week = (
                round_index + 1 if round_index < 14 else
                (15 if index < 8 else 16) if round_index == 14 else
                17 if round_index == 15 else 18
            )
            serial += 1
            game_id = f"synthetic_2027_{week:02d}_{serial:03d}"
            # Invented, mathematically matching terminal game scores.
            h = 18 + serial % 23
            a = 10 + (serial * 7) % 24
            sources.append({
                "game_id": game_id, "season": 2027, "week": week,
                "season_type": "REG", "home_team": home, "away_team": away,
                "pbp_home_margin": h - a,
            })
            independent.append({
                "event_id": f"synthetic_espn_event_{serial:03d}",
                "season": 2027, "week": week, "game_type": "REG",
                "status": "final", "home_team": home, "away_team": away,
                "home_score_final": h, "away_score_final": a,
                "provider": FAKE_PROVIDER,
            })
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]

    first = sources[0]
    observation = {
        "game_id": first["game_id"], "play_id": "101",
        "season": 2027, "week": first["week"], "quarter": 4, "down": 4,
        "offense_team": first["home_team"],
        "home_team": first["home_team"],
        "away_team": first["away_team"],
        "site": "HOME", "game_seconds_remaining": 299.,
        "yardline_100": 6., "ydstogo": 4.,
        "score_differential": -4., "posteam_timeouts_remaining": 2,
        "defteam_timeouts_remaining": 3,
        "feature_asof": "pre_snap", "policy_commit": POLICY_COMMIT,
        # These are arithmetic fixtures, NOT outputs of the NFL simulator.
        "policy_action": "FIELD_GOAL",
        "eligible_actions": ["NORMAL_GO_PASS", "FIELD_GOAL"],
        "q_wp_by_action": {"NORMAL_GO_PASS": .34, "FIELD_GOAL": .37},
        "propensity_policy": .4, "mu_policy": .55,
        "close_call": True,
    }
    skipped = {
        "game_id": first["game_id"], "play_id": "202",
        "reason": "MODEL_CONTEXT_MISSING",
        "detail": "Q2_SECOND_HALF_RECEIVER_UNAVAILABLE",
    }
    false_keys = (
        "source_labels_accessed", "observed_actions_accessed",
        "final_game_results_accessed", "2026_reserved_data_accessed",
        "real_2027_outcomes_accessed", "future_season_scoring_authorized",
        "independent_pre_snap_provenance", "calibration_evaluated",
    )
    return {
        "forecast_bundle": {
            "manifest": {
                "mode": "synthetic", "source_type": "fabricated_only",
                "policy_commit": POLICY_COMMIT,
                "reconstruction_asof": "POSTGAME_RECONSTRUCTED_NOT_WITNESSED",
                "real_2027_holdout_gate": "NOT_AUTHORIZED_OR_COMPLETED",
                "actual_pre_snap_forecasts_observed": False,
                "engine_used": "SYNTHETIC_PRECOMPUTED_FIXTURE_NOT_REGISTERED_MODEL",
                "full_precision_requested": False,
            },
            "synthetic_forecasts": [observation],
            "skipped_opportunities": [skipped],
            "per_forecast_engine_audit": [],
            "audit": {
                "source_fourth_down_candidates": 2,
                "forecasted_fabricated_opportunities": 1,
                "unscored_opportunities": 1,
                "skip_reason_counts": {"MODEL_CONTEXT_MISSING": 1},
                "synthetic_prediction_sha256": _canonical_hash([observation]),
                **{flag: False for flag in false_keys},
            },
        },
        "observed_plays": [
            {
                "game_id": first["game_id"], "play_id": "101",
                "observed_action": "FIELD_GOAL", "no_play": False,
            },
            {
                "game_id": first["game_id"], "play_id": "202",
                "observed_action": "NO_PLAY", "no_play": True,
            },
        ],
        "pbp_games": sources,
        "reference_games": independent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fabricated-only", action="store_true")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.fabricated_only is not True:
        parser.error("Explicit --fabricated-only flag is required")
    result = run_fabricated_postseason_once(
        fabricate_whole_season(), args.outdir, mode=MODE,
    )
    print(json.dumps({
        "status": result["status"],
        "joined_decisions": result["joined_decisions"],
        "source_fourth_down_candidates": result["source_fourth_down_candidates"],
        "source_game_score_matches": result["source_crosscheck"]["score_matches"],
        "real_2027_evaluated": result["real_2027_evaluated"],
        "real_holdout_release_authorized": result["real_holdout_release_authorized"],
        "output_dir": str(args.outdir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
