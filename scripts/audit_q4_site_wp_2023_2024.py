"""Inspect 2023-24 development Q4 HOME/AWAY downstream WP reliability.

Unlike future-forward Step 66/83 gates, this deliberately fetches ONLY
previously examined 2023-2024 REG PBP. It scores selected PRE-SNAP states
against final winning outcomes, reports game-cluster uncertainty, and
integrates a correctly mirrored site-only synthetic sensitivity case.

No model replacement, 2025+/reserved outcomes, or policy promotion.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl

from src.q4_site_wp_research import (
    GROUPS, WP_FEATURES, classifier_probabilities, mirror_offense_sites,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "models/win_probability_model.joblib"
SPEC = ROOT / "data/win_probability_spec.json"
OUTPUT = ROOT / "artifacts/q4_site_wp_2023_2024_development.json"
PINNED_BLOBS = {
    FROZEN: "4cdafd1687594ff624560ad140725d9c32d72a17",
    SPEC: "857ef77f7ca0a041502ed28811fdcbf48a5d6e95",
}


def check_pinned() -> None:
    for path, expected in PINNED_BLOBS.items():
        data = path.read_bytes()
        actual = hashlib.sha1(
            f"blob {len(data)}\0".encode("ascii") + data
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen model/spec changed: {path.name}")


def import_presnap_builder():
    """Reuse the existing Step 67 historical state extraction exactly."""
    path = ROOT / "src/67_train_modern_wp_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "development_only_wp_state_builder", str(path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load pinned historical WP feature builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_after_failed_go_situation(model) -> dict:
    """Frozen WP at opponent's own 6, 1st-and-10, leading 4 in Q4.

    This resembles one failed fourth-down continuation, but is NOT a
    replay of any actual fourth-down, true coaching advantage or holdout.
    """
    setup = {
        "game_seconds_remaining": 290., "half_seconds_remaining": 290.,
        "qtr": 4., "down": 1., "ydstogo": 10.,
        "yardline_100": 94., "goal_to_go": 0.,
        "home_score_differential": 4.,
        "home_timeouts_remaining": 3., "away_timeouts_remaining": 3.,
        "is_home_posteam": 1., "is_kickoff": 0., "is_overtime": 0.,
        "final_five_minutes": 1.,
    }
    X = np.asarray([[setup[f] for f in WP_FEATURES]], dtype=float)
    home, away = mirror_offense_sites(X)
    _, tie_h, home_wp = classifier_probabilities(model, home)
    away_wp, tie_a, _ = classifier_probabilities(model, away)
    return {
        "state": "Q4 4:50, opponent receiving team own 6, leading 4, first-and-10",
        "receiver_home_wp": float(home_wp[0]),
        "receiver_away_wp": float(away_wp[0]),
        "receiver_neutral_wp_by_definition": float((home_wp[0] + away_wp[0])/2),
        "tie_home": float(tie_h[0]), "tie_away": float(tie_a[0]),
        "receiver_home_minus_away_pp": float(100*(home_wp[0]-away_wp[0])),
        "qualification": "Single model WP site mirror only; no calibrated action comparison",
    }


def main() -> None:
    check_pinned()
    feature_spec = json.loads(SPEC.read_text())
    features = tuple(feature_spec["features"])
    if features != WP_FEATURES:
        raise RuntimeError("WP training feature order differs from frozen artifact.")
    state_builder = import_presnap_builder()
    # DEVELOPMENT-ONLY years. 2025 is examined elsewhere; NOT used here.
    raw = nfl.load_pbp([2023, 2024])
    if not isinstance(raw, pl.DataFrame):
        raise TypeError("Expected nflreadpy Polars pre-snap PBP frame.")
    if missing := (set(state_builder.DATA_COLUMNS) | {"location"}).difference(raw.columns):
        raise ValueError(f"Historical PBP missing {sorted(missing)}")
    # Neutral REG venues require a separate explicit no-home-advantage axis;
    # do not mislabel those games as true HOME or AWAY site samples.
    reg = raw.filter(
        pl.col("season_type").eq("REG")
        & pl.col("location").is_not_null()
        & pl.col("location").ne("Neutral")
    )
    raw_selected = reg.select(state_builder.DATA_COLUMNS).to_pandas()
    if not set(raw_selected["season"]) == {2023, 2024}:
        raise RuntimeError("Unexpected seasons in development PBP input.")
    states = state_builder.usable_regulation_states(
        raw_selected, list(features)
    )
    if not set(states["season"]) == {2023, 2024}:
        raise RuntimeError("Historical feature builder leaked a season.")
    model = joblib.load(FROZEN)
    report = summarize(states, model, draws=1000)
    report["synthetic_after_failed_go_site_case"] = fixed_after_failed_go_situation(model)
    report["frozen_model_blob"] = PINNED_BLOBS[FROZEN]
    report["source_method"] = "nflreadpy 2023/2024 REG; pre-snap Step 67 feature extractor"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")

    print("=== Q4 site-conditioned WP historical reliability (2023-24 only) ===", flush=True)
    print("Number of pre-snap states:", report["states"],
          "games:", report["games"], flush=True)
    for group in GROUPS:
        print(f"\n{group}:", flush=True)
        for site in ("all", "HOME", "AWAY"):
            record = report["summary_by_fixed_situation_and_offense_site"][group][site]
            print(f"  {site}: {record}", flush=True)
        print("  mirrored sensitivity:",
              report["site_mirror_prediction_sensitivity_not_causal"][group],
              flush=True)
    print("Fixed after-failed-GO state:",
          report["synthetic_after_failed_go_site_case"], flush=True)
    print("Report:", OUTPUT, flush=True)
    print("NO production model changes; no reserved future season use.", flush=True)


if __name__ == "__main__":
    main()
