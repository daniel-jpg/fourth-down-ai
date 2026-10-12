"""Predeclared 2014-22 fit / 2023-24 exploratory constrained WP study.

Run: PYTHONPATH=. python scripts/evaluate_constrained_wp_2014_2024.py

The single fixed-spec football-monotonic WP candidate already exists in
src/wp_monotonic_candidate.py; this audit DOES NOT tune its hyperparameters,
rewrite the production engine, or read any 2025+ data/outcomes. Even
passing retrospective metrics cannot validate a causal FG-versus-GO edge.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import nflreadpy as nfl
import numpy as np
import polars as pl

from src.wp_constrained_development_eval import WP_FEATURES, compare
from src.wp_monotonic_candidate import ConstrainedRegulationWP, fit_binary_wp

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / "models/win_probability_model.joblib"
BASE_SPEC = ROOT / "data/win_probability_spec.json"
OUTDIR = ROOT / "artifacts"
OUTPUT = OUTDIR / "wp_constrained_2014_2024_development.json"
EXPECTED_BLOBS = {
    BASE_MODEL: "4cdafd1687594ff624560ad140725d9c32d72a17",
    BASE_SPEC: "857ef77f7ca0a041502ed28811fdcbf48a5d6e95",
}
TRAIN_START, TRAIN_END = 2014, 2022
EVAL_YEARS = (2023, 2024)


def git_blob_hash(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(content)}\0".encode("ascii") + content
    ).hexdigest()


def pinned_inputs() -> None:
    for path, expected in EXPECTED_BLOBS.items():
        if not path.is_file() or git_blob_hash(path) != expected:
            raise RuntimeError(f"Frozen model/spec mismatch: {path.name}")


def existing_presnap_extractor():
    """Import the published pure historical feature reconstruction, not main()."""
    p = ROOT / "src/67_train_modern_wp_candidate.py"
    spec = importlib.util.spec_from_file_location("wp_67_presnap_development", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing historical state feature code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dev_states():
    builder = existing_presnap_extractor()
    years = list(range(TRAIN_START, EVAL_YEARS[-1] + 1))
    if years != list(range(2014, 2025)):
        raise RuntimeError("Forbidden prospective data year in load list")
    print("Loading ONLY 2014-2024 REG development PBP...", flush=True)
    raw = nfl.load_pbp(years)
    if not isinstance(raw, pl.DataFrame):
        raise TypeError("Expected pre-play nflreadpy Polars data")
    if missing := (set(builder.DATA_COLUMNS) | {"location"}).difference(raw.columns):
        raise ValueError(f"Historical PBP columns missing: {sorted(missing)}")
    # Neutral designations are not evidence for a physical home/away advantage.
    data = raw.filter(
        pl.col("season_type").eq("REG")
        & pl.col("location").is_not_null()
        & pl.col("location").ne("Neutral")
    ).select(builder.DATA_COLUMNS).to_pandas()
    if not set(data["season"]).issubset(set(years)):
        raise RuntimeError("Input includes a future season")
    frames = builder.usable_regulation_states(data, list(WP_FEATURES))
    if not set(frames["season"]).issubset(set(years)):
        raise RuntimeError("Pre-snap builder includes future-season rows")
    return frames, builder


def main() -> None:
    pinned_inputs()
    if tuple(json.loads(BASE_SPEC.read_text())["features"]) != WP_FEATURES:
        raise RuntimeError("Frozen WP features differ from evaluated feature order")
    frames, builder = load_dev_states()
    train = frames.loc[frames["season"].between(TRAIN_START, TRAIN_END)]
    valid = frames.loc[frames["season"].isin(EVAL_YEARS)]
    if train.empty or valid.empty or train["season"].max() >= valid["season"].min():
        raise RuntimeError("Chronological train/eval split missing or contaminated")
    if set(train["game_id"]) & set(valid["game_id"]):
        raise RuntimeError("Historical game-ID leakage across train/eval")
    if train["game_id"].nunique() < 100 or valid["game_id"].nunique() < 100:
        raise RuntimeError("Require at least 100 distinct games in both sets")

    print(f"Training 2014-22 on {len(train):,} non-neutral REG states "
          f"from {train['game_id'].nunique():,} games.", flush=True)
    frozen = joblib.load(BASE_MODEL)
    X_train = train[list(WP_FEATURES)].to_numpy(dtype=float)
    binary = fit_binary_wp(
        X_train,
        train["final_home_margin"].to_numpy(dtype=float),
        list(WP_FEATURES),
        train["game_id"].to_numpy(),
    )
    print("Constrained candidate trained; starting fixed invariants...", flush=True)
    invariants = builder.constraint_smoke(binary, valid, list(WP_FEATURES))
    if invariants["violations"]:
        raise RuntimeError("Football invariants violated")
    candidate = ConstrainedRegulationWP(binary, frozen, list(WP_FEATURES))
    print("Scoring 2023-24 retrospectively examined development...", flush=True)
    report = compare(valid, frozen, candidate, draws=1000)
    report["frozen_wp_blob_sha"] = EXPECTED_BLOBS[BASE_MODEL]
    report["frozen_wp_feature_spec_blob_sha"] = EXPECTED_BLOBS[BASE_SPEC]
    report["train_states"] = int(len(train))
    report["train_games"] = int(train["game_id"].nunique())
    report["train_years"] = "2014-2022"
    report["validation_years"] = "2023-2024 already-examined exploratory"
    report["constraint_smoke"] = invariants
    report["future_outcomes_inspected"] = False
    report["2025_outcomes_used"] = False
    report["2026_2027_outcomes_used"] = False
    report["model_artifact_role"] = "SEPARATE RESEARCH CANDIDATE, NOT LIVE ENGINE"

    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    joblib.dump(candidate, OUTDIR / "wp_monotonic_candidate_2014_2022.joblib")
    print(f"Evaluated {len(valid):,} states from {valid['game_id'].nunique():,} "
          "games; no 2025+ outcomes.", flush=True)
    for name, row in report["group_reports"].items():
        print(f"{name}: {row}", flush=True)
    print("Q4 site mirror difference (model diagnostic, not observed causality):", flush=True)
    for name in ("Q4_ALL", "Q4_FINAL_5", "Q4_FINAL_5_OWN_10",
                 "Q4_FINAL_5_OWN_10_LEADING_4"):
        print(name+":", report["q4_site_mirror_sensitivity_not_causal"][name], flush=True)
    print("Candidate deployment gate:", report["promotion_gate"], flush=True)
    print(f"Saved unpromoted research artifact and report: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
