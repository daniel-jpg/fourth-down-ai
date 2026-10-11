"""Step 67: train/test an actually new, football-constrained regulation WP model.

This is an automated MODEL DEVELOPMENT job, not a completed production fix.
The current 14-feature model was trained through 2022 and has physically
inconsistent field-position behavior in several late-game states.

Train two fixed-spec candidates:
  - 2014-2023 REG scrimmage -> compare versus frozen production on 2024;
  - 2014-2024 REG scrimmage -> compare versus frozen production on 2025.
The 2025 comparison is exploratory because 2025 has been inspected in earlier
studies. DO NOT use it to tune hyperparameters or claim independent validation.
2026 is NEVER accessed. All 2026 Week 6+ games remain reserved.

The candidate preserves the existing 3-class output API by retaining the
frozen model's tie probability. Regulation non-kickoff WP is replaced; for
OT/kickoff-pending states the candidate delegates to the frozen model.

Output: a separate, NOT-PRODUCTION .joblib artifact and a metrics report.
Never overwrite models/win_probability_model.joblib, update app.py, or commit
generated artifacts automatically. No promotion without decision-engine and
prospective validation.

Run:
  python src/67_train_modern_wp_candidate.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wp_monotonic_candidate import (  # noqa: E402
    ConstrainedRegulationWP, fit_binary_wp, home_advantage_matrix,
)

BASE_MODEL = ROOT / "models/win_probability_model.joblib"
BASE_SPEC = ROOT / "data/win_probability_spec.json"
EXPECTED_BLOBS = {
    BASE_MODEL: "4cdafd1687594ff624560ad140725d9c32d72a17",
    BASE_SPEC: "857ef77f7ca0a041502ed28811fdcbf48a5d6e95",
}
ARTIFACT_DIR = ROOT / "artifacts"
TRAIN_START = 2014
TEST_YEAR = 2025
# Predeclared: no hyperparameter search and no 2026 reads.
REGULAR_SCRIMMAGE = {
    "run", "pass", "no_play", "field_goal", "punt", "qb_kneel", "qb_spike",
}
DATA_COLUMNS = [
    "season", "season_type", "game_id", "play_type", "qtr",
    "game_seconds_remaining", "half_seconds_remaining",
    "down", "ydstogo", "yardline_100", "goal_to_go",
    "posteam_score", "defteam_score", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "posteam", "home_team", "result",
]


def frozen_blob(path: Path) -> str:
    binary = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(binary)}\0".encode("ascii") + binary
    ).hexdigest()


def assert_frozen_inputs() -> None:
    for path, expected in EXPECTED_BLOBS.items():
        if not path.exists() or frozen_blob(path) != expected:
            raise RuntimeError(
                f"Frozen production artifact changed: {path.name}; stop."
            )


def usable_regulation_states(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build the EXACT existing engine feature contract from pre-play data."""
    missing = set(DATA_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"Missing historical PBP fields: {sorted(missing)}")
    if raw.empty or not raw["season"].between(TRAIN_START, TEST_YEAR).all():
        raise ValueError("Only 2014-2025 historical PBP is eligible.")

    quarter = pd.to_numeric(raw["qtr"], errors="coerce")
    clock = pd.to_numeric(raw["game_seconds_remaining"], errors="coerce")
    completed = pd.to_numeric(raw["result"], errors="coerce")
    valid = (
        raw["season_type"].eq("REG")
        & raw["play_type"].isin(REGULAR_SCRIMMAGE)
        & raw["posteam"].notna()
        & raw["home_team"].notna()
        & quarter.between(1, 4)
        & clock.between(0, 3600)
        & completed.notna()
        & raw["posteam_score"].notna()
        & raw["defteam_score"].notna()
        & raw["yardline_100"].notna()
    )
    frame = raw.loc[valid].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid historical REG scrimmage states.")

    def numeric(name: str) -> np.ndarray:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)

    is_home = frame["posteam"].eq(frame["home_team"]).to_numpy(bool)
    raw_margin = numeric("posteam_score") - numeric("defteam_score")
    own_tos = numeric("posteam_timeouts_remaining")
    def_tos = numeric("defteam_timeouts_remaining")
    qtr = numeric("qtr")
    secs = numeric("game_seconds_remaining")
    result = frame[["season", "game_id"]].copy()
    result["final_home_margin"] = numeric("result")
    engineered = {
        "game_seconds_remaining": secs,
        "half_seconds_remaining": numeric("half_seconds_remaining"),
        "qtr": qtr,
        "down": numeric("down"),
        "ydstogo": numeric("ydstogo"),
        "yardline_100": numeric("yardline_100"),
        "goal_to_go": numeric("goal_to_go"),
        "home_score_differential": np.where(is_home, raw_margin, -raw_margin),
        "home_timeouts_remaining": np.where(is_home, own_tos, def_tos),
        "away_timeouts_remaining": np.where(is_home, def_tos, own_tos),
        "is_home_posteam": is_home.astype(float),
        "is_kickoff": np.zeros(len(frame)),
        "is_overtime": np.zeros(len(frame)),
        "final_five_minutes": ((qtr == 4) & (secs <= 300)).astype(float),
    }
    if set(engineered) != set(features):
        raise RuntimeError(
            "Engine feature spec differs from engineered features. "
            f"Missing={sorted(set(features)-set(engineered))}; "
            f"unexpected={sorted(set(engineered)-set(features))}"
        )
    for name in features:
        result[name] = engineered[name]
    if not np.isfinite(result["home_score_differential"].to_numpy(float)).all():
        raise RuntimeError("Malformed score feature in historical input.")
    return result


def load_seasons() -> pd.DataFrame:
    parts = []
    for year in range(TRAIN_START, TEST_YEAR + 1):
        print(f"Loading {year} regular-season play-by-play...", flush=True)
        raw = nfl.load_pbp([year])
        if missing := set(DATA_COLUMNS).difference(raw.columns):
            raise ValueError(f"Season {year} missing columns: {sorted(missing)}")
        parts.append(raw.select(DATA_COLUMNS).to_pandas())
    return pd.concat(parts, ignore_index=True)


def cluster_brier_ci(
    deltas: np.ndarray, game_ids: np.ndarray, *,
    seed: int = 67, n_boot: int = 1200,
) -> tuple[float, float]:
    """Confidence interval for baseline - candidate Brier, whole-game draws."""
    d = np.asarray(deltas, dtype=float)
    games = np.asarray(game_ids).astype(str)
    if d.ndim != 1 or d.shape != games.shape or len(d) == 0:
        raise ValueError("Brier improvements and game labels must align.")
    if not np.isfinite(d).all() or n_boot < 100:
        raise ValueError("Finite Brier deltas and >=100 replicates required.")
    ids, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(ids))
    ns = np.bincount(inv, minlength=len(ids))
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(ids), size=(n_boot, len(ids)))
    means = sums[draws].sum(axis=1) / ns[draws].sum(axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def measure(
    title: str, data: pd.DataFrame, original: np.ndarray,
    alternative: np.ndarray,
) -> dict:
    if data.empty:
        return {"name": title, "n": 0}
    y = (data["final_home_margin"].to_numpy(float) > 0).astype(float)
    p = np.asarray(original, dtype=float)
    q = np.asarray(alternative, dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite predicted probabilities.")
    base_brier = float(np.mean((y-p)**2))
    new_brier = float(np.mean((y-q)**2))
    deltas = (y-p)**2 - (y-q)**2
    lo, hi = cluster_brier_ci(deltas, data["game_id"].to_numpy())
    result = {
        "name": title, "n": len(data), "games": data["game_id"].nunique(),
        "base_brier": base_brier,
        "candidate_brier": new_brier,
        "brier_improvement": base_brier-new_brier,
        "95ci": [lo, hi],
        "actual_home_win": float(y.mean()),
        "baseline_mean_wp": float(p.mean()),
        "candidate_mean_wp": float(q.mean()),
    }
    print(
        f"{title}: n={result['n']} games={result['games']} "
        f"baseline Brier={base_brier:.5f} candidate Brier={new_brier:.5f} "
        f"improvement={base_brier-new_brier:+.5f} "
        f"95% game CI=[{lo:+.5f},{hi:+.5f}]"
    )
    return result


def compare_year(
    year: int, data: pd.DataFrame, binary_model, frozen, features: list[str],
) -> list[dict]:
    selected = data.loc[data["season"].eq(year)].copy()
    if selected.empty:
        raise ValueError(f"Missing evaluation season {year}.")
    candidate = ConstrainedRegulationWP(binary_model, frozen, features)
    X = selected[features].to_numpy(dtype=float)
    freeze_ix = {int(v):j for j,v in enumerate(frozen.classes_)}
    base = frozen.predict_proba(X)[:, freeze_ix[2]]
    cand = candidate.predict_proba(X)[:, 2]

    def subset(label: str, mask: np.ndarray):
        return measure(f"{year} {label}", selected.loc[mask],
                       base[mask], cand[mask])

    home_offense = selected["is_home_posteam"].to_numpy(bool)
    quarter = selected["qtr"].to_numpy(float)
    secs = selected["game_seconds_remaining"].to_numpy(float)
    home_margin = selected["home_score_differential"].to_numpy(float)
    results = [
        subset("ALL SCRIMMAGE", np.ones(len(selected), dtype=bool)),
        subset("Q4", quarter == 4),
        subset("Q4 LAST FIVE", (quarter == 4) & (secs <= 300)),
        subset("HOME POSSESSION", home_offense),
        subset("AWAY POSSESSION", ~home_offense),
        subset("LAST FIVE HOME POSSESSION", (quarter == 4) & (secs <= 300) & home_offense),
        subset("LAST FIVE AWAY POSSESSION", (quarter == 4) & (secs <= 300) & ~home_offense),
        subset("LAST FIVE HOME TRAILING 1-3",
               (quarter == 4) & (secs <= 300)
               & (home_margin >= -3) & (home_margin <= -1)),
    ]
    return results


def constraint_smoke(binary_model, states: pd.DataFrame,
                     features: list[str]) -> dict:
    """Prove score and field-position direction in the binary candidate."""
    template = states.loc[
        states["qtr"].eq(4)
        & states["game_seconds_remaining"].between(60, 300)
        & states["ydstogo"].eq(10)
        & states["down"].eq(1)
    ].head(15)
    if template.empty:
        template = states.head(15)
    if template.empty:
        raise ValueError("Empty sample for constraints.")
    n = 0
    for is_home in [0., 1.]:
        for _, row in template.iterrows():
            rows = pd.DataFrame([row.copy() for _ in range(11)])
            rows["is_home_posteam"] = is_home
            # Receiving team at own 20...70, fixed state / score / timeouts.
            home_adv = np.arange(20., 75., 5.)
            rows["yardline_100"] = (
                100-home_adv if is_home else home_adv
            )
            mat = home_advantage_matrix(rows[features].to_numpy(float), features)
            p = binary_model.predict_proba(mat)[:, 1]
            if np.any(np.diff(p) < -1e-9):
                raise AssertionError(
                    "Constrained model violates home field-advantage monotonicity."
                )
            n += 1
            rows = pd.DataFrame([row.copy() for _ in range(11)])
            rows["home_score_differential"] = np.arange(-5., 6.)
            mat = home_advantage_matrix(rows[features].to_numpy(float), features)
            p = binary_model.predict_proba(mat)[:, 1]
            if np.any(np.diff(p) < -1e-9):
                raise AssertionError(
                    "Constrained model violates home score-margin monotonicity."
                )
            n += 1
    return {"monotonic_line_tests": n, "violations": 0}


def main() -> None:
    os.chdir(ROOT)
    print("\n=== STEP 67: MODERN CONSTRAINED REGULATION WP CANDIDATE ===")
    print("NO 2026 data; no tuning from 2025; no production changes.")
    assert_frozen_inputs()
    frozen = joblib.load(BASE_MODEL)
    features = list(json.loads(BASE_SPEC.read_text())["features"])
    raw = load_seasons()
    data = usable_regulation_states(raw, features)
    del raw
    print(f"Total valid 2014-2025 regulation states: {len(data):,}")

    reports = []
    for train_end, valid_year in ((2023, 2024), (2024, 2025)):
        train = data.loc[data["season"].between(2014, train_end)]
        valid = data.loc[data["season"].eq(valid_year)]
        if train.empty or valid.empty or train["season"].max() >= valid["season"].min():
            raise RuntimeError("Time leakage in candidate training split.")
        if set(train["game_id"]) & set(valid["game_id"]):
            raise RuntimeError("Game leakage in candidate training split.")
        print(f"\nTraining on 2014-{train_end}, validation {valid_year}: "
              f"{len(train)} training rows, {len(valid)} test rows.",
              flush=True)
        X = train[features].to_numpy(dtype=float)
        model = fit_binary_wp(
            X,
            train["final_home_margin"].to_numpy(float),
            features, train["game_id"].to_numpy(),
        )
        invariants = constraint_smoke(model, valid, features)
        metrics = compare_year(valid_year, data, model, frozen, features)
        reports.append({
            "training_years": f"2014-{train_end}",
            "validation_year": valid_year,
            "metrics": metrics,
            "invariants": invariants,
        })
        if valid_year == 2025:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            candidate = ConstrainedRegulationWP(model, frozen, features)
            # Artifact is only in GitHub Actions downloadable artifacts.
            # NEVER committed and NEVER used by Streamlit automatically.
            joblib.dump(candidate, ARTIFACT_DIR / "wp_regulation_candidate.joblib")

    report_path = ARTIFACT_DIR / "wp_candidate_development_report.json"
    report_path.write_text(json.dumps({
        "candidate": "score+field monotonic constrained regulation",
        "baseline_train": "2014-2022",
        "2025_status": "exploratory; previously analyzed, not pristine",
        "2026_status": "unopened",
        "promotion": "NOT APPROVED; decision-engine and independent validation required",
        "folds": reports,
    }, indent=2))
    print(f"\nWrote candidate artifacts to {ARTIFACT_DIR}.")
    print("NO production promotion. No 2026 data accessed.")


if __name__ == "__main__":
    main()
