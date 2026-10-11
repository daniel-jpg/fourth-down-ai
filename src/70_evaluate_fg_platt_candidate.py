"""Step 70: fixed-spec field-goal make calibration candidate.

Motivation: the existing frozen field-goal make model underpredicts made 50+
yard field goals on observed 2023-24 development and 2025 exploratory games.
We fit a single POSITIVE-SLOPE monotone Platt mapping on 2023 only and
evaluate 2024, then fit 2023+24 and evaluate 2025 separately. The mapping
operates on the existing pre-play make probability; it NEVER sees outcomes of
unattempted field goals and cannot validate fourth-down recommendations.

No hyperparameter search, no 2026 data, no production deployment.
2025 has already been examined and is NOT a pristine holdout.
Outputs: a development-only JSON and logs in ignored artifacts/.
Run through GitHub Actions rather than requiring local work.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP69 = ROOT / "src/69_audit_2025_event_models.py"
OUT = ROOT / "artifacts/2025_fg_platt_candidate_report.json"
C = 10.0  # fixed, no search/tuning based on 2025


def load_event_audit():
    spec = importlib.util.spec_from_file_location("frozen_event_audit", STEP69)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing frozen component audit.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logit_scores(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1-1e-7)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Calibrator requires one-dimensional probabilities.")
    return np.log(p/(1-p)).reshape(-1, 1)


def fit_fixed_calibrator(train: pd.DataFrame) -> dict:
    """No field-position bins, 2025 outcomes, or tuned hyperparameters."""
    if train.empty or not train["season"].between(2023, 2024).all():
        raise ValueError("Only development seasons 2023-24 can fit calibration.")
    if not train["action"].eq("FIELD_GOAL").all():
        raise ValueError("FG probability calibration can only use observed FG.")
    labels = train["actual"].to_numpy(dtype=int)
    if len(train) < 500 or len(np.unique(labels)) != 2:
        raise ValueError("Insufficient historical successes/failures.")
    p = train["predicted"].to_numpy(dtype=float)
    calibrator = LogisticRegression(C=C, max_iter=1500, random_state=70)
    calibrator.fit(logit_scores(p), labels)
    slope = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise RuntimeError("Calibrator must be strictly increasing in source WP.")
    return {
        "slope": slope, "intercept": intercept,
        "fit_years": sorted(int(v) for v in train["season"].unique()),
        "fit_observations": len(train),
        "fit_games": int(train["game_id"].nunique()),
        "fixed_C": C,
    }


def apply_fixed_calibrator(p: np.ndarray, params: dict) -> np.ndarray:
    """Monotone in existing make probability; preserve original rank order."""
    score = logit_scores(p)[:, 0]
    z = params["intercept"] + params["slope"]*score
    return np.exp(-np.logaddexp(0., -z))


def paired_game_bootstrap(
    original_squared: np.ndarray, calibrated_squared: np.ndarray,
    games: np.ndarray, *, repetitions: int = 1500, seed: int = 70,
) -> tuple[float, float]:
    base = np.asarray(original_squared, dtype=float)
    cand = np.asarray(calibrated_squared, dtype=float)
    ids = np.asarray(games).astype(str)
    if (base.shape != cand.shape or base.shape != ids.shape
            or base.ndim != 1 or len(base) == 0 or repetitions < 100
            or not np.isfinite(base).all() or not np.isfinite(cand).all()):
        raise ValueError("Need valid aligned game-level Brier losses.")
    groups, inverse = np.unique(ids, return_inverse=True)
    delta = base-cand
    totals = np.bincount(inverse, weights=delta, minlength=len(groups))
    counts = np.bincount(inverse, minlength=len(groups))
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(repetitions, len(groups)))
    sample_mean = totals[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return tuple(float(v) for v in np.quantile(sample_mean, [.025, .975]))


def paired_outcome_metrics(name: str, frame: pd.DataFrame,
                           params: dict) -> dict:
    """Paired real FG attempts only, never causal unchosen action results."""
    n = len(frame)
    out: dict = {"name": name, "n": n,
                 "games": int(frame["game_id"].nunique())}
    if not n:
        out["status"] = "NO SUPPORT"
        return out
    y = frame["actual"].to_numpy(dtype=float)
    orig = frame["predicted"].to_numpy(dtype=float)
    cand = apply_fixed_calibrator(orig, params)
    if not np.isfinite(cand).all() or not ((0 <= cand)&(cand <= 1)).all():
        raise ValueError("Calibration returned invalid probability.")
    delta = (y-orig)**2-(y-cand)**2
    ci = paired_game_bootstrap(
        (y-orig)**2, (y-cand)**2, frame["game_id"].to_numpy()
    )
    def ll(p):
        clipped = np.clip(p, 1e-9, 1-1e-9)
        return float(np.mean(-y*np.log(clipped)-(1-y)*np.log(1-clipped)))
    out.update({
        "actual_rate": float(y.mean()),
        "production_rate": float(orig.mean()),
        "candidate_rate": float(cand.mean()),
        "production_brier": float(np.mean((y-orig)**2)),
        "candidate_brier": float(np.mean((y-cand)**2)),
        "brier_improvement": float(np.mean(delta)),
        "improvement_ci95": list(ci),
        "production_logloss": ll(orig),
        "candidate_logloss": ll(cand),
        "status": ("LOW SUPPORT" if n < 30
                   else "EXPLORATORY"),
    })
    return out


def prespecified_groups(data: pd.DataFrame, params: dict) -> list[dict]:
    d = data["fg_distance"]
    late = data["qtr"].eq(4)&data["seconds"].le(300)
    masks = [
        ("ALL", np.ones(len(data), dtype=bool)),
        ("18-29 YARDS", d.between(18, 29)),
        ("30-39 YARDS", d.between(30, 39)),
        ("40-49 YARDS", d.between(40, 49)),
        ("50+ YARDS", d.ge(50)),
        ("Q4 LAST FIVE", late),
        ("Q4 LAST FIVE 18-29 YARDS", late&d.between(18, 29)),
    ]
    return [
        paired_outcome_metrics(name, data.loc[mask], params)
        for name, mask in masks
    ]


def main() -> None:
    os.chdir(ROOT)
    print("=== STEP 70: FIXED FIELD-GOAL CALIBRATION CANDIDATE ===", flush=True)
    print("Fit only 2023-24 development, check 2024 and examined 2025.")
    print("Never retrain production FG model; no 2026 data or policy promotion.")
    audit = load_event_audit()
    source = audit.validate_historical_rows(
        pd.read_parquet(audit.DATA)
    )
    data = audit.enrich_roof(source)
    engine = audit.load_engine()
    preds, excluded = audit.predict_observed_events(data, engine)
    if len(preds) < .90*len(data):
        raise RuntimeError("Unexpected observation exclusions: "+str(excluded))
    fg = preds.loc[preds["action"].eq("FIELD_GOAL")].copy()
    report: dict = {
        "purpose": "development-only FG Platt calibration candidate",
        "no_2026_used": True,
        "2025_status": "previously examined, exploratory",
        "scope": "observed chosen field goals, not policy validation",
        "promotion": "NOT AUTHORIZED",
        "frozen_production_model": "models/field_goal_outcome_model.joblib",
        "folds": [],
    }
    for train_years, validation_year in [
        ((2023,), 2024),
        ((2023, 2024), 2025),
    ]:
        fit = fg.loc[fg["season"].isin(train_years)].copy()
        evaluated = fg.loc[fg["season"].eq(validation_year)].copy()
        if fit.empty or evaluated.empty or fit["season"].max() >= validation_year:
            raise RuntimeError("Chronological train/evaluation leakage.")
        if set(fit["game_id"]) & set(evaluated["game_id"]):
            raise RuntimeError("Game-ID leakage into later test season.")
        params = fit_fixed_calibrator(fit)
        metrics = prespecified_groups(evaluated, params)
        report["folds"].append({
            "fit": params, "evaluation_year": validation_year,
            "metrics": metrics,
        })
        print(f"\nTRAIN {train_years}, TEST {validation_year}: "
              f"Platt slope {params['slope']:.4f}, "
              f"intercept {params['intercept']:+.4f}", flush=True)
        for row in metrics:
            if not row["n"]:
                print(f"  {row['name']:<28} n=0")
                continue
            ci = row["improvement_ci95"]
            print(f"  {row['name']:<28} n={row['n']:>4} "
                  f"actual={row['actual_rate']:.1%} "
                  f"orig={row['production_rate']:.1%} "
                  f"cal={row['candidate_rate']:.1%} "
                  f"Brier {row['production_brier']:.5f}"
                  f"->{row['candidate_brier']:.5f} "
                  f"gain={row['brier_improvement']:+.5f} "
                  f"95%game=[{ci[0]:+.5f},{ci[1]:+.5f}] "
                  f"{row['status']}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nDevelopment-only report: {OUT.relative_to(ROOT)}")
    print("Do not promote based on reexamined 2025. No production change.")


if __name__ == "__main__":
    main()
