"""Exploratory early-2026 check of the 2025-selected smooth score+clock model.

The score+clock feature definition and L2 strength were fixed in the 2025
chronological benchmark (src/58). This script fits on 2025 REG Q4 only, then
checks 2026 REG weeks 1-5 against a 2025 constant. Since 2026 has already been
inspected, results are NOT a fresh holdout and must NOT be used to tune or
promote a production model. Weeks 6+ of 2026 are intentionally untouched.

Run from the repository root:
  ./.venv/bin/python src/59_evaluate_2026_smooth_kickoff.py
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kickoff_chrono_benchmark import (  # noqa: E402
    regular_season_kickoffs, smooth_features,
)
from src.kickoff_touchback_model import ordinary_q4  # noqa: E402

EARLY_2026_MAX_WEEK = 5


def fit_2025_regular_score_clock(train: pd.DataFrame):
    """Fit the frozen smooth model without reading any 2026 outcomes."""
    needed = {"season", "week", "game_id", "status", "receiver_margin", "kick_seconds"}
    if needed.difference(train.columns):
        raise ValueError(f"Missing training columns: {sorted(needed.difference(train.columns))}")
    if (train.empty or not train["season"].eq(2025).all()
            or not train["week"].between(1, 18).all()
            or not train["status"].isin(["touchback", "return"]).all()):
        raise ValueError("Training must contain 2025 REG Q4 ordinary kickoffs only.")
    y = train["status"].eq("touchback").to_numpy(dtype=int)
    if len(y) < 200 or min(int(y.sum()), int(len(y)-y.sum())) < 30:
        raise ValueError("Insufficient training observations.")
    model = LogisticRegression(C=0.25, solver="lbfgs", max_iter=1000)
    model.fit(smooth_features(train, "score_clock"), y)
    return model, float(y.mean())


def early_2026_regular_kickoffs(enriched: pd.DataFrame,
                               pbp: pd.DataFrame) -> pd.DataFrame:
    """Pair weeks 1-5 only; do not consume future 2026 games."""
    required = {"season", "game_id", "play_id", "play_type", "season_type", "week"}
    if required.difference(pbp.columns):
        raise ValueError(f"Missing PBP metadata: {sorted(required.difference(pbp.columns))}")
    if (pbp.empty or not pbp["season"].eq(2026).all()
            or enriched.empty or not enriched["season"].eq(2026).all()):
        raise ValueError("2026 evaluation needs 2026-only PBP and kickoff pairs.")
    kicks = pbp.loc[pbp["play_type"].eq("kickoff"), [
        "game_id", "play_id", "season_type", "week",
    ]].rename(columns={"play_id": "kick_play_id"})
    if kicks.duplicated(["game_id", "kick_play_id"]).any():
        raise ValueError("Duplicate kickoff identifiers.")
    joined = enriched.merge(kicks, on=["game_id", "kick_play_id"],
                            how="left", validate="one_to_one")
    eligible = ordinary_q4(joined)
    week = pd.to_numeric(eligible["week"], errors="coerce")
    sample = eligible.loc[
        eligible["season_type"].eq("REG")
        & week.between(1, EARLY_2026_MAX_WEEK)
    ].copy()
    if sample.empty:
        raise ValueError("No eligible 2026 REG Q4 kickoffs in weeks 1-5.")
    sample["week"] = pd.to_numeric(sample["week"]).astype(int)
    return sample.reset_index(drop=True)


def compare_2026(model, constant_rate: float, sample: pd.DataFrame,
                 *, bootstrap_repeats: int = 1200, seed: int = 2025) -> dict:
    """Score both frozen forecasts; resample by game to account for clustering."""
    if (sample.empty or not sample["season"].eq(2026).all()
            or not sample["week"].between(1, EARLY_2026_MAX_WEEK).all()):
        raise ValueError("Evaluation is restricted to 2026 REG weeks 1-5.")
    if not (0 < constant_rate < 1):
        raise ValueError("The constant rate must be strictly between zero and one.")
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative.")
    y = sample["status"].eq("touchback").to_numpy(dtype=int)
    predicted = model.predict_proba(smooth_features(sample, "score_clock"))[:, 1]
    constant = np.full(len(y), constant_rate, dtype=float)
    base_sqerr = (y-constant)**2
    diff = base_sqerr - (y-predicted)**2
    ids = sample["game_id"].to_numpy()
    games = pd.unique(ids)
    by_game = [np.flatnonzero(ids == g) for g in games]
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_repeats, dtype=float)
    for i in range(bootstrap_repeats):
        draws = rng.integers(0, len(games), len(games))
        idx = np.concatenate([by_game[j] for j in draws])
        boot[i] = diff[idx].mean()
    ci = (np.quantile(boot, [0.025, 0.975]) if bootstrap_repeats
          else np.array([np.nan, np.nan]))
    return {
        "n": len(y),
        "games": len(games),
        "observed_rate": float(y.mean()),
        "constant_forecast": float(constant_rate),
        "smooth_forecast_mean": float(predicted.mean()),
        "constant_brier": float(brier_score_loss(y, constant)),
        "smooth_brier": float(brier_score_loss(y, predicted)),
        "constant_log_loss": float(log_loss(y, constant, labels=[0, 1])),
        "smooth_log_loss": float(log_loss(y, predicted, labels=[0, 1])),
        "brier_improvement": float(diff.mean()),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }


def main() -> None:
    import nflreadpy as nfl

    pairer = runpy.run_path(str(ROOT / "src/55_audit_2025_kickoffs.py"))["audit_kickoff_pairs"]
    enricher = runpy.run_path(str(ROOT / "src/56_audit_2025_kickoff_context.py"))["add_receiver_margin"]

    print("\n=== FROZEN 2025 SCORE+CLOCK CANDIDATE ===")
    print("2026 is exploratory only; NO 2026 training or parameter tuning.")
    pbp25 = nfl.load_pbp([2025]).sort(["game_id", "play_id"]).to_pandas()
    enriched25 = enricher(pairer(pbp25, 2025), pbp25)
    train = regular_season_kickoffs(enriched25, pbp25)
    model, constant = fit_2025_regular_score_clock(train)
    print(f"2025 REG ordinary Q4 training: {len(train)} kickoffs, "
          f"{train['game_id'].nunique()} games")
    print(f"2025 REG constant: {constant:.2%}")
    for margin, desc in [(1, "FG, receiver leading by 1"),
                         (-3, "TD+XP, receiver trailing by 3"),
                         (-4, "TD+2, receiver trailing by 4")]:
        inputs = pd.DataFrame({"receiver_margin": [margin], "kick_seconds": [299]})
        pred = model.predict_proba(smooth_features(inputs, "score_clock"))[0, 1]
        print(f"Predicted touchback at 4:59: {desc}: {pred:.2%}")

    print("\n=== EXPLORATORY 2026 REG WEEKS 1-5 ===")
    pbp26 = nfl.load_pbp([2026]).sort(["game_id", "play_id"]).to_pandas()
    enriched26 = enricher(pairer(pbp26, 2026), pbp26)
    test = early_2026_regular_kickoffs(enriched26, pbp26)
    r = compare_2026(model, constant, test)
    print(f"2026 regular-season weeks 1-5: {r['n']} ordinary Q4 kickoffs "
          f"in {r['games']} games")
    print(f"Observed touchbacks: {r['observed_rate']:.2%}")
    print(f"Constant prediction: {r['constant_forecast']:.2%}")
    print(f"Smooth prediction avg: {r['smooth_forecast_mean']:.2%}")
    print(f"Constant Brier: {r['constant_brier']:.5f}, "
          f"log loss: {r['constant_log_loss']:.5f}")
    print(f"Smooth Brier:   {r['smooth_brier']:.5f}, "
          f"log loss: {r['smooth_log_loss']:.5f}")
    print(f"Brier improvement (+ better): {r['brier_improvement']:+.5f}")
    print(f"Game-cluster bootstrap 95% CI: [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]")
    print("\nNO production change. 2026 weeks 6+ remain unexamined by this script.")
    print("Rare kickoff outcomes, onside kicks, and penalties are excluded.")
    print("This is NOT independent validation because early 2026 was examined already.")


if __name__ == "__main__":
    main()
