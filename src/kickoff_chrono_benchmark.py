"""Predeclared chronological 2025 benchmark for ordinary Q4 kickoff touchbacks.

No 2026 data is loaded or used for fitting, benchmarking, or model selection.
Only REG weeks 1-18 are eligible. No result is promoted into production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# Chronological expanding-window validation; no game belongs to two holdouts.
# (last training week, first validation week, last validation week)
FOLDS = ((8, 9, 12), (12, 13, 15), (15, 16, 18))
CANDIDATES = ("constant", "score", "clock", "score_clock")


def regular_season_kickoffs(enriched: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Attach kickoff week, filter to 2025 regular season, keep exclusions out.

    The enriched data comes from the existing causal kickoff pairing and
    receiving-team scoreboard audits. No play after the kickoff supplies any
    predictive feature; only kickoff score and remaining clock are used.
    """
    from src.kickoff_touchback_model import ordinary_q4

    required = {"game_id", "play_id", "play_type", "week", "season_type", "season"}
    if missing := required.difference(pbp.columns):
        raise ValueError(f"Missing kickoff metadata columns: {sorted(missing)}")
    if pbp.empty or not pbp["season"].eq(2025).all():
        raise ValueError("Benchmark PBP must contain only 2025; 2026 is excluded.")
    if enriched.empty or not enriched["season"].eq(2025).all():
        raise ValueError("Benchmark pairs must contain only 2025; 2026 is excluded.")

    kicks = pbp.loc[pbp["play_type"].eq("kickoff"), [
        "game_id", "play_id", "week", "season_type",
    ]].rename(columns={"play_id": "kick_play_id"})
    if kicks.duplicated(["game_id", "kick_play_id"]).any():
        raise ValueError("Duplicate kickoff identifiers in PBP.")

    joined = enriched.merge(kicks, on=["game_id", "kick_play_id"],
                            how="left", validate="one_to_one")
    sample = ordinary_q4(joined)
    weeks = pd.to_numeric(sample["week"], errors="coerce")
    ordinary = sample.loc[
        sample["season_type"].eq("REG") & weeks.between(1, 18)
    ].copy()
    ordinary["week"] = pd.to_numeric(ordinary["week"]).astype(int)
    if ordinary.empty:
        raise ValueError("No 2025 regular-season ordinary Q4 kickoffs.")
    if ordinary["game_id"].isna().any():
        raise ValueError("Missing game IDs, cannot ensure chronological separation.")
    return ordinary.reset_index(drop=True)


def smooth_features(sample: pd.DataFrame, candidate: str) -> np.ndarray:
    """Continuous, scaled score/clock features, without -3/-4 discontinuity.

    Fixed preprocessing and fixed L2 strength are defined before reviewing
    chronological validation scores; these choices are NOT tuned using 2026.
    """
    if candidate not in CANDIDATES or candidate == "constant":
        raise ValueError("smooth_features expects score, clock, or score_clock")
    margin = pd.to_numeric(sample["receiver_margin"], errors="coerce").to_numpy(float)
    seconds = pd.to_numeric(sample["kick_seconds"], errors="coerce").to_numpy(float)
    if (not np.isfinite(margin).all() or not np.isfinite(seconds).all()
            or np.any((seconds < 0) | (seconds > 900))):
        raise ValueError("Need finite receiver score differential and Q4 clock in [0, 900].")
    score = np.clip(margin, -14.0, 14.0) / 7.0
    magnitude = np.abs(score)
    clock = seconds / 900.0
    if candidate == "score":
        return np.column_stack((score, magnitude))
    if candidate == "clock":
        return clock.reshape(-1, 1)
    return np.column_stack((score, magnitude, clock))


def candidate_predictions(train: pd.DataFrame, valid: pd.DataFrame,
                          candidate: str) -> np.ndarray:
    """Train ONLY before the validation weeks, with explicit season guards."""
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate!r}")
    if train.empty or valid.empty:
        raise ValueError("Training and validation must both be nonempty.")
    if (not train["season"].eq(2025).all() or not valid["season"].eq(2025).all()):
        raise ValueError("Only 2025 development observations are allowed.")
    if int(train["week"].max()) >= int(valid["week"].min()):
        raise ValueError("Chronological leakage: training week overlaps validation.")
    if set(train["game_id"]) & set(valid["game_id"]):
        raise ValueError("Game leakage across training and validation.")
    y = train["status"].eq("touchback").to_numpy(dtype=int)
    if len(y) < 100 or min(int(y.sum()), int(len(y) - y.sum())) < 10:
        raise ValueError("Insufficient train sample for regularized logistic benchmark.")
    if candidate == "constant":
        return np.full(len(valid), float(y.mean()), dtype=float)
    fit = LogisticRegression(C=0.25, solver="lbfgs", max_iter=1000)
    fit.fit(smooth_features(train, candidate), y)
    return fit.predict_proba(smooth_features(valid, candidate))[:, 1]


def benchmark(sample: pd.DataFrame) -> pd.DataFrame:
    """Return strictly out-of-sample per-kickoff predictions for each candidate."""
    required = {"season", "game_id", "week", "status", "kick_seconds", "receiver_margin"}
    if missing := required.difference(sample.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if sample.empty or not sample["season"].eq(2025).all():
        raise ValueError("Benchmark is restricted to 2025, not 2026.")
    if not sample["week"].between(1, 18).all():
        raise ValueError("Only weeks 1-18 of 2025 regular season may enter benchmark.")
    if not sample["status"].isin(["touchback", "return"]).all():
        raise ValueError("Onside, penalties, and other unmatched states excluded.")
    if sample.groupby("game_id")["week"].nunique().gt(1).any():
        raise ValueError("Game IDs must occur in exactly one week.")

    batches = []
    for train_end, val_start, val_end in FOLDS:
        train = sample.loc[sample["week"].le(train_end)]
        valid = sample.loc[sample["week"].between(val_start, val_end)]
        if valid.empty:
            raise ValueError(f"No validation data in weeks {val_start}-{val_end}.")
        scored = valid[["season", "game_id", "week", "status"]].copy()
        scored["train_end"] = train_end
        scored["val_block"] = f"{val_start}-{val_end}"
        for candidate in CANDIDATES:
            scored[candidate] = candidate_predictions(train, valid, candidate)
        batches.append(scored)
    result = pd.concat(batches, ignore_index=True)
    if not result["week"].gt(result["train_end"]).all():
        raise AssertionError("Validation contains training weeks.")
    return result


def summarize_predictions(predictions: pd.DataFrame,
                          bootstrap_repeats: int = 1000, seed: int = 2025) -> pd.DataFrame:
    """Game-clustered Brier improvements against fold-specific constant rates."""
    if bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be nonnegative")
    y = predictions["status"].eq("touchback").to_numpy(int)
    game_ids = predictions["game_id"].to_numpy()
    games = pd.unique(game_ids)
    by_game = [np.flatnonzero(game_ids == game) for game in games]
    rng = np.random.default_rng(seed)
    # All candidates get identical bootstrap resamples (paired comparison).
    index_sets = [np.concatenate([by_game[i] for i in
        rng.integers(0, len(games), size=len(games))])
        for _ in range(bootstrap_repeats)]
    baseline = predictions["constant"].to_numpy(float)
    baseline_err = (y - baseline)**2
    rows = []
    for name in CANDIDATES:
        p = np.clip(predictions[name].to_numpy(float), 1e-8, 1 - 1e-8)
        delta = baseline_err - (y-p)**2
        boot = np.array([delta[idx].mean() for idx in index_sets])
        ci = (np.quantile(boot, [0.025, 0.975]) if bootstrap_repeats
              else [float("nan"), float("nan")])
        rows.append({
            "candidate": name,
            "n": len(y), "games": len(games),
            "predicted_rate": float(p.mean()),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "improvement_vs_constant": float(delta.mean()),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]),
        })
    return pd.DataFrame(rows)
