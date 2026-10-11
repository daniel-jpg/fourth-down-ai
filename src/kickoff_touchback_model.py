"""Regularized, 2025-trained Q4 ordinary-kickoff touchback probability model.

Scope is *conditional on an ordinary kickoff* (not an onside kick, return TD,
penalty, or unmatched continuation). Never train on 2026: its first five weeks
are reserved for out-of-time evaluation. No production inference is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


FEATURES = (
    "receiver_leading_1to3",
    "receiver_leading_4plus",
    "receiver_trailing_1to3",
    "receiver_trailing_4plus",
    "last_five_minutes",
    "last_two_minutes",
    "fraction_q4_remaining",
    "trailing_4plus_in_last_five",
)


@dataclass(frozen=True)
class ContextTouchbackModel:
    estimator: LogisticRegression
    training_touchback_rate: float
    training_count: int

    def probability(self, receiver_margin: float, seconds_remaining: float) -> float:
        """Use a *post-score receiving-team* margin, not kicking-team margin."""
        features = feature_matrix(pd.DataFrame({
            "receiver_margin": [receiver_margin],
            "kick_seconds": [seconds_remaining],
        }))
        return float(self.estimator.predict_proba(features)[0, 1])


def feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Fixed, prespecified score and clock features; no holdout-derived tuning."""
    needed = {"receiver_margin", "kick_seconds"}
    if needed.difference(data.columns):
        raise ValueError(f"Missing input columns: {sorted(needed.difference(data.columns))}")
    margin = pd.to_numeric(data["receiver_margin"], errors="coerce").to_numpy(dtype=float)
    clock = pd.to_numeric(data["kick_seconds"], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(margin).all() or not np.isfinite(clock).all()
            or np.any((clock < 0) | (clock > 900))
            or np.any(np.abs(margin - np.rint(margin)) > 0.001)):
        raise ValueError("Need finite integer receiving-team margins and Q4 clock in [0,900].")

    last_five = (clock <= 300).astype(float)
    trailing_four = (margin <= -4).astype(float)
    values = {
        "receiver_leading_1to3": ((margin >= 1) & (margin <= 3)).astype(float),
        "receiver_leading_4plus": (margin >= 4).astype(float),
        "receiver_trailing_1to3": ((margin <= -1) & (margin >= -3)).astype(float),
        "receiver_trailing_4plus": trailing_four,
        "last_five_minutes": last_five,
        "last_two_minutes": (clock <= 120).astype(float),
        "fraction_q4_remaining": clock / 900.0,
        "trailing_4plus_in_last_five": trailing_four * last_five,
    }
    return pd.DataFrame(values, columns=list(FEATURES))


def ordinary_q4(pairs: pd.DataFrame) -> pd.DataFrame:
    """Drop exclusions transparently; do not treat onside attempts as returns."""
    required = {"season", "game_id", "kick_qtr", "kick_seconds", "receiver_margin", "status"}
    if required.difference(pairs.columns):
        raise ValueError(f"Missing paired-kickoff columns: {sorted(required.difference(pairs.columns))}")
    sample = pairs.loc[
        pairs["kick_qtr"].eq(4)
        & pairs["status"].isin(["touchback", "return"])
        & pd.to_numeric(pairs["receiver_margin"], errors="coerce").notna()
    ].copy()
    if not sample.empty:
        feature_matrix(sample)
    return sample.reset_index(drop=True)


def fit_2025_model(pairs: pd.DataFrame) -> ContextTouchbackModel:
    """Explicit season guard prevents accidental training on the 2026 holdout."""
    if pairs.empty or not pairs["season"].eq(2025).all():
        raise ValueError("Training input MUST contain only 2025 kickoffs; 2026 is holdout.")
    sample = ordinary_q4(pairs)
    y = sample["status"].eq("touchback").to_numpy(dtype=int)
    if len(y) < 200 or min(int(y.sum()), int(len(y) - y.sum())) < 30:
        raise ValueError("Insufficient 2025 ordinary Q4 support for touchback modeling.")
    features = feature_matrix(sample)
    # Fixed L2 regularization, chosen before touching 2026 evaluation data.
    estimator = LogisticRegression(C=0.5, solver="lbfgs", max_iter=1000)
    estimator.fit(features, y)
    return ContextTouchbackModel(estimator, float(y.mean()), len(y))


def summarize_holdout(model: ContextTouchbackModel, pairs: pd.DataFrame,
                      *, bootstrap_repeats: int = 1000, seed: int = 2026) -> dict:
    """2026-only evaluation; bootstrap is clustered by game (not individual kicks)."""
    if pairs.empty or not pairs["season"].eq(2026).all():
        raise ValueError("Out-of-time evaluation MUST contain only 2026 kickoffs.")
    sample = ordinary_q4(pairs)
    if sample.empty:
        raise ValueError("No evaluable 2026 Q4 ordinary kickoffs.")
    y = sample["status"].eq("touchback").to_numpy(dtype=int)
    pred = model.estimator.predict_proba(feature_matrix(sample))[:, 1]
    base = np.full(len(y), model.training_touchback_rate, dtype=float)
    contextual_brier = float(brier_score_loss(y, pred))
    constant_brier = float(brier_score_loss(y, base))
    improvement = constant_brier - contextual_brier
    game_ids = sample["game_id"].to_numpy()
    unique_games = pd.unique(game_ids)
    indices_by_game = [np.flatnonzero(game_ids == game) for game in unique_games]
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_repeats)
    for i in range(bootstrap_repeats):
        sampled_game_indices = rng.integers(0, len(unique_games), len(unique_games))
        indices = np.concatenate([indices_by_game[k] for k in sampled_game_indices])
        boot[i] = float(np.mean((y[indices] - base[indices])**2 - (y[indices] - pred[indices])**2))
    ci = np.quantile(boot, [0.025, 0.975]) if bootstrap_repeats else np.array([np.nan, np.nan])
    return {
        "n": len(sample), "games": len(unique_games),
        "observed_touchback_rate": float(y.mean()),
        "constant_prediction": model.training_touchback_rate,
        "contextual_prediction_mean": float(pred.mean()),
        "constant_brier": constant_brier, "contextual_brier": contextual_brier,
        "brier_improvement": improvement, "improvement_ci": (float(ci[0]), float(ci[1])),
        "constant_log_loss": float(log_loss(y, base, labels=[0, 1])),
        "contextual_log_loss": float(log_loss(y, pred, labels=[0, 1])),
        "sample": sample,
        "predictions": pred,
    }
