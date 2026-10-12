"""Research-only field-position-aware normal/fake GO conversion challenger.

Uses exactly the frozen 14-feature conversion model interface as baseline,
and a SINGLE predeclared challenger with continuous yardline_100 appended.
No 2025, reserved 2026, or 2027 outcomes, no production model rewrite, and
no automatic policy promotion. This estimates conversion of historically
chosen actions, not the causal benefit of choosing GO rather than PUNT.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GO_ACTIONS = frozenset({
    "NORMAL_GO_RUN", "NORMAL_GO_PASS",
    "FAKE_PUNT_RUN", "FAKE_PUNT_PASS",
    "FAKE_FG_RUN", "FAKE_FG_PASS",
})
NORMAL = frozenset({"NORMAL_GO_RUN", "NORMAL_GO_PASS"})
# Must exactly match src/42_build_decision_engine.py:GO_FEATURES.
FROZEN_FEATURES = (
    "ydstogo", "goal_to_go", "qtr", "game_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "short_yardage", "inside_10",
    "inside_20", "final_two_minutes", "is_pass", "fake_punt", "fake_fg",
)
CHALLENGER_FEATURES = FROZEN_FEATURES + ("yardline_100",)
DATA_FEATURES = frozenset({
    "season", "split", "game_id", "action", "converted", "yardline_100",
    *FROZEN_FEATURES,
}) - {"is_pass", "fake_punt", "fake_fg"}
# Explicit frozen existing normal-run adjustment is measured separately.
RUN_SHIFT = 0.2925


def prepared_data(data: pd.DataFrame) -> pd.DataFrame:
    """Enforce chronological development only and exact action/feature scope."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Nonempty 2014-2024 development frame required.")
    if missing := DATA_FEATURES.difference(data.columns):
        raise ValueError(f"Missing GO modeling fields: {sorted(missing)}")
    f = data.loc[data["action"].isin(GO_ACTIONS)].copy().reset_index(drop=True)
    if f.empty:
        raise ValueError("No recognized historical GO decisions.")
    yr = pd.to_numeric(f["season"], errors="coerce")
    expected = np.where(yr <= 2022, "train", "validation")
    if (yr.isna().any() or not yr.between(2014, 2024).all()
            or not np.array_equal(f["split"].to_numpy(), expected)):
        raise ValueError("Only 2014-22 train and 2023-24 validation are permitted.")
    if f["game_id"].isna().any() or f["game_id"].astype(str).str.strip().eq("").any():
        raise ValueError("All games require IDs for paired cluster uncertainty.")
    target = pd.to_numeric(f["converted"], errors="coerce").to_numpy(float)
    if not np.isfinite(target).all() or not np.isin(target, [0., 1.]).all():
        raise ValueError("Conversion labels must be actual binary outcomes.")
    f["is_pass"] = f["action"].str.endswith("_PASS").astype(int)
    f["fake_punt"] = f["action"].str.startswith("FAKE_PUNT").astype(int)
    f["fake_fg"] = f["action"].str.startswith("FAKE_FG").astype(int)
    for name in CHALLENGER_FEATURES:
        raw = pd.to_numeric(f[name], errors="coerce").to_numpy(float)
        if not np.isfinite(raw).all():
            raise ValueError(f"Pre-snap feature {name} has missing/nonfinite values.")
        f[name] = raw
    if (not f["yardline_100"].between(.5, 99.5).all()
            or not f["ydstogo"].between(0, 100).all()):
        raise ValueError("Invalid field-position / yards-to-go inputs.")
    return f


def split_dev(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepared_data(data)
    train = data.loc[data["split"].eq("train")].copy()
    valid = data.loc[data["split"].eq("validation")].copy()
    if train.empty or valid.empty:
        raise ValueError("2014-22 train and 2023-24 validation both required.")
    if set(train["game_id"]) & set(valid["game_id"]):
        raise ValueError("Game leakage between training and validation.")
    return train, valid


def train_candidate(train: pd.DataFrame) -> Pipeline:
    """Same regularization/learner as frozen pooled GO, one extra feature."""
    if train.empty or not train["split"].eq("train").all():
        raise ValueError("Challenger may fit only 2014-22 training records.")
    labels = train["converted"].to_numpy(dtype=int)
    if len(set(labels)) != 2:
        raise ValueError("GO training needs both conversion outcomes.")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=.25, max_iter=3000, random_state=42)),
    ])
    model.fit(train[list(CHALLENGER_FEATURES)].to_numpy(float), labels)
    return model


def paired_game_cluster_ci(
    baseline_sq_error: np.ndarray,
    challenger_sq_error: np.ndarray,
    game_ids: Sequence, *, draws: int = 1000, seed: int = 2024,
) -> list[float] | None:
    """Whole-game percentile bootstrap for baseline-minus-challenger Brier."""
    b = np.asarray(baseline_sq_error, dtype=float)
    c = np.asarray(challenger_sq_error, dtype=float)
    g = np.asarray(game_ids).astype(str)
    if b.shape != c.shape or b.ndim != 1 or g.shape != b.shape:
        raise ValueError("Game IDs and paired loss vectors must align.")
    if (len(b) == 0 or not np.isfinite(b).all() or not np.isfinite(c).all()
            or isinstance(draws, bool) or draws < 20):
        raise ValueError("Nonempty finite losses and >=20 bootstraps required.")
    ids, inv = np.unique(g, return_inverse=True)
    if len(ids) < 2:
        return None
    diff = b-c
    sums = np.bincount(inv, weights=diff, minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    # Bounded memory even for large game counts (no draws x plays matrix).
    for i in range(draws):
        selected = rng.integers(0, len(ids), size=len(ids))
        samples[i] = sums[selected].sum() / counts[selected].sum()
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def benchmark(
    val: pd.DataFrame, old: np.ndarray, new: np.ndarray,
    *, seed: int = 2024, draws: int = 1000,
) -> dict:
    """Observed-action predictive calibration, never counterfactual wins."""
    y = val["converted"].to_numpy(float)
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)
    if y.shape != old.shape or y.shape != new.shape or not len(y):
        raise ValueError("Predictions must match a nonempty validation subset.")
    if not (np.isfinite(old).all() and np.isfinite(new).all()
            and np.isin(y, [0., 1.]).all()
            and ((old >= 0) & (old <= 1)).all()
            and ((new >= 0) & (new <= 1)).all()):
        raise ValueError("Finite [0,1] predictions and binary outcomes required.")
    old_brier = float(brier_score_loss(y, old))
    new_brier = float(brier_score_loss(y, new))
    return {
        "decisions": len(y),
        "distinct_games": int(val["game_id"].nunique()),
        "observed_conversion_rate": float(y.mean()),
        "frozen_mean_predicted": float(old.mean()),
        "challenger_mean_predicted": float(new.mean()),
        "frozen_brier": old_brier,
        "challenger_brier": new_brier,
        "brier_gain_frozen_minus_challenger": old_brier - new_brier,
        "brier_gain_95pct_game_cluster_ci": paired_game_cluster_ci(
            (y-old)**2, (y-new)**2, val["game_id"].to_numpy(),
            draws=draws, seed=seed,
        ),
        "frozen_log_loss": float(log_loss(y, np.clip(old, 1e-12, 1-1e-12), labels=[0, 1])),
        "challenger_log_loss": float(log_loss(y, np.clip(new, 1e-12, 1-1e-12), labels=[0, 1])),
        "frozen_auc": float(roc_auc_score(y, old)) if len(set(y)) == 2 else None,
        "challenger_auc": float(roc_auc_score(y, new)) if len(set(y)) == 2 else None,
    }


def support_counts(train: pd.DataFrame, val: pd.DataFrame) -> list[dict]:
    """Nested own-1..N short-yardage data support, with no smoothing."""
    result = []
    for own_max in (5, 10, 20, 35):
        threshold = 100-own_max
        for frame, label in ((train, "train_2014_2022"),
                             (val, "validation_2023_2024")):
            normal = frame.loc[frame["action"].isin(NORMAL)]
            subset = normal.loc[
                normal["yardline_100"].ge(threshold)
                & normal["ydstogo"].between(1.5, 3.5)
            ]
            result.append({
                "territory": f"OWN_1_TO_{own_max}",
                "period": label,
                "fourth_and_2_or_3_count": len(subset),
                "run_count": int(subset["action"].eq("NORMAL_GO_RUN").sum()),
                "pass_count": int(subset["action"].eq("NORMAL_GO_PASS").sum()),
                "converted_count": int(subset["converted"].sum()),
            })
    return result


def extra_go_features(state: dict, *, is_pass: bool) -> dict[str, float]:
    """Q1 own-5 4th-and-3 fixture, no outcome-based feature construction."""
    base = {
        "ydstogo": 3., "goal_to_go": 0., "qtr": 1.,
        "game_seconds_remaining": 3030., "score_differential": 0.,
        "posteam_timeouts_remaining": 3., "defteam_timeouts_remaining": 3.,
        "short_yardage": 0., "inside_10": 0., "inside_20": 0.,
        "final_two_minutes": 0., "is_pass": float(is_pass),
        "fake_punt": 0., "fake_fg": 0., "yardline_100": 95.,
    }
    base.update(state)
    return base


def adjusted_run_probability(p: float, action: str, distance: float) -> float:
    """Mirror the frozen engine's separate normal short-run logit shift."""
    if action != "NORMAL_GO_RUN" or not (0. < distance <= 3.):
        return float(p)
    p = float(np.clip(p, 1e-9, 1-1e-9))
    odds = p/(1-p)*math.exp(RUN_SHIFT)
    return float(odds/(1+odds))
