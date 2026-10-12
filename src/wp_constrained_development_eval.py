"""Research-only out-of-time evaluation of a constrained regulation WP challenger.

Compares a SINGLE fixed, already-specified monotonic challenger with the
unchanged frozen 3-class WP model. Labels are final HOME winners; the input
features are all pre-snap state. This module does not fetch data or fit
models. All scenarios and sample-size thresholds are fixed before evaluation.

2014-22 training, 2023-24 development checks are NOT an untouched future
test. A prospective model-replacement claim remains explicitly blocked.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numpy as np
import pandas as pd

from src.q4_site_wp_research import WP_FEATURES, mirror_offense_sites

GROUPS = (
    "REGULATION_ALL",
    "Q1_TO_Q3",
    "Q4_ALL",
    "Q4_FINAL_5",
    "Q4_FINAL_5_HOME",
    "Q4_FINAL_5_AWAY",
    "Q4_FINAL_5_OWN_20",
    "Q4_FINAL_5_OWN_10",
    "Q4_FINAL_5_OWN_10_LEADING_1_TO_8",
    "Q4_FINAL_5_OWN_10_LEADING_4",
    "Q4_FINAL_15_OWN_20_LEADING_1_TO_8",
    "REGULATION_2023",
    "REGULATION_2024",
)
MIN_STABLE_GAMES = 100
SEED = 2024
DRAWS = 1000


def _finite_proba(values: np.ndarray, label: str, n: int) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all():
        raise ValueError(f"{label} must be a finite 1D prediction per state")
    if (p < 0).any() or (p > 1).any():
        raise ValueError(f"{label} must stay within [0,1]")
    return p


def validate_states(states: pd.DataFrame) -> pd.DataFrame:
    """Never infer as-of time, final outcomes or missing team coordinates."""
    if not isinstance(states, pd.DataFrame) or states.empty:
        raise ValueError("Need pre-snap 2023-2024 development states.")
    required = set(WP_FEATURES) | {
        "season", "game_id", "final_home_margin"
    }
    if missing := required.difference(states.columns):
        raise ValueError(f"Missing required WP fields: {sorted(missing)}")
    if not states["season"].isin([2023, 2024]).all():
        raise ValueError("Only already-examined 2023-24 development allowed")
    if not states["qtr"].between(1, 4).all() or not states["is_kickoff"].eq(0).all():
        raise ValueError("Regulation scrimmage states only")
    if states["game_id"].isna().any() or states["game_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Game IDs must be present for cluster resampling")
    final_margin = pd.to_numeric(states["final_home_margin"], errors="coerce").to_numpy(float)
    if not np.isfinite(final_margin).all():
        raise ValueError("Final home win labels must be finite")
    home_offense = states["is_home_posteam"].to_numpy(float)
    if not np.isin(home_offense, (0., 1.)).all():
        raise ValueError("Unknown HOME/AWAY possession site")
    for col in ("yardline_100", "game_seconds_remaining",
                "home_score_differential", "home_timeouts_remaining",
                "away_timeouts_remaining"):
        if not np.isfinite(states[col].to_numpy(float)).all():
            raise ValueError(f"Missing/nonfinite pre-snap {col}")
    if not states["yardline_100"].between(.5, 99.5).all():
        raise ValueError("Invalid field position")
    return states.reset_index(drop=True)


def fixed_groups(states: pd.DataFrame) -> dict[str, np.ndarray]:
    qtr = states["qtr"].to_numpy(float)
    secs = states["game_seconds_remaining"].to_numpy(float)
    yard = states["yardline_100"].to_numpy(float)
    home = states["is_home_posteam"].to_numpy(float) == 1
    score = states["home_score_differential"].to_numpy(float)
    offense_margin = np.where(home, score, -score)
    q4 = qtr == 4
    five = q4 & (secs >= 45) & (secs <= 300)
    fifteen = q4 & (secs >= 45) & (secs <= 900)
    own10 = yard >= 90
    own20 = yard >= 80
    leading = (offense_margin >= 1) & (offense_margin <= 8)
    groups = {
        "REGULATION_ALL": np.ones(len(states), dtype=bool),
        "Q1_TO_Q3": ~q4,
        "Q4_ALL": q4,
        "Q4_FINAL_5": five,
        "Q4_FINAL_5_HOME": five & home,
        "Q4_FINAL_5_AWAY": five & ~home,
        "Q4_FINAL_5_OWN_20": five & own20,
        "Q4_FINAL_5_OWN_10": five & own10,
        "Q4_FINAL_5_OWN_10_LEADING_1_TO_8": five & own10 & leading,
        "Q4_FINAL_5_OWN_10_LEADING_4": five & own10 & (offense_margin == 4),
        "Q4_FINAL_15_OWN_20_LEADING_1_TO_8": fifteen & own20 & leading,
        "REGULATION_2023": states["season"].to_numpy(int) == 2023,
        "REGULATION_2024": states["season"].to_numpy(int) == 2024,
    }
    if set(groups) != set(GROUPS):
        raise RuntimeError("Prespecified WP groups changed; do not select post hoc.")
    return groups


def paired_game_cluster_ci(
    per_state_gain: Sequence[float], games: Sequence,
    *, draws: int = DRAWS, seed: int = SEED,
) -> list[float] | None:
    delta = np.asarray(per_state_gain, dtype=float)
    gid = np.asarray(games).astype(str)
    if delta.ndim != 1 or delta.shape != gid.shape or not len(delta):
        raise ValueError("Paired game losses must align and be nonempty")
    if not np.isfinite(delta).all() or isinstance(draws, bool) or draws < 20:
        raise ValueError("Finite loss gain and >=20 draws required")
    ids, inv = np.unique(gid, return_inverse=True)
    if len(ids) < 2:
        return None
    total = np.bincount(inv, weights=delta, minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids))
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        picks = rng.integers(len(ids), size=len(ids))
        values[i] = total[picks].sum() / counts[picks].sum()
    return [float(n) for n in np.quantile(values, [.025, .975])]


def group_report(
    y: np.ndarray, original: np.ndarray, candidate: np.ndarray,
    game_ids: np.ndarray, *, draws: int = DRAWS, seed: int = SEED,
) -> dict:
    n = len(y)
    games = int(len(set(map(str, game_ids))))
    if not n:
        return {"states": 0, "games": 0, "status": "NO_SUPPORT"}
    p = _finite_proba(original, "frozen", n)
    q = _finite_proba(candidate, "challenger", n)
    outcome = np.asarray(y, dtype=float)
    if outcome.shape != p.shape or not np.isin(outcome, (0, 1)).all():
        raise ValueError("Final home win labels must be binary")
    base_brier = float(np.mean((outcome-p)**2))
    new_brier = float(np.mean((outcome-q)**2))
    gain = (outcome-p)**2 - (outcome-q)**2
    return {
        "states": n,
        "games": games,
        "observed_home_win": float(outcome.mean()),
        "frozen_mean_home_wp": float(p.mean()),
        "challenger_mean_home_wp": float(q.mean()),
        "frozen_brier": base_brier,
        "challenger_brier": new_brier,
        "brier_gain_frozen_minus_challenger": float(base_brier-new_brier),
        "brier_gain_95pct_game_cluster_ci": paired_game_cluster_ci(
            gain, game_ids, draws=draws, seed=seed
        ),
        "status": (
            "SPARSE_DESCRIPTIVE_ONLY" if games < MIN_STABLE_GAMES
            else "EXPLORATORY_2023_24_ALREADY_EXAMINED"
        ),
    }


def site_mirror_summary(states: pd.DataFrame, frozen_model, candidate_model):
    """Measure model sensitivity ONLY. Site counterfactuals have no observed labels."""
    X = states[list(WP_FEATURES)].to_numpy(float)
    hx, ax = mirror_offense_sites(X)
    def model_gap(model):
        home_idx = {int(v):i for i, v in enumerate(model.classes_)}
        if 0 not in home_idx or 2 not in home_idx:
            raise ValueError("Expected 3-class home/away win model")
        ph = np.asarray(model.predict_proba(hx), dtype=float)[:, home_idx[2]]
        pa = np.asarray(model.predict_proba(ax), dtype=float)[:, home_idx[0]]
        if ph.shape != (len(X),) or pa.shape != (len(X),):
            raise ValueError("Invalid WP site-mirror shape")
        if not np.isfinite(ph).all() or not np.isfinite(pa).all():
            raise ValueError("Invalid WP site-mirror probability")
        return 100.*(ph-pa)
    mask = fixed_groups(states)
    p = model_gap(frozen_model)
    q = model_gap(candidate_model)
    return {
        label: {
            "states": int(selection.sum()),
            "frozen_home_minus_away_pp_mean": (
                float(np.mean(p[selection])) if selection.any() else None
            ),
            "candidate_home_minus_away_pp_mean": (
                float(np.mean(q[selection])) if selection.any() else None
            ),
            "candidate_minus_frozen_site_gap_pp_mean": (
                float(np.mean(q[selection]-p[selection])) if selection.any() else None
            ),
        }
        for label, selection in mask.items()
    }


def compare(
    states: pd.DataFrame, frozen_model, candidate_model,
    *, draws: int = DRAWS,
) -> dict:
    states = validate_states(states)
    X = states[list(WP_FEATURES)].to_numpy(float)
    def home_win_probability(model) -> np.ndarray:
        classes = {int(v):i for i, v in enumerate(model.classes_)}
        if 2 not in classes or 0 not in classes:
            raise ValueError("Expected home-win/away-win classifier classes")
        prob = np.asarray(model.predict_proba(X), dtype=float)
        if prob.ndim != 2 or prob.shape != (len(X), len(classes)):
            raise ValueError("Wrong WP class probability shape")
        if (not np.isfinite(prob).all() or (prob < 0).any()
                or (prob > 1).any()
                or not np.allclose(prob.sum(axis=1), 1., atol=1e-6)):
            raise ValueError("WP output must be valid multiclass probabilities")
        return prob[:, classes[2]]
    p = home_win_probability(frozen_model)
    q = home_win_probability(candidate_model)
    y = (states["final_home_margin"].to_numpy(float)>0).astype(float)
    gid = states["game_id"].astype(str).to_numpy()
    groups = fixed_groups(states)
    reports = {}
    for name in GROUPS:
        keep = groups[name]
        reports[name] = group_report(
            y[keep], p[keep], q[keep], gid[keep], draws=draws
        )
    # Never count synthetic mirrored states as factual test observations.
    mirror = site_mirror_summary(states.loc[groups["Q4_ALL"]], frozen_model, candidate_model)
    q4 = reports["Q4_ALL"]
    all_reg = reports["REGULATION_ALL"]
    gate = {
        "prospectively_validated": False,
        "decision_action_validated": False,
        "may_replace_frozen_model": False,
        "reasons": [
            "2023-24 already examined as development; no pristine holdout",
            "Factual state-win calibration does not prove unchosen GO/FG value",
            "No independent future Q4 site-balanced validation or action-level evidence",
            "Kickoff and overtime states still delegate to frozen model",
        ],
        "development_all_reg_brier_gain": all_reg["brier_gain_frozen_minus_challenger"],
        "development_q4_brier_gain": q4["brier_gain_frozen_minus_challenger"],
    }
    return {
        "study": "constrained_wp_challenger_chronological_development",
        "training": "2014-22 only",
        "evaluation": "2023-24 exploratory chronological development",
        "frozen_model_replaced": False,
        "n_states": len(states),
        "n_distinct_games": states["game_id"].nunique(),
        "bootstrap_draws": draws,
        "group_reports": reports,
        "q4_site_mirror_sensitivity_not_causal": mirror,
        "promotion_gate": gate,
    }
