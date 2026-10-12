"""Descriptive, read-only fourth-quarter HOME/AWAY WP support diagnostics.

Input must contain 2023-2024 development-period REG regulation scrimmage
state features and final game margins. Outcome is used ONLY for scoring,
never to build state features or counterfactual site-mirror inputs.

No 2025+ data, no future-study contamination, no classifier updates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WP_FEATURES = (
    "game_seconds_remaining", "half_seconds_remaining", "qtr",
    "down", "ydstogo", "yardline_100", "goal_to_go",
    "home_score_differential", "home_timeouts_remaining",
    "away_timeouts_remaining", "is_home_posteam", "is_kickoff",
    "is_overtime", "final_five_minutes",
)
GROUPS = (
    "Q4_ALL", "Q4_FINAL_5", "Q4_FINAL_15",
    "Q4_FINAL_5_OWN_10", "Q4_FINAL_15_OWN_10",
    "Q4_FINAL_5_OWN_20", "Q4_FINAL_15_OWN_20",
    "Q4_FINAL_5_OWN_10_LEADING_1_TO_8",
    "Q4_FINAL_15_OWN_20_LEADING_1_TO_8",
    "Q4_FINAL_5_OWN_10_LEADING_4",
)


def check_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(WP_FEATURES) | {"season", "game_id", "final_home_margin"}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Nonempty development-only WP dataframe required.")
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing WP calibration columns: {sorted(missing)}")
    if (not frame["season"].isin([2023, 2024]).all()
            or not frame["qtr"].between(1, 4).all()):
        raise ValueError("Only 2023-2024 regulation development is authorized.")
    if frame["game_id"].isna().any():
        raise ValueError("Game IDs required for game-cluster uncertainty.")
    site = frame["is_home_posteam"].to_numpy(float)
    kick = frame["is_kickoff"].to_numpy(float)
    if (not np.isin(site, [0, 1]).all()
            or not np.isin(kick, [0]).all()):
        raise ValueError("Requires non-kickoff states and known real game-site axis.")
    for key in ("home_score_differential", "home_timeouts_remaining",
                "away_timeouts_remaining", "game_seconds_remaining",
                "yardline_100", "final_home_margin"):
        n = frame[key].to_numpy(float)
        if not np.isfinite(n).all():
            raise ValueError(f"Missing or nonfinite {key}.")
    if not frame["yardline_100"].between(.5, 99.5).all():
        raise ValueError("Invalid receiver/offense field position.")
    return frame.reset_index(drop=True)


def classifier_probabilities(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(model.predict_proba(X), dtype=float)
    names = np.asarray(model.classes_).astype(int)
    ix = {int(c): k for k, c in enumerate(names)}
    if 0 not in ix or 2 not in ix:
        raise ValueError("Frozen WP must provide away-win and home-win classes.")
    if probs.ndim != 2 or probs.shape != (len(X), len(names)):
        raise ValueError("WP model returned incorrect probability dimensions.")
    if (not np.isfinite(probs).all() or np.any(probs < 0)
            or np.any(probs > 1)
            or not np.allclose(probs.sum(axis=1), 1., atol=1e-6)):
        raise ValueError("WP output must contain valid class probabilities.")
    return (
        probs[:, ix[0]],
        probs[:, ix[1]] if 1 in ix else np.zeros(len(X)),
        probs[:, ix[2]],
    )


def mirror_offense_sites(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror all home-axis score, timeouts and possession, NOT just a flag.

    Each pair describes the exact same offense's yardline/score/timeouts,
    once if that offense is HOME and once if it is AWAY. This is a
    sensitivity diagnostic on generated states, NOT an observed causal
    change in venue. Never use final outcomes here.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(WP_FEATURES):
        raise ValueError("WP matrix must follow frozen 14-feature contract.")
    idx = {f: k for k, f in enumerate(WP_FEATURES)}
    p = arr[:, idx["is_home_posteam"]]
    if not np.isin(p, [0, 1]).all():
        raise ValueError("Need actual HOME or AWAY possession for a mirror.")
    home = arr.copy()
    away = arr.copy()
    home_margin = arr[:, idx["home_score_differential"]]
    offense_margin = np.where(p == 1., home_margin, -home_margin)
    offense_to = np.where(
        p == 1., arr[:, idx["home_timeouts_remaining"]],
        arr[:, idx["away_timeouts_remaining"]],
    )
    defense_to = np.where(
        p == 1., arr[:, idx["away_timeouts_remaining"]],
        arr[:, idx["home_timeouts_remaining"]],
    )
    home[:, idx["is_home_posteam"]] = 1.
    away[:, idx["is_home_posteam"]] = 0.
    home[:, idx["home_score_differential"]] = offense_margin
    away[:, idx["home_score_differential"]] = -offense_margin
    home[:, idx["home_timeouts_remaining"]] = offense_to
    home[:, idx["away_timeouts_remaining"]] = defense_to
    away[:, idx["home_timeouts_remaining"]] = defense_to
    away[:, idx["away_timeouts_remaining"]] = offense_to
    return home, away


def gap_game_cluster_ci(
    gaps: np.ndarray, games: np.ndarray, *, draws: int = 1000, seed: int = 2024,
) -> list[float] | None:
    e = np.asarray(gaps, dtype=float)
    ids = np.asarray(games).astype(str)
    if e.ndim != 1 or e.shape != ids.shape or not len(e) or not np.isfinite(e).all():
        raise ValueError("Finite gaps and same-length game identifiers required.")
    if isinstance(draws, bool) or draws < 20:
        raise ValueError("At least 20 whole-game resamples required.")
    g, inverse = np.unique(ids, return_inverse=True)
    if len(g) < 2:
        return None
    sums = np.bincount(inverse, weights=e)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        draw = rng.integers(len(g), size=len(g))
        values[i] = sums[draw].sum()/counts[draw].sum()
    return [float(z) for z in np.quantile(values, [.025, .975])]


def fixed_situation_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    qtr = frame["qtr"].to_numpy(float)
    seconds = frame["game_seconds_remaining"].to_numpy(float)
    yard = frame["yardline_100"].to_numpy(float)
    pos = frame["is_home_posteam"].to_numpy(float)
    margin = frame["home_score_differential"].to_numpy(float)
    offense_margin = np.where(pos == 1., margin, -margin)
    q4 = qtr == 4.
    five = q4 & (seconds >= 45) & (seconds <= 300)
    fifteen = q4 & (seconds >= 45) & (seconds <= 900)
    own10 = yard >= 90
    own20 = yard >= 80
    small_lead = (offense_margin >= 1) & (offense_margin <= 8)
    return {
        "Q4_ALL": q4,
        "Q4_FINAL_5": five,
        "Q4_FINAL_15": fifteen,
        "Q4_FINAL_5_OWN_10": five & own10,
        "Q4_FINAL_15_OWN_10": fifteen & own10,
        "Q4_FINAL_5_OWN_20": five & own20,
        "Q4_FINAL_15_OWN_20": fifteen & own20,
        "Q4_FINAL_5_OWN_10_LEADING_1_TO_8": five & own10 & small_lead,
        "Q4_FINAL_15_OWN_20_LEADING_1_TO_8": fifteen & own20 & small_lead,
        "Q4_FINAL_5_OWN_10_LEADING_4": five & own10 & (offense_margin == 4),
    }


def summarize(frame: pd.DataFrame, model, *, draws: int = 1000) -> dict:
    """Report factual calibration with game clustering, plus mirrored sensitivity."""
    frame = check_data(frame)
    X = frame[list(WP_FEATURES)].to_numpy(float)
    away, tied, home = classifier_probabilities(model, X)
    # Binary observed offense-win calibration excludes tied final games.
    original_home = frame["is_home_posteam"].to_numpy(float) == 1.
    actual_margin = frame["final_home_margin"].to_numpy(float)
    settled = actual_margin != 0
    obs = np.where(original_home, actual_margin > 0, actual_margin < 0)
    forecast = np.where(original_home, home, away)
    game_ids = frame["game_id"].to_numpy()
    sites = np.where(original_home, "HOME", "AWAY")
    masks = fixed_situation_masks(frame)
    if set(masks) != set(GROUPS):
        raise AssertionError("Prespecified situation coverage changed.")

    groups = {}
    for name in GROUPS:
        mask = masks[name]
        data = {"all": mask, "HOME": mask & (sites == "HOME"),
                "AWAY": mask & (sites == "AWAY")}
        groups[name] = {}
        for tag, m in data.items():
            supported = m & settled
            size = int(supported.sum())
            games = len(set(game_ids[supported]))
            if size:
                values = forecast[supported]
                truth = obs[supported].astype(float)
                residual = values - truth
                groups[name][tag] = {
                    "states": size, "games": games,
                    "mean_predicted_offense_wp": float(values.mean()),
                    "observed_offense_win_rate": float(truth.mean()),
                    "brier": float(np.mean((truth-values)**2)),
                    "predicted_minus_observed_wp": float(residual.mean()),
                    "gap_95pct_game_cluster_ci": gap_game_cluster_ci(
                        residual, game_ids[supported], draws=draws
                    ),
                    "status": (
                        "SPARSE_DESCRIPTIVE_ONLY" if games < 20
                        else "EXPLORATORY_2023_24_DEVELOPMENT"
                    ),
                }
            else:
                groups[name][tag] = {
                    "states": 0, "games": 0, "status": "NO_OBSERVED_SUPPORT"
                }
    # A paired hypothetical site flip is NOT real-world site calibration.
    # Compute it on first-snap contexts, not on labels, and separately.
    homeX, awayX = mirror_offense_sites(X)
    _, _, home_if_home = classifier_probabilities(model, homeX)
    away_if_away, _, _ = classifier_probabilities(model, awayX)
    site_diff = 100. * (home_if_home - away_if_away)
    sensitivity = {}
    for name in GROUPS:
        selected = site_diff[masks[name]]
        sensitivity[name] = {
            "states": len(selected),
            "home_minus_away_mirrored_pp_mean": (
                float(selected.mean()) if len(selected) else None
            ),
            "home_minus_away_mirrored_pp_p10_p50_p90": (
                [float(z) for z in np.quantile(selected, [.1, .5, .9])]
                if len(selected) else None
            ),
        }
    return {
        "status": "DESCRIPTIVE DEVELOPMENT ONLY; NO WP MODEL PROMOTION",
        "source": "2023-2024 REG non-neutral regulation scrimmage states",
        "2025_outcomes": "NOT READ",
        "2026_2027_outcomes": "NOT READ",
        "states": len(frame),
        "games": frame["game_id"].nunique(),
        "final_tied_states_excluded_from_binary_scores": int((~settled).sum()),
        "summary_by_fixed_situation_and_offense_site": groups,
        "site_mirror_prediction_sensitivity_not_causal": sensitivity,
        "caveats": (
            "Repeated play states in a game share the same final outcome. "
            "Game-cluster bootstraps do not remove coach-selection bias or "
            "prove that an unchosen fourth-down action was optimal. "
            "Own-10 late leading states may be extremely sparse."
        ),
    }
