"""2014-25 historical-only coach propensity and outcome-reward nuisances.

Both models use ONLY source-certified pre-snap features. Historical observed
action trains e(a|X) and historical final win trains mu(a,X), but neither
observed action nor final win may be supplied to the prediction API.

No 2026/27 records, original frozen model overwrite, public UI changes,
early protected kickoff validation or automatic 2027 policy-scoring jobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.fourth_down_policy_study import ACTIONS
from src.historical_fourth_down_cohort import (
    FEATURE_COLUMNS, KEY, PRESNAP_COLUMNS,
)

ACTION_ORDER = (
    "NORMAL_GO_RUN", "NORMAL_GO_PASS", "FIELD_GOAL", "PUNT",
)
if frozenset(ACTION_ORDER) != ACTIONS:
    raise RuntimeError("Coach nuisance action set diverged from frozen 2027 policy.")

# A fixed, predeclared *subset* of state-by-action interactions improves
# conditional reward model flexibility without peeking at labels.
INTERACTIONS = (
    "ydstogo", "yardline_100", "game_seconds_remaining",
    "score_differential", "quarter",
)
LOGISTIC_C = 0.35
MAX_ITER = 3000
RANDOM_STATE = 2027
DEV_TRAIN_END = 2022
DEV_VALIDATION_YEARS = (2023, 2024)
FINAL_FIT_END = 2025


@dataclass
class HistoricalNuisances:
    propensity: Pipeline
    reward: Pipeline
    training_end_year: int
    feature_names: tuple[str, ...] = FEATURE_COLUMNS
    actions: tuple[str, ...] = ACTION_ORDER


def certified_prespnap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact schema: future-observed action, labels, and result are rejected."""
    if not isinstance(frame, pd.DataFrame) or set(frame.columns) != set(PRESNAP_COLUMNS):
        raise ValueError("Pre-snap prediction input must have EXACT feature-only schema")
    if frame.empty:
        raise ValueError("Pre-snap input may not be empty")
    if not frame["feature_asof"].eq("pre_snap").all():
        raise ValueError("Only pre_snap predictions are permitted")
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Pre-snap keys must be present and unique")
    for column in FEATURE_COLUMNS:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid pre-snap feature: {column}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Missing/nonfinite pre-snap feature: {column}")
    if not frame["site"].isin(("HOME", "AWAY", "NEUTRAL")).all():
        raise ValueError("Unknown home/away/neutral site")
    orientation = frame["site"].map({"HOME": 1, "AWAY": -1, "NEUTRAL": 0})
    if not np.array_equal(
        orientation.to_numpy(int),
        frame["site_advantage"].to_numpy(int),
    ):
        raise ValueError("Site and site_advantage disagree")
    return frame


def _x(pre: pd.DataFrame) -> np.ndarray:
    certified_prespnap_frame(pre)
    return pre[list(FEATURE_COLUMNS)].to_numpy(dtype=float)


def _action_matrix(X: np.ndarray, calls: Sequence[str]) -> np.ndarray:
    """Candidate ACTION is a hypothetical do(a), not a logged future action."""
    if len(X) != len(calls) or any(a not in ACTION_ORDER for a in calls):
        raise ValueError("Must supply one supported hypothetical action per row")
    indicator = np.zeros((len(X), len(ACTION_ORDER)), dtype=float)
    for j, a in enumerate(calls):
        indicator[j, ACTION_ORDER.index(a)] = 1.
    interact = X[:, [FEATURE_COLUMNS.index(f) for f in INTERACTIONS]]
    pairwise = (indicator[:, :, None] * interact[:, None, :]).reshape(len(X), -1)
    return np.column_stack((X, indicator, pairwise))


def _pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            C=LOGISTIC_C, max_iter=MAX_ITER, random_state=RANDOM_STATE,
        )),
    ])


def _training(frame: pd.DataFrame, end_year: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Need nonempty source-audited historical training rows")
    expected = set(PRESNAP_COLUMNS) | {"observed_action", "offense_win"}
    if set(frame.columns) != expected:
        raise ValueError("Training rows must contain only certified pre-snap plus separate action/win labels")
    presnap = certified_prespnap_frame(frame[PRESNAP_COLUMNS].copy())
    years = pd.to_numeric(frame["season"], errors="coerce")
    if (not years.between(2014, end_year).all() or
            end_year not in (DEV_TRAIN_END, FINAL_FIT_END)):
        raise ValueError("Nuisance model fits may only use 2014-22 or 2014-25 historical data")
    observed = frame["observed_action"].to_numpy()
    if not np.isin(observed, ACTION_ORDER).all():
        raise ValueError("Observed action outside four normal policy actions")
    label = pd.to_numeric(frame["offense_win"], errors="coerce").to_numpy(float)
    if not np.isin(label, [0., 1.]).all():
        raise ValueError("Final offense win labels must be binary 0/1")
    if len(set(observed)) != len(ACTION_ORDER) or len(set(label)) != 2:
        raise ValueError("Need all historical actions and both final win classes")
    return presnap, observed, label


def fit_nuisances(frame: pd.DataFrame, end_year: int) -> HistoricalNuisances:
    p, obs, outcomes = _training(frame, end_year)
    states = _x(p)
    # Propensities estimate historical coach frequencies per DECISION.
    coach = _pipeline()
    coach.fit(states, obs)
    # Reward is conditional on logged action and pre-snap state.
    # Weight every game equally to prevent frequent fourth-down decisions
    # from dominating the single shared final-game outcome.
    group_size = p["game_id"].map(p["game_id"].value_counts()).to_numpy(float)
    weights = 1.0/group_size
    reward = _pipeline()
    reward.fit(
        _action_matrix(states, obs), outcomes,
        logit__sample_weight=weights,
    )
    return HistoricalNuisances(coach, reward, end_year)


def predict_policy_nuisances(
    fitted: HistoricalNuisances, pre_snap: pd.DataFrame,
    hypothetical_actions: Sequence[str],
) -> pd.DataFrame:
    """No observed action/outcomes can be passed, even accidentally.

    Returns e(policy|X) and mu(policy,X) for the GIVEN (sealed) candidate
    actions, plus the four-action propensity distribution for diagnostics.
    This method does not look up or infer what the historical coach chose.
    """
    states = _x(pre_snap)
    requested = list(hypothetical_actions)
    if len(states) != len(requested) or any(a not in ACTION_ORDER for a in requested):
        raise ValueError("Hypothetical policy action must be a known normal action")
    p = fitted.propensity.predict_proba(states)
    class_order = list(fitted.propensity.classes_)
    if set(class_order) != set(ACTION_ORDER):
        raise RuntimeError("Incomplete fitted coach propensity action space")
    reindexed = p[:, [class_order.index(a) for a in ACTION_ORDER]]
    if (not np.isfinite(reindexed).all() or
            not np.allclose(reindexed.sum(axis=1), 1., atol=1e-8)):
        raise RuntimeError("Invalid historical coach-action probability simplex")
    idx = np.array([ACTION_ORDER.index(a) for a in requested], dtype=int)
    propensity = reindexed[np.arange(len(states)), idx]
    reward = fitted.reward.predict_proba(
        _action_matrix(states, requested)
    )
    winning_classes = list(fitted.reward.classes_)
    mu = reward[:, winning_classes.index(1)]
    if (not np.isfinite(mu).all() or
            np.any(mu < 0) or np.any(mu > 1)
            or np.any(propensity < 0) or np.any(propensity > 1)):
        raise RuntimeError("Invalid conditional policy probabilities")
    result = pre_snap[KEY].copy().reset_index(drop=True)
    result["policy_action"] = requested
    result["propensity_policy"] = propensity
    result["mu_policy"] = mu
    for i, action in enumerate(ACTION_ORDER):
        result[f"coach_probability_{action}"] = reindexed[:, i]
    return result


def development_metrics(
    fitted: HistoricalNuisances, development: pd.DataFrame,
) -> dict[str, Any]:
    """Observed factual-action calibration only; NEVER causal OPE uplift."""
    if not development["season"].isin(DEV_VALIDATION_YEARS).all():
        raise ValueError("Development diagnostics require 2023-24 only")
    pre = development[PRESNAP_COLUMNS].copy()
    y = development["offense_win"].to_numpy(int)
    obs = development["observed_action"].tolist()
    preds = predict_policy_nuisances(fitted, pre, obs)
    e = preds["propensity_policy"].to_numpy(float)
    mu = preds["mu_policy"].to_numpy(float)
    all_prob = preds[[f"coach_probability_{a}" for a in ACTION_ORDER]].to_numpy(float)
    labels = np.array([ACTION_ORDER.index(a) for a in obs])
    n_games = int(development["game_id"].nunique())
    by_action = {}
    for act in ACTION_ORDER:
        subset = np.array([a == act for a in obs])
        by_action[act] = {
            "decisions": int(subset.sum()),
            "distinct_games": int(development.loc[subset, "game_id"].nunique()),
            "coach_mean_actual_action_probability": (
                float(e[subset].mean()) if subset.any() else None
            ),
            "factual_reward_brier": (
                float(brier_score_loss(y[subset], mu[subset]))
                if subset.any() else None
            ),
            "empirical_win_rate": (
                float(y[subset].mean()) if subset.any() else None
            ),
            "factual_common_support_fraction_e_05_to_95": (
                float(np.mean((e[subset] >= .05) & (e[subset] <= .95)))
                if subset.any() else None
            ),
        }
    return {
        "note": (
            "2014-22 train -> already examined 2023-24 development; "
            "factual coach choices only; no causal unchosen-action results"
        ),
        "dev_decisions": len(development),
        "dev_distinct_games": n_games,
        "observed_coach_action_log_loss": float(
            log_loss(labels, all_prob, labels=list(range(len(ACTION_ORDER))))
        ),
        "observed_coach_action_accuracy": float(
            np.mean(np.argmax(all_prob, axis=1) == labels)
        ),
        "factual_reward_win_brier": float(brier_score_loss(y, mu)),
        "factual_reward_win_log_loss": float(log_loss(
            y, np.clip(mu, 1e-10, 1-1e-10), labels=[0, 1]
        )),
        "propensity_p05_p50_p95": [
            float(v) for v in np.quantile(e, [.05, .5, .95])
        ],
        "by_factual_action": by_action,
    }
