"""Read-only, portable historical nuisance predictions from text-only parameters.

The JSON is unapproved, research-only 2014-25 historical data. This module
does not load arbitrary pickle/joblib bytes, download game data, rerun training,
execute the frozen decision engine, or alter public recommendations.

Portable numerical logits allow the EXACT archived fitted coefficients to
survive GitHub Actions' temporary artifact retention and offer another
auditable representation for later model-provenance review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.fourth_down_policy_study import POLICY_COMMIT
from src.historical_coach_nuisances import (
    ACTION_ORDER, INTERACTIONS, _action_matrix, _x,
)
from src.historical_fourth_down_cohort import FEATURE_COLUMNS, KEY

PARAMETER_FILE = Path(__file__).resolve().parents[1] / "data/historical_nuisance_2014_2025_development_parameters.json"
EXPECTED_STATUS = "RESEARCH_ONLY_UNAPPROVED_2014_2025_NUISANCES"


def load_historical_parameters(path: Path = PARAMETER_FILE) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text())
    if (not isinstance(obj, dict) or obj.get("schema_version") != 1
            or obj.get("status") != EXPECTED_STATUS
            or obj.get("frozen_policy_commit") != POLICY_COMMIT
            or tuple(obj.get("feature_names", [])) != FEATURE_COLUMNS
            or tuple(obj.get("reward_action_interactions", [])) != INTERACTIONS
            or tuple(obj.get("hypothetical_action_order", [])) != ACTION_ORDER):
        raise ValueError("Incompatible or promoted nuisance parameter snapshot")
    if not isinstance(obj.get("models"), dict) or set(obj["models"]) != {
        "propensity", "conditional_win_reward",
    }:
        raise ValueError("Require two separate historical nuisance parameter sets")
    return obj


def assert_historical_source_integrity(snapshot: Mapping[str, Any]) -> None:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "extractor_sha256": root / "src/historical_fourth_down_cohort.py",
        "trainer_sha256": root / "src/historical_coach_nuisances.py",
        "source_sha256": root / "data/fourth_downs_labeled.parquet",
    }
    for field, path in paths.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != snapshot.get(field):
            raise RuntimeError(f"Historical nuisance provenance mismatch: {field}")


def _proba(params: Mapping[str, Any], X: np.ndarray) -> tuple[list[str], np.ndarray]:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Predictors must be finite 2D pre-snap values")
    mean = np.asarray(params["scaler_mean"], dtype=float)
    scale = np.asarray(params["scaler_scale"], dtype=float)
    coef = np.asarray(params["coef"], dtype=float)
    intercept = np.asarray(params["intercept"], dtype=float)
    classes = params["classes"]
    if (mean.shape != (matrix.shape[1],)
            or scale.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()
            or coef.ndim != 2 or coef.shape[1] != matrix.shape[1]
            or intercept.shape != (coef.shape[0],)
            or not np.isfinite(coef).all() or not np.isfinite(intercept).all()
            or not isinstance(classes, list) or len(classes) < 2
            or len(set(classes)) != len(classes)):
        raise ValueError("Corrupt standardization or logistic parameters")
    logits = ((matrix-mean)/scale) @ coef.T + intercept
    if len(classes) == 2 and logits.shape[1] == 1:
        score = np.clip(logits[:, 0], -700, 700)
        p = 1.0/(1.0+np.exp(-score))
        output = np.column_stack((1-p, p))
    elif logits.shape[1] == len(classes):
        centered = logits-logits.max(axis=1, keepdims=True)
        values = np.exp(centered)
        output = values/values.sum(axis=1, keepdims=True)
    else:
        raise ValueError("Logistic class count and coefficient shape disagree")
    return list(classes), output


def predict_portable_policy(
    snapshot: Mapping[str, Any], pre_snap: pd.DataFrame,
    hypothetical_actions: Sequence[str],
) -> pd.DataFrame:
    if (snapshot.get("schema_version") != 1
            or snapshot.get("status") != EXPECTED_STATUS
            or snapshot.get("frozen_policy_commit") != POLICY_COMMIT
            or tuple(snapshot.get("feature_names", [])) != FEATURE_COLUMNS
            or tuple(snapshot.get("reward_action_interactions", [])) != INTERACTIONS
            or tuple(snapshot.get("hypothetical_action_order", [])) != ACTION_ORDER):
        raise ValueError("Incompatible unapproved nuisance snapshot")
    X = _x(pre_snap)
    actions = list(hypothetical_actions)
    if len(actions) != len(X) or any(a not in ACTION_ORDER for a in actions):
        raise ValueError("Unknown hypothetical policy action")
    classes, propensity = _proba(snapshot["models"]["propensity"], X)
    if set(classes) != set(ACTION_ORDER):
        raise ValueError("Coach probability classes differ from four normal actions")
    aligned = propensity[:, [classes.index(a) for a in ACTION_ORDER]]
    reward_classes, conditional = _proba(
        snapshot["models"]["conditional_win_reward"],
        _action_matrix(X, actions),
    )
    if set(reward_classes) != {"0.0", "1.0"}:
        raise ValueError("Reward must predict binary eventual win 0.0/1.0")
    idx = [ACTION_ORDER.index(a) for a in actions]
    chosen = aligned[np.arange(len(X)), idx]
    mu = conditional[:, reward_classes.index("1.0")]
    out = pre_snap[KEY].copy().reset_index(drop=True)
    out["policy_action"] = actions
    out["propensity_policy"] = chosen
    out["mu_policy"] = mu
    for i, a in enumerate(ACTION_ORDER):
        out[f"coach_probability_{a}"] = aligned[:, i]
    return out
