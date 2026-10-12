"""Synthetic-only pinned-policy -> historical-nuisance bridge.

Pre-snap forecasts never receive observed coach action or final winner.
The registered source commit is 31176d50af25db9dd42dae07e9b17cb5c2b930b4.
This does NOT authorize a future 2027 evaluation or 2026 kickoff run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.fourth_down_policy_study import ACTIONS, POLICY_COMMIT
from src.fourth_down_cohort_builder import PRE_SNAP_KEYS
from src.historical_coach_nuisances import certified_prespnap_frame
from src.historical_fourth_down_cohort import PRESNAP_COLUMNS
from src.historical_nuisance_portable import (
    EXPECTED_STATUS, load_historical_parameters,
    assert_historical_source_integrity, predict_portable_policy,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "src/42_build_decision_engine.py"
PINNED_ENGINE_BLOB = "fa69ee309221d8e25af86026956c082d1862d89a"
# All hashes are Git blobs at POLICY_COMMIT, also verified equal on main.
PINNED_ASSETS = {
    "models/fake_conversion_model.joblib": "d4f2dd5037209d263c20cf28bca50d70138245d4",
    "models/field_goal_outcome_model.joblib": "99dacab74861f4f4172b0db5b483a64721e1ad8d",
    "models/punt_field_position_model.joblib": "cc69d8ff315bef8ca6a4270dcb7262b236ca0ea2",
    "models/win_probability_model.joblib": "4cdafd1687594ff624560ad140725d9c32d72a17",
    "data/go_shared_transition_pool.parquet": "018535dd7ccf98bed0e411c6a4bb08e1a1924b5a",
    "data/punt_ordinary_transition_pool.parquet": "b06f98edfda6d28e898a04066e587131abfb5779",
    "data/punt_rare_transition_pool.parquet": "ed72e5a2d6d5a8a2cf627519c75c5e5856c9c51d",
    "data/punt_transition_spec.json": "8c07471db806978517339c1b1f11a6ef9d8853d0",
    "data/fg_made_clock_transition_pool.parquet": "ea3a0d380825b4df66207161217062f61d042e7b",
    "data/fg_block_transition_pool.parquet": "5b5f36e741dba124f35347949409d02a0854099a",
    "data/fg_live_transition_rates.parquet": "090bbd4cfb119e7ed4bcc3ace895989fe32ccff5",
    "data/win_probability_spec.json": "857ef77f7ca0a041502ed28811fdcbf48a5d6e95",
    "data/fourth_down_modeling_split.parquet": "fa1d64f187810c08edc9a113051c8b633e2e9168",
}
# Historical X intentionally excludes roof/kickoff receiver: those remain
# decision-engine context and must be sourced independently at snap time.
SNAPSHOT_KEYS = frozenset(PRESNAP_COLUMNS) | {
    "down", "offense_team", "home_team", "away_team",
    "roof", "second_half_receiver", "source_type",
}
ROOF_VALUES = frozenset({"outdoors", "open", "closed", "dome", "indoors"})
NORMAL_ORDER = ("NORMAL_GO_RUN", "NORMAL_GO_PASS", "PUNT", "FIELD_GOAL")
N_SIMULATIONS, RANDOM_SEED = 4000, 42
MC_HEURISTIC_Z = 1.96
if frozenset(NORMAL_ORDER) != ACTIONS:
    raise RuntimeError("Normal action vocabulary differs from registered 2027 policy")


def git_blob_sha1(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def check_pinned_engine(root: Path = ROOT) -> dict[str, str]:
    """Fail closed if any input changed from the pinned policy commit."""
    result = {}
    for relative, pinned in {
        "src/42_build_decision_engine.py": PINNED_ENGINE_BLOB,
        **PINNED_ASSETS,
    }.items():
        path = root / relative
        if not path.is_file() or git_blob_sha1(path) != pinned:
            raise RuntimeError(f"Frozen 2027 policy asset mismatch: {relative}")
        result[relative] = pinned
    return result


def load_verified_engine(root: Path = ROOT):
    """Import only on explicit request AFTER verifying source and artifacts."""
    check_pinned_engine(root)
    if Path.cwd().resolve() != root.resolve():
        raise RuntimeError("Run frozen engine tests from repository root")
    spec = importlib.util.spec_from_file_location(
        "registered_2027_policy_engine", str(root / "src/42_build_decision_engine.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load registered 2027 policy module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.N_SIMULATIONS != N_SIMULATIONS or module.RANDOM_SEED != RANDOM_SEED
            or module.ALLOW_FAKE_RECOMMENDATIONS is not False):
        raise RuntimeError("Policy seed, simulations, or fake action eligibility changed")
    return module


def _int(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be integer {low}..{high}")
    if not low <= int(value) <= high:
        raise ValueError(f"{label} must be integer {low}..{high}")
    return int(value)


def _num(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be finite numeric")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{label} must be finite within [{low}, {high}]")
    return number


def validate_synthetic_snapshot(raw: Mapping[str, Any]) -> pd.DataFrame:
    """Exact as-of contract; extra result/action columns cannot enter predictors."""
    if not isinstance(raw, dict) or set(raw) != SNAPSHOT_KEYS:
        raise ValueError("Exact synthetic pre-snap keys required; no outcome/action labels")
    if raw["source_type"] != "fabricated_only":
        raise ValueError("Real/prospective 2027 inference is not authorized")
    _int(raw["season"], "season", 2027, 2027)
    _int(raw["week"], "week", 1, 18)
    _int(raw["down"], "down", 4, 4)
    q = _int(raw["quarter"], "quarter", 1, 4)
    offense, home, away = (raw[k] for k in (
        "offense_team", "home_team", "away_team",
    ))
    if (not all(isinstance(x, str) and x.strip() for x in (offense, home, away))
            or home == away or offense not in (home, away)):
        raise ValueError("Offense/home/away teams must be valid and distinct")
    site = raw["site"]
    if (site not in ("HOME", "AWAY", "NEUTRAL")
            or site == "HOME" and offense != home
            or site == "AWAY" and offense != away):
        raise ValueError("Site must match actual offense-perspective home/away teams")
    if raw["roof"] not in ROOF_VALUES:
        raise ValueError("Actual stadium roof required; default outdoor is prohibited")
    receiver = raw["second_half_receiver"]
    if q == 2 and receiver not in ("OFFENSE", "DEFENSE"):
        raise ValueError("Q2 requires known second-half kickoff receiver")
    if q != 2 and receiver is not None and receiver not in ("OFFENSE", "DEFENSE"):
        raise ValueError("Invalid second-half kickoff receiver")
    clock = _num(raw["game_seconds_remaining"], "clock", 0, 3600)
    lo, hi = {1:(2700,3600), 2:(1800,2700),
              3:(900,1800), 4:(0,900)}[q]
    if not lo <= clock <= hi:
        raise ValueError("Clock incompatible with regulation quarter")
    yard = _num(raw["yardline_100"], "yardline_100", .5, 99.5)
    distance = _num(raw["ydstogo"], "ydstogo", 0, 100)
    if distance > max(1., yard) + 1e-9:
        raise ValueError("Yards to go beyond opponent goal line")
    goal = _int(raw["goal_to_go"], "goal_to_go", 0, 1)
    if bool(goal) != bool(yard <= distance + 1e-9):
        raise ValueError("goal_to_go conflicts with first-down line/end zone")
    _num(raw["score_differential"], "score_differential", -150, 150)
    for k in ("posteam_timeouts_remaining", "defteam_timeouts_remaining"):
        _int(raw[k], k, 0, 3)
    presnap = pd.DataFrame.from_records(
        [{k: raw[k] for k in PRESNAP_COLUMNS}], columns=PRESNAP_COLUMNS,
    )
    certified_prespnap_frame(presnap)
    return presnap


def engine_state(raw: Mapping[str, Any]) -> dict[str, Any]:
    validate_synthetic_snapshot(raw)
    result = {
        "qtr": int(raw["quarter"]),
        "game_seconds_remaining": float(raw["game_seconds_remaining"]),
        "yardline_100": float(raw["yardline_100"]),
        "ydstogo": float(raw["ydstogo"]),
        "goal_to_go": int(raw["goal_to_go"]),
        "score_differential": float(raw["score_differential"]),
        "posteam_timeouts_remaining": int(raw["posteam_timeouts_remaining"]),
        "defteam_timeouts_remaining": int(raw["defteam_timeouts_remaining"]),
        "site": raw["site"], "roof": raw["roof"],
    }
    if raw["second_half_receiver"] is not None:
        result["second_half_receiver"] = raw["second_half_receiver"]
    return result


def build_synthetic_policy_forecast(
    manifest: Mapping[str, Any], raw: Mapping[str, Any], *,
    engine: Any, nuisance_snapshot: Mapping[str, Any],
    n_simulations: int = N_SIMULATIONS, seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Synthetic pre-snap forecast -> existing cohort_builder input.

    All available normal eligible actions and their Q values are retained.
    The actual coach's action, result of the play, and game winner are
    explicitly not allowed in raw, manifest, or nuisance model features.
    """
    if not isinstance(manifest, dict) or manifest != {
        "mode": "synthetic",
        "source_type": "fabricated_only",
        "policy_commit": POLICY_COMMIT,
    }:
        raise ValueError("Only synthetic fabricated pinned manifest allowed")
    n = _int(n_simulations, "n_simulations", 2, N_SIMULATIONS)
    if type(seed) is not int or seed != RANDOM_SEED:
        raise ValueError("Policy seed must stay pinned to 42")
    pre = validate_synthetic_snapshot(raw)
    state = engine_state(raw)
    output = engine.recommend(state, n_simulations=n, seed=seed)
    if not isinstance(output, dict) or not isinstance(output.get("results"), pd.DataFrame):
        raise RuntimeError("Frozen engine returned no result table")
    scored = output["results"]
    if not {"action", "eligible", "expected_win_probability", "mc_se"}.issubset(scored):
        raise RuntimeError("Malformed frozen engine action matrix")
    if scored["action"].duplicated().any():
        raise RuntimeError("Duplicate frozen engine action in result matrix")
    eligible, errors = {}, {}
    normalized = engine.normalize_state(state)
    for action in NORMAL_ORDER:
        rows = scored.loc[scored["action"].eq(action)]
        if len(rows) != 1:
            raise RuntimeError(f"Missing normal policy action {action}")
        actual = rows.iloc[0]
        expected_eligible, _ = engine.action_eligible(normalized, action)
        if bool(actual["eligible"]) != bool(expected_eligible):
            raise RuntimeError(f"Engine eligibility inconsistent for {action}")
        if not expected_eligible:
            if pd.notna(actual["expected_win_probability"]):
                raise RuntimeError(f"Ineligible {action} has nonempty WP")
            continue
        eligible[action] = _num(actual["expected_win_probability"], action+" WP", 0, 1)
        errors[action] = _num(actual["mc_se"], action+" MC SE", 0, 1)
    ranked = [a for a in scored.loc[scored["eligible"], "action"] if a in eligible]
    if len(ranked) != len(eligible) or set(ranked) != set(eligible):
        raise RuntimeError("Normal policy action ranking/eligibility mismatch")
    if not ranked:
        raise RuntimeError("No eligible normal actions")
    best = ranked[0]
    if eligible[best] < max(eligible.values())-1e-12:
        raise RuntimeError("Engine leader is not max-WP among normal actions")
    close_call = False
    if len(ranked) >= 2:
        next_action = ranked[1]
        # Descriptive uncertainty heuristic; NOT a causal or UI parity claim.
        denominator = math.hypot(errors[best], errors[next_action])
        close_call = denominator > 0 and (
            eligible[best]-eligible[next_action] < MC_HEURISTIC_Z*denominator
        )
    model = predict_portable_policy(nuisance_snapshot, pre, [best])
    if len(model) != 1 or model["policy_action"].iloc[0] != best:
        raise RuntimeError("Policy and historical nuisance actions differ")
    e = _num(model["propensity_policy"].iloc[0], "e(policy|X)", 0, 1)
    mu = _num(model["mu_policy"].iloc[0], "mu(policy,X)", 0, 1)

    candidate = {
        **{k: raw[k] for k in (
            "game_id", "play_id", "season", "week", "quarter",
            "down", "offense_team", "home_team", "away_team",
            "site", "game_seconds_remaining", "yardline_100",
            "ydstogo", "score_differential",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining",
            "feature_asof",
        )},
        "policy_commit": POLICY_COMMIT, "policy_action": best,
        "eligible_actions": list(eligible),
        "q_wp_by_action": eligible,
        "propensity_policy": e, "mu_policy": mu,
        "close_call": bool(close_call),
    }
    if set(candidate) != PRE_SNAP_KEYS:
        raise RuntimeError("Cohort-builder forecast schema changed")
    return {
        "pre_snap": candidate,
        "audit": {
            "source": "SYNTHETIC ONLY; NO REAL FUTURE OUTCOMES",
            "frozen_policy_commit": POLICY_COMMIT,
            "engine_blob_sha1": PINNED_ENGINE_BLOB,
            "historical_nuisance": EXPECTED_STATUS,
            "simulation_count": n, "seed": seed,
            "full_precision": n == N_SIMULATIONS,
            "site": raw["site"], "goal_to_go": raw["goal_to_go"],
            "roof": raw["roof"],
            "second_half_receiver": raw["second_half_receiver"],
            "close_call_rule": "descriptive MC heuristic, 1.96 times combined SE",
            "observed_action_access": False,
            "outcome_access": False, "prospectively_validated": False,
        },
    }
