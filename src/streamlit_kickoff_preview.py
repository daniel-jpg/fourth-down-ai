"""Experimental Streamlit comparison for 2025+ kickoff rules.

NEVER modify Streamlit's globally cached production engine: opt-in
scenario evaluation loads a distinct Python module namespace for EACH
request. This avoids concurrent Streamlit sessions leaking a temporary
monkeypatch into other users' official recommendations.

This is only an ordinary-Q4-kickoff sensitivity analysis. The 2024
development donor return distribution is used with a user-assumed
touchback rate and 2025+ touchback placement (own 35/20).
Onside kickoffs, return touchdowns and penalties are omitted.
No 2025/2026 parameter tuning or win-probability model retraining.

Do NOT treat preview rankings as validated production advice.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .kickoff_scenario import use_q4_kickoff_scenario
from .kickoff_2025_empirical import (
    FROZEN_PATH as OBSERVED_2025_PATH,
    load_frozen_pool,
    use_frozen_2025_kickoffs,
)
from .kickoff_2025_onside_sensitivity import (
    load_onside_verified,
    use_q4_onside_what_if,
    validate_what_if,
)


ROOT = Path(__file__).resolve().parents[1]
DONOR_PATH = ROOT / "data/kickoff_2024_dev_transition_pool.parquet"
ENGINE_PATH = ROOT / "src/42_build_decision_engine.py"
OBSERVED_2025_COVERAGE_PATH = ROOT / "data/kickoff_2025_q4_coverage.json"
EXPERIMENTAL_SIMULATIONS = 2400
EXPERIMENTAL_ACTION_SEED = 42
EXPERIMENTAL_KICKOFF_SEED = 2026

LABELS = {
    "NORMAL_GO_PASS": "PASS / DROPBACK",
    "NORMAL_GO_RUN": "RUN",
    "FIELD_GOAL": "FIELD GOAL",
    "PUNT": "PUNT",
}


def isolated_engine_for_preview(engine_path: Path = ENGINE_PATH) -> dict:
    """Load a private engine globals namespace, never the cached live engine."""
    if not engine_path.is_file():
        raise FileNotFoundError(f"Missing experimental engine: {engine_path}")
    spec = importlib.util.spec_from_file_location(
        "fourth_down_isolated_kickoff_preview", str(engine_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create a private engine module.")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    if not callable(getattr(module, "recommend", None)):
        raise RuntimeError("Private experimental engine lacks recommend().")
    return vars(module)


def preview_recommendation(
    state: Mapping, *,
    touchback_rate: float = 0.30,
    donor_path: Path = DONOR_PATH,
    simulations: int = EXPERIMENTAL_SIMULATIONS,
    action_seed: int = EXPERIMENTAL_ACTION_SEED,
) -> dict:
    """Evaluate a single scenario using PRIVATE, temporary engine globals.

    Only regulation Q4 is currently supported. Returns a standalone
    result; production caches and model files are untouched.
    """
    if int(state.get("qtr", 0)) != 4:
        raise ValueError("Experimental kickoff continuation is Q4 only.")
    if not np.isfinite(touchback_rate) or not 0 <= touchback_rate <= 1:
        raise ValueError("Touchback rate must be between zero and one.")
    if simulations < 200:
        raise ValueError("At least 200 simulations are needed for the preview.")
    if not donor_path.is_file():
        raise FileNotFoundError(f"Frozen 2024 return donors missing: {donor_path}")

    pool = pd.read_parquet(donor_path)
    preview_engine = isolated_engine_for_preview()
    with use_q4_kickoff_scenario(
        preview_engine,
        pool,
        touchback_rate=touchback_rate,
        seed=EXPERIMENTAL_KICKOFF_SEED,
        rules_year=2026,
    ):
        result = preview_engine["recommend"](
            dict(state), n_simulations=simulations, seed=action_seed,
        )
    return result



def preview_observed_2025_recommendation(
    state: Mapping, *,
    donor_path: Path = OBSERVED_2025_PATH,
    simulations: int = EXPERIMENTAL_SIMULATIONS,
    action_seed: int = EXPERIMENTAL_ACTION_SEED,
) -> dict:
    """Run a 2025-observed kickoff continuation in a PRIVATE engine namespace.

    No guessed touchback rate. The empirical 2025 ordinary kickoff type,
    position and clock are drawn JOINTLY from a frozen data file.
    The 2025 season has been examined and is not a pristine holdout.
    This is NOT a validated 2026+ forecast and never changes production.
    """
    if int(state.get("qtr", 0)) != 4:
        raise ValueError("Observed 2025 kickoff preview is Q4 only.")
    if simulations < 200:
        raise ValueError("At least 200 simulations are required for the preview.")
    pool = load_frozen_pool(donor_path)
    isolated = isolated_engine_for_preview()
    with use_frozen_2025_kickoffs(
        isolated, pool,
        seed=EXPERIMENTAL_KICKOFF_SEED,
        rules_year=2026,
    ):
        return isolated["recommend"](
            dict(state),
            n_simulations=simulations,
            seed=action_seed,
        )



def preview_observed_2025_with_onside(
    state: Mapping, *,
    attempt_probability: float,
    recovery_probability: float,
    simulations: int = EXPERIMENTAL_SIMULATIONS,
    action_seed: int = EXPERIMENTAL_ACTION_SEED,
) -> dict:
    """Selected 2025 onside outcomes plus explicit, NOT calibrated assumptions.

    Only after a Q4 FG while still trailing; all non-onside branches are
    identical to the ordinary 2025 scenario, with preserved RNG stream.
    Returns a separate engine result; production remains untouched.
    """
    if int(state.get("qtr", 0)) != 4:
        raise ValueError("Onside sensitivity is Q4 regulation only.")
    if float(state.get("score_differential", 0)) + 3 >= 0:
        raise ValueError(
            "This experiment only covers FGs that leave the kicking team trailing."
        )
    if simulations < 200:
        raise ValueError("At least 200 simulations are required.")
    validate_what_if(attempt_probability, recovery_probability)
    ordinary = load_frozen_pool()
    onside, _ = load_onside_verified()
    private = isolated_engine_for_preview()
    with use_q4_onside_what_if(
        private, ordinary, onside,
        p_attempt=attempt_probability,
        p_recovery=recovery_probability,
        ordinary_seed=EXPERIMENTAL_KICKOFF_SEED,
    ):
        return private["recommend"](
            dict(state), n_simulations=simulations, seed=action_seed,
        )


def observed_2025_onside_summary() -> dict:
    """Descriptive historical count, not attempt or recovery forecast."""
    _, info = load_onside_verified()
    return {
        "attempts": int(info["all_on_side_eligible"]),
        "recovered": int(info["official_kicker_recoveries"]),
        "matched": int(info["matched_reliable"]),
        "success_donor_count": int(
            info["reliable_first_snap"]["kicker_recovers"]["n"]
        ),
        "failed_donor_count": int(
            info["reliable_first_snap"]["receiver_recovers"]["n"]
        ),
    }


def observed_2025_pool_summary(
    donor_path: Path = OBSERVED_2025_PATH,
) -> dict:
    """Describe frozen observed kickoff assumptions, not forward calibration."""
    pool = load_frozen_pool(donor_path)
    touchback = pool["transition_type"].eq("touchback")
    return {
        "kickoffs": int(len(pool)),
        "games": int(pool["game_id"].nunique()),
        "touchback_rate": float(touchback.mean()),
        "receiver_own_avg": float(pool["own_yardline"].mean()),
        "mean_return_elapsed": float(
            pool.loc[~touchback, "elapsed_seconds"].mean()
        ),
    }



def observed_2025_kickoff_coverage(
    coverage_path: Path = OBSERVED_2025_COVERAGE_PATH,
    *,
    donor_path: Path = OBSERVED_2025_PATH,
) -> dict:
    """Validate a versioned, descriptive event-coverage census for the UI.

    The 2025 preview conditions on ordinary kickoffs. Other event branches
    are NOT modeled, and observed rates are not forecasts for 2026+ rules.
    A missing/stale/malformed report is an error, not an invitation to invent
    a denominator or quietly claim that 100% of kickoffs are supported.
    """
    if not coverage_path.is_file():
        raise FileNotFoundError(
            f"Frozen 2025 kickoff coverage evidence missing: {coverage_path}"
        )
    obj = json.loads(coverage_path.read_text())
    if (
        obj.get("data_year") != 2025
        or obj.get("no_2026_outcomes") is not True
        or obj.get("not_a_forecast") is not True
        or obj.get("regulation_only") is not True
        or obj.get("kickoff_spot_yardline_100") != 35
        or obj.get("kickoff_seconds_floor_exclusive") != 40.0
    ):
        raise ValueError("Frozen kickoff coverage metadata is not eligible.")

    required = {
        "2025 Q4 STANDARD 35, >40 SEC": "all",
        "2025 Q4 FINAL 5 MIN, >40 SEC": "late",
        "Q4 FINAL 5, RECEIVER LEADS 1-3": "receiver_leads_1to3",
        "Q4 FINAL 5, RECEIVER TRAILS 1-3": "receiver_trails_1to3",
    }
    category_labels = {
        "modeled_ordinary", "onside_attempt", "kickoff_return_td",
        "penalty_or_rekick", "other_unmatched",
    }
    entries = obj.get("groups")
    if not isinstance(entries, list) or len(entries) != len(required):
        raise ValueError("Missing or duplicate historical kickoff contexts.")
    result: dict = {}
    for entry in entries:
        name = entry.get("name")
        if name not in required or required[name] in result:
            raise ValueError("Unknown or duplicate kickoff context.")
        counts = entry.get("counts")
        if not isinstance(counts, dict) or set(counts) != category_labels:
            raise ValueError("Unmodeled kickoff categories cannot be inferred.")
        if any(type(c) is not int or c < 0 for c in counts.values()):
            raise ValueError("Invalid kickoff outcome counts.")
        total = entry.get("total_kickoffs")
        if type(total) is not int or total < 0 or sum(counts.values()) != total:
            raise ValueError("Kickoff outcomes fail the exhaustive-count check.")
        share = entry.get("ordinary_share")
        expected = counts["modeled_ordinary"] / total if total else None
        if share is None and expected is not None:
            raise ValueError("Missing historical ordinary kickoff share.")
        if share is not None and (
            expected is None or abs(float(share) - expected) > 1e-10
        ):
            raise ValueError("Kickoff coverage fraction disagrees with counts.")
        group = {
            "total_kickoffs": total,
            "games": int(entry["games"]),
            "counts": counts.copy(),
            "ordinary_share": expected,
            "low_support": bool(entry["low_support"]),
        }
        if group["games"] < 0 or group["games"] > total:
            raise ValueError("Invalid kickoff coverage game counts.")
        result[required[name]] = group

    full = result["all"]
    if full["counts"]["modeled_ordinary"] != len(load_frozen_pool(donor_path)):
        raise ValueError(
            "Historical coverage and frozen ordinary kickoff pool mismatch."
        )
    if result["late"]["total_kickoffs"] > full["total_kickoffs"]:
        raise ValueError("Historical late Q4 set exceeds Q4 universe.")
    for key in ("receiver_leads_1to3", "receiver_trails_1to3"):
        if result[key]["total_kickoffs"] > result["late"]["total_kickoffs"]:
            raise ValueError("Historical score subset exceeds late Q4 universe.")
    return result


def action_comparison(
    production_result: Mapping, scenario_result: Mapping,
) -> pd.DataFrame:
    """Side-by-side probabilities and MC SEs for eligible normal actions."""
    def extract(result: Mapping, prefix: str) -> pd.DataFrame:
        raw = result["results"]
        eligible = raw.loc[
            raw["eligible"] & raw["action"].isin(LABELS),
            ["action", "expected_win_probability", "mc_se"],
        ].copy()
        if eligible["action"].duplicated().any():
            raise ValueError("Duplicate eligible action in recommendation.")
        eligible = eligible.rename(columns={
            "expected_win_probability": f"{prefix}_wp",
            "mc_se": f"{prefix}_mc_se",
        })
        return eligible

    p = extract(production_result, "production")
    s = extract(scenario_result, "scenario")
    joined = p.merge(s, on="action", how="outer", validate="one_to_one")
    if joined.empty:
        raise ValueError("No normal eligible actions to compare.")
    if joined[["production_wp", "scenario_wp"]].isna().any().any():
        raise ValueError("Eligibility differs between production and preview.")

    table = pd.DataFrame({
        "Action": joined["action"].map(LABELS),
        "Production WP (%)": 100 * joined["production_wp"],
        "2025+ scenario WP (%)": 100 * joined["scenario_wp"],
        "Scenario MC SE (pp)": 100 * joined["scenario_mc_se"],
    })
    table["Scenario minus production (pp)"] = (
        table["2025+ scenario WP (%)"] - table["Production WP (%)"]
    )
    return table.sort_values(
        "2025+ scenario WP (%)", ascending=False,
    ).reset_index(drop=True)



def recommendation_sensitivity(
    production_result: Mapping, scenario_result: Mapping,
) -> dict:
    """Explain an action-order reversal or simulated near-tie, not causality."""
    prod = production_result["results"]
    eligible = prod.loc[
        prod["eligible"] & prod["action"].isin(LABELS),
        ["action", "expected_win_probability"],
    ].sort_values("expected_win_probability", ascending=False)
    if eligible.empty:
        raise ValueError("No eligible production action to compare.")
    official = LABELS[str(eligible.iloc[0]["action"])]
    current = preview_rank_with_uncertainty(scenario_result)
    flipped = official != current["leader"]
    uncertain = bool(current["near_tie"])
    return {
        "production_leader": official,
        "scenario_point_leader": current["leader"],
        "reversed": flipped,
        "scenario_near_tie": uncertain,
        "model_sensitive": bool(flipped or uncertain),
    }


def preview_rank_with_uncertainty(scenario_result: Mapping) -> dict:
    """Describe scenario winner, explicitly preserving near ties."""
    rows = scenario_result["results"].loc[
        scenario_result["results"]["eligible"]
        & scenario_result["results"]["action"].isin(LABELS)
    ].sort_values("expected_win_probability", ascending=False)
    if rows.empty:
        raise ValueError("No eligible scenario actions.")
    first = rows.iloc[0]
    second = rows.iloc[1] if len(rows) > 1 else None
    out = {
        "leader": LABELS[str(first["action"])],
        "leader_wp_pct": 100 * float(first["expected_win_probability"]),
        "runner_up": None,
        "gap_pp": None,
        "near_tie": False,
    }
    if second is None:
        return out
    gap = float(first["expected_win_probability"] - second["expected_win_probability"])
    se = float(np.hypot(first["mc_se"], second["mc_se"]))
    out.update({
        "runner_up": LABELS[str(second["action"])],
        "gap_pp": 100 * gap,
        "near_tie": bool(not np.isfinite(se) or gap <= 1.96 * se),
    })
    return out
