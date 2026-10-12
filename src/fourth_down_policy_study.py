"""Prospective fourth-down decision audit: no network, no 2026 data, no model refit.

The input is a sealed list of *decision-time* records, evaluated after the
2027 REGULAR season finishes. Only synthetic records belong in CI.

This is an observational, single-decision off-policy diagnostic, NOT a
causal guarantee that choosing the model's action improves NFL wins.
It deliberately cannot read nflverse/2026 PBP, execute Step 66/83, or
change the public Streamlit recommendation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

POLICY_COMMIT = "31176d50af25db9dd42dae07e9b17cb5c2b930b4"
TARGET_SEASON = 2027
ACTIONS = frozenset({
    "NORMAL_GO_RUN", "NORMAL_GO_PASS", "FIELD_GOAL", "PUNT",
})
PROPENSITY_FLOOR = 0.05
PROPENSITY_CEILING = 0.95
MIN_SUPPORTED_DECISIONS = 200
MIN_SUPPORTED_GAMES = 100
MIN_EFFECTIVE_SAMPLE_SIZE = 100.0
MIN_DISAGREEMENTS = 40
MIN_SUPPORT_FRACTION = 0.60
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 2027
CALIBRATION_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
REQUIRED = {
    "game_id", "play_id", "season", "week", "quarter",
    "observed_action", "policy_action", "eligible_actions",
    "offense_win", "q_wp_observed", "q_wp_policy",
    "mu_policy", "propensity_policy", "feature_asof",
}


def _probability(value: Any, label: str, *, allow_missing: bool = False) -> float | None:
    if value is None and allow_missing:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not Boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return number


def _integer(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a JSON integer")
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> str:
    mode = manifest.get("mode")
    if mode not in {"synthetic", "prospective"}:
        raise ValueError("mode must be synthetic or prospective")
    if manifest.get("policy_commit") != POLICY_COMMIT:
        raise ValueError("policy_commit must match the frozen research-demo snapshot")
    if manifest.get("feature_policy") != "pre_snap_only":
        raise ValueError("feature_policy must be pre_snap_only")
    if manifest.get("holdout_season") != TARGET_SEASON:
        raise ValueError("Only independent 2027 holdout records are accepted")
    end_year = manifest.get("nuisance_training_end_season")
    if isinstance(end_year, bool) or not isinstance(end_year, int) or end_year > 2025:
        raise ValueError("Nuisance fits must exclude all 2026+ football outcomes")
    if mode == "prospective" and manifest.get("final_2027_regular_season_complete") is not True:
        raise ValueError("Prospective audit requires a completed 2027 regular season")
    if mode == "prospective" and not manifest.get("nuisance_provenance"):
        raise ValueError("Record propensity/outcome nuisance fitting provenance")
    return mode


def _optional_bounded_number(value: Any, label: str, *, low: float, high: float) -> float | None:
    """Validate optional decision-time continuous covariates without coercing flags."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not Boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")
    return number


def _validated_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("At least one record is required")
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, r in enumerate(records):
        missing = REQUIRED.difference(r)
        if missing:
            raise ValueError(f"row {i} lacks {sorted(missing)}")
        game_id, play_id = str(r["game_id"]), str(r["play_id"])
        if not game_id or game_id == "None" or not play_id or play_id == "None":
            raise ValueError("Every decision needs an identifiable game_id and play_id")
        pair = (game_id, play_id)
        if pair in seen:
            raise ValueError("Duplicate decision game_id/play_id")
        seen.add(pair)
        _integer(r["season"], "season", TARGET_SEASON, TARGET_SEASON)
        _integer(r["week"], "week", 1, 18)
        _integer(r["quarter"], "quarter", 1, 4)
        if r["feature_asof"] != "pre_snap":
            raise ValueError("Only pre-snap features are allowed")
        action, policy = r["observed_action"], r["policy_action"]
        if action not in ACTIONS or policy not in ACTIONS:
            raise ValueError("Only observed normal GO, FG and punt actions are eligible")
        eligible = r["eligible_actions"]
        if (
            not isinstance(eligible, (list, tuple)) or not eligible
            or len(set(eligible)) != len(eligible)
            or any(a not in ACTIONS for a in eligible)
            or policy not in eligible
        ):
            raise ValueError("Policy action must be eligible from pre-snap rules")
        y = r["offense_win"]
        if isinstance(y, bool) or y not in (0, 1):
            raise ValueError("Use completed non-tied offense-win labels 0 or 1")
        observed_wp = _probability(
            r["q_wp_observed"], "q_wp_observed", allow_missing=True
        )
        if action in eligible and observed_wp is None:
            raise ValueError("Factual supported actions require q_wp_observed")
        policy_wp = _probability(r["q_wp_policy"], "q_wp_policy")
        mu = _probability(r["mu_policy"], "mu_policy")
        propensity = _probability(r["propensity_policy"], "propensity_policy")
        site = r.get("site")
        if site is not None and site not in ("HOME", "AWAY", "NEUTRAL"):
            raise ValueError("site must be HOME, AWAY or NEUTRAL")
        close_call = r.get("close_call")
        if close_call is not None and type(close_call) is not bool:
            raise ValueError("close_call must be a pre-snap Boolean")
        yardline = _optional_bounded_number(
            r.get("yardline_100"), "yardline_100", low=0.5, high=99.5
        )
        togo = _optional_bounded_number(
            r.get("ydstogo"), "ydstogo", low=0., high=100.
        )
        if yardline is not None and togo is not None:
            if togo > max(1., yardline) + 1e-9:
                raise ValueError("ydstogo may not cross the opposing end zone")
        validated.append({
            "game_id": game_id,
            "play_id": play_id,
            "quarter": r["quarter"],
            "observed_action": action,
            "policy_action": policy,
            "eligible_actions": tuple(eligible),
            "offense_win": int(y),
            "q_wp_observed": observed_wp,
            "q_wp_policy": policy_wp,
            "mu_policy": mu,
            "propensity_policy": propensity,
            "site": site,
            "close_call": close_call,
            "yardline_100": yardline,
            "ydstogo": togo,
            "agree": action == policy,
        })
    return validated


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], fraction: float) -> float:
    values = sorted(values)
    at = (len(values) - 1) * fraction
    low = math.floor(at)
    high = math.ceil(at)
    return values[low] * (1.0 - (at - low)) + values[high] * (at - low)


def _ci_game_cluster(
    rows: Sequence[dict[str, Any]],
    contribution, *,
    draws: int, seed: int,
) -> list[float] | None:
    """Percentile interval resampling whole games, never individual plays."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["game_id"]].append(row)
    games = sorted(groups)
    if len(games) < 2:
        return None
    rng = random.Random(seed)
    draws_values = []
    for _ in range(draws):
        sample_games = rng.choices(games, k=len(games))
        total = 0.0
        n = 0
        for game in sample_games:
            for row in groups[game]:
                total += contribution(row)
                n += 1
        draws_values.append(total / n)
    return [
        _percentile(draws_values, 0.025),
        _percentile(draws_values, 0.975),
    ]


def _brier_record(row: dict[str, Any]) -> float:
    return (row["offense_win"] - row["q_wp_observed"]) ** 2


def _calibration_report(
    rows: Sequence[dict[str, Any]], *,
    draws: int, seed: int,
) -> dict[str, Any]:
    factual = [
        r for r in rows
        if r["q_wp_observed"] is not None
        and r["observed_action"] in r["eligible_actions"]
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in factual:
        buckets[row["observed_action"]].append(row)
    detail: dict[str, Any] = {}
    for action in sorted(ACTIONS):
        action_rows = buckets[action]
        detail[action] = {
            "decisions": len(action_rows),
            "distinct_games": len({r["game_id"] for r in action_rows}),
            "brier": _mean([_brier_record(r) for r in action_rows]),
            # Subgroup bins below 20 are descriptive only: no calibration claim.
        }
    reliability = []
    for i, (left, right) in enumerate(
        zip(CALIBRATION_BINS, CALIBRATION_BINS[1:])
    ):
        group = [
            r for r in factual
            if left <= r["q_wp_observed"] < right
            or (i == len(CALIBRATION_BINS) - 2
                and r["q_wp_observed"] == right)
        ]
        reliability.append({
            "probability_bin": [left, right],
            "n": len(group),
            "average_estimated_wp": _mean(
                [r["q_wp_observed"] for r in group]
            ),
            "observed_win_rate": _mean([r["offense_win"] for r in group]),
            "adequate_bin_size": len(group) >= 20,
        })
    return {
        "factual_supported_decisions": len(factual),
        "factual_unsupported_decisions": len(rows) - len(factual),
        "distinct_games": len({r["game_id"] for r in factual}),
        "brier": _mean([_brier_record(r) for r in factual]),
        "brier_95pct_game_cluster_ci": _ci_game_cluster(
            factual, _brier_record, draws=draws, seed=seed,
        ),
        "by_observed_action": detail,
        "reliability_bins": reliability,
    }


def _prespecified_subgroups(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fixed, DESCRIPTIVE breakouts registered before 2027 outcomes exist.

    Missing synthetic-fixture metadata is reported explicitly, never
    assigned to HOME/away/not-a-close-call by default. No subgroup
    selection or threshold uses the observed winner.
    """
    def summarize(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        factual = [
            row for row in group
            if row["q_wp_observed"] is not None
            and row["observed_action"] in row["eligible_actions"]
        ]
        return {
            "decisions": len(group),
            "distinct_games": len({r["game_id"] for r in group}),
            "factual_supported_decisions": len(factual),
            "factual_unsupported_decisions": len(group) - len(factual),
            "factual_brier": _mean([_brier_record(r) for r in factual]),
            "mean_factual_predicted_wp": _mean(
                [r["q_wp_observed"] for r in factual]
            ),
            "factual_observed_win_rate": _mean(
                [r["offense_win"] for r in factual]
            ),
        }

    def by_fixed_bucket(bucket, names: tuple[str, ...]) -> dict[str, Any]:
        return {
            label: summarize([r for r in rows if bucket(r) == label])
            for label in names
        }

    def deep_own_short_bucket(r: dict[str, Any]) -> str:
        pos = r["yardline_100"]
        distance = r["ydstogo"]
        if pos is None or distance is None:
            return "NOT_RECORDED"
        return ("OWN_1_TO_10_4TH_2_OR_3"
                if pos >= 90.0 and 1.5 <= distance <= 3.5 else "OTHER")

    return {
        "classification": (
            "Preregistered descriptive summaries only; NOT independently "
            "validated subgroup improvements or causal policy effects."
        ),
        "q4_vs_q1_to_q3": by_fixed_bucket(
            lambda r: "Q4" if r["quarter"] == 4 else "Q1_TO_Q3",
            ("Q1_TO_Q3", "Q4"),
        ),
        "offense_site": by_fixed_bucket(
            lambda r: r["site"] if r["site"] is not None else "NOT_RECORDED",
            ("HOME", "AWAY", "NEUTRAL", "NOT_RECORDED"),
        ),
        "close_call": by_fixed_bucket(
            lambda r: (
                "YES" if r["close_call"] is True else
                "NO" if r["close_call"] is False else "NOT_RECORDED"
            ),
            ("YES", "NO", "NOT_RECORDED"),
        ),
        "deep_own_short_yardage": by_fixed_bucket(
            deep_own_short_bucket,
            ("OWN_1_TO_10_4TH_2_OR_3", "OTHER", "NOT_RECORDED"),
        ),
    }


def _dr_policy_win(row: dict[str, Any]) -> float:
    estimate = row["mu_policy"]
    if row["agree"]:
        estimate += (
            row["offense_win"] - row["mu_policy"]
        ) / row["propensity_policy"]
    return estimate


def _policy_report(
    rows: Sequence[dict[str, Any]], *,
    draws: int, seed: int,
) -> dict[str, Any]:
    supported = [
        r for r in rows
        if PROPENSITY_FLOOR <= r["propensity_policy"] <= PROPENSITY_CEILING
    ]
    weights = [
        1.0 / r["propensity_policy"] if r["agree"] else 0.0
        for r in supported
    ]
    weight_total = sum(weights)
    weight_squared = sum(w * w for w in weights)
    ess = weight_total * weight_total / weight_squared if weight_squared else 0.0
    disagreements = sum(not r["agree"] for r in supported)
    sufficient = bool(
        len(supported) >= MIN_SUPPORTED_DECISIONS
        and len({r["game_id"] for r in supported}) >= MIN_SUPPORTED_GAMES
        and ess >= MIN_EFFECTIVE_SAMPLE_SIZE
        and disagreements >= MIN_DISAGREEMENTS
        and len(supported) / len(rows) >= MIN_SUPPORT_FRACTION
    )
    def difference(row: dict[str, Any]) -> float:
        return _dr_policy_win(row) - row["offense_win"]

    return {
        "supported_decisions": len(supported),
        "distinct_supported_games": len({r["game_id"] for r in supported}),
        "support_coverage": len(supported) / len(rows),
        "disagreements_with_coaches": disagreements,
        "matched_importance_weight_ess": ess,
        "support_rule": f"{PROPENSITY_FLOOR:.2f} <= e(policy|pre-snap X) <= {PROPENSITY_CEILING:.2f}",
        "sufficient_overlap_to_report_exploratory_comparison": sufficient,
        "coaches_observed_win_rate_on_supported": _mean(
            [r["offense_win"] for r in supported]
        ),
        "dr_one_step_policy_value": _mean(
            [_dr_policy_win(r) for r in supported]
        ),
        "dr_minus_observed_coaches_value": _mean(
            [difference(r) for r in supported]
        ),
        "dr_difference_95pct_game_cluster_ci": _ci_game_cluster(
            supported, difference, draws=draws, seed=seed + 1,
        ),
        "interpretation": (
            "Exploratory observational single-decision comparison, "
            "NOT causal proof, not a full-season sequential rollout, "
            "and not permission to remove coaching-uncertainty caveats."
        ),
    }


def evaluate(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Audit precomputed holdout records; never fetch games or run models."""
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise ValueError("bootstrap_draws must be an integer")
    if not 20 <= bootstrap_draws <= 10000:
        raise ValueError("bootstrap_draws must be between 20 and 10000")
    mode = _validate_manifest(manifest)
    rows = _validated_rows(records)
    if mode == "prospective" and len({r["game_id"] for r in rows}) < 100:
        raise ValueError("Require >=100 distinct games for a prospective report")
    if mode == "prospective" and any(
        row[key] is None
        for row in rows
        for key in ("site", "close_call", "yardline_100", "ydstogo")
    ):
        raise ValueError(
            "Prospective study requires complete pre-snap site, "
            "close_call, yardline_100 and ydstogo subgroup metadata"
        )
    return {
        "study": "2027_frozen_fourth_down_action_policy_diagnostic",
        "mode": mode,
        "policy_commit": POLICY_COMMIT,
        "season": TARGET_SEASON,
        "n_decisions": len(rows),
        "n_games": len({r["game_id"] for r in rows}),
        "agreement_fraction": _mean([float(r["agree"]) for r in rows]),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "factual_action_calibration": _calibration_report(
            rows, draws=bootstrap_draws, seed=seed,
        ),
        "observational_one_step_policy_comparison": _policy_report(
            rows, draws=bootstrap_draws, seed=seed,
        ),
        "prespecified_descriptive_subgroups": _prespecified_subgroups(rows),
        "warning": (
            "Only 2027 completed non-tied REGULATION play outcomes, "
            "with pre-snap features and frozen policy predictions. "
            "No evaluation on reserved 2026 outcomes. "
            "A positive observational difference cannot identify "
            "unobserved FG/GO counterfactual wins or justify removing "
            "a causal coaching disclaimer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SEALED 2027 decision rows; never scrape game data."
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    payload = json.loads(args.input_json.read_text())
    result = evaluate(payload["records"], payload["manifest"])
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
