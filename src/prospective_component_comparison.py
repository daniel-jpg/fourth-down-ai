"""Sealed component-level candidate validation (synthetic-only in current CI).

This compares factual frozen-versus-candidate normal GO CONVERSION predictions
and regulation scrimmage WIN PROBABILITY predictions. These descriptive
metrics are NOT estimates of an unchosen FG/GO/PUNT coaching action's effect.

No network, no NFL data downloader, no model fitting, no 2026 Step 66/83
dispatch, no automatic app/release changes. For future 2027 prospective
scoring, predictions must be snap-time-only, separately sealed with their
sha256 BEFORE observed outcomes are joined, and candidate artifact hashes
must be locked in an independently reviewed source-of-truth.

A manifest can ASSERT provenance but cannot prove an upstream extractor
was leakage-free. Unreviewed prospective outputs are not approval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.fourth_down_policy_study import POLICY_COMMIT

TRAINING_END = 2022
HOLDOUT_SEASON = 2027
GO_ACTIONS = frozenset({"NORMAL_GO_RUN", "NORMAL_GO_PASS"})
WP_KEYS = frozenset({
    "game_id", "play_id", "season", "quarter", "site",
    "yardline_100", "ydstogo", "game_seconds_remaining",
    "offense_score_margin", "feature_asof", "frozen_wp", "candidate_wp",
})
GO_KEYS = WP_KEYS.difference({"frozen_wp", "candidate_wp"}) | {
    "observed_go_action", "frozen_conversion", "candidate_conversion",
}
WP_LABEL_KEYS = frozenset({"game_id", "play_id", "offense_win"})
GO_LABEL_KEYS = frozenset({"game_id", "play_id", "converted"})
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 2027
MIN_MEANINGFUL_GAMES = 20


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value.strip()


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    n = float(value)
    if not math.isfinite(n) or not low <= n <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return n


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _schema(row: Any, keys: frozenset, name: str) -> None:
    if not isinstance(row, dict) or set(row) != keys:
        missing = sorted(keys.difference(row) if isinstance(row, dict) else keys)
        extra = sorted(set(row).difference(keys) if isinstance(row, dict) else [])
        raise ValueError(
            f"{name} must have EXACT approved keys; missing={missing}; "
            f"extra={extra}; no post-snap fields in predictions"
        )


def _key(record: Mapping[str, Any]) -> tuple[str, str]:
    return _str(record["game_id"], "game_id"), _str(record["play_id"], "play_id")


def _predictor_rows(rows: Any, name: str, keys: frozenset) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} needs nonempty pre-snap prediction rows")
    result = {}
    for row in rows:
        _schema(row, keys, name)
        key = _key(row)
        if key in result:
            raise ValueError(f"Duplicate {name} play identifier {key}")
        _integer(row["season"], "season", HOLDOUT_SEASON, HOLDOUT_SEASON)
        _integer(row["quarter"], "quarter", 1, 4)
        if row["feature_asof"] != "pre_snap":
            raise ValueError("Only pre_snap features are permitted")
        if row["site"] not in ("HOME", "AWAY", "NEUTRAL"):
            raise ValueError("Offense site must be HOME/AWAY/NEUTRAL")
        yard = _number(row["yardline_100"], "yardline_100", .5, 99.5)
        togo = _number(row["ydstogo"], "ydstogo", 0, 100)
        if togo > max(1, yard) + 1e-9:
            raise ValueError("Yards-to-go must not cross opposing goal line")
        clock = _number(row["game_seconds_remaining"], "game_seconds_remaining", 0, 3600)
        low, high = {1:(2700,3600), 2:(1800,2700), 3:(900,1800), 4:(0,900)}[row["quarter"]]
        if not low <= clock <= high:
            raise ValueError("Clock and quarter are inconsistent")
        _number(row["offense_score_margin"], "offense_score_margin", -150, 150)
        for p in (
            ("frozen_wp", "candidate_wp") if name == "wp"
            else ("frozen_conversion", "candidate_conversion")
        ):
            _number(row[p], p, 0, 1)
        if name == "go" and row["observed_go_action"] not in GO_ACTIONS:
            raise ValueError("Factual GO cohort must contain only normal RUN/PASS GO")
        result[key] = dict(row)
    return result


def _label_rows(rows: Any, keys: frozenset, name: str) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} needs final factual outcome rows")
    result = {}
    for row in rows:
        _schema(row, keys, name)
        key = _key(row)
        if key in result:
            raise ValueError("Duplicate factual outcome identifier")
        label_key = "offense_win" if name == "wp" else "converted"
        _integer(row[label_key], label_key, 0, 1)
        result[key] = dict(row)
    return result


def forecast_sha256(predictions: Mapping[str, Any]) -> str:
    if not isinstance(predictions, dict) or set(predictions) != {"wp", "go"}:
        raise ValueError("Need wp and go pre-snap prediction collections")
    payload = json.dumps(
        predictions, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(manifest: Any, predictions: Mapping[str, Any]) -> str:
    if not isinstance(manifest, dict):
        raise ValueError("Missing frozen manifest")
    mode = manifest.get("mode")
    if mode not in ("synthetic", "prospective"):
        raise ValueError("Audit mode must be synthetic or prospective")
    for k, v in (
        ("holdout_season", HOLDOUT_SEASON),
        ("training_end_season", TRAINING_END),
        ("feature_policy", "pre_snap_only"),
        ("decision_policy_commit", POLICY_COMMIT),
    ):
        if manifest.get(k) != v:
            raise ValueError(f"Frozen independent-validation manifest mismatch: {k}")
    if mode == "synthetic":
        if manifest.get("source_type") != "fabricated_only":
            raise ValueError("Synthetic mode must use fabricated_only source")
        if manifest.get("final_2027_regular_season_complete") is True:
            raise ValueError("Synthetic fixtures cannot claim future season complete")
    else:
        # This is deliberately strict; the manifest alone does not establish
        # genuine time-of-forecast provenance or a trusted final-game source.
        if manifest.get("final_2027_regular_season_complete") is not True:
            raise ValueError("Independent prospective test requires completed 2027 REG")
        if manifest.get("source_type") != "independently_verified_sealed":
            raise ValueError("Unreviewed prospective sources are prohibited")
        if manifest.get("sealed_before_outcomes") is not True:
            raise ValueError("Model forecasts must be sealed before joining outcomes")
        for key in (
            "go_candidate_artifact_sha256", "wp_candidate_artifact_sha256",
            "extractor_source_sha256", "prediction_archive_sha256",
        ):
            val = manifest.get(key)
            if (not isinstance(val, str) or len(val) != 64
                    or any(c not in "0123456789abcdef" for c in val)
                    or val == "0"*64):
                raise ValueError(f"Missing independently pinned provenance: {key}")
        if manifest["prediction_archive_sha256"] != forecast_sha256(predictions):
            raise ValueError("Sealed pre-snap forecast SHA-256 does not match payload")
    return mode


def _match(predictions: Mapping, observations: Mapping, label: str) -> list:
    if set(predictions) != set(observations):
        raise ValueError(
            f"Unmatched {label} forecast/outcome IDs: missing predictions="
            f"{len(set(observations)-set(predictions))}, "
            f"missing outcomes={len(set(predictions)-set(observations))}"
        )
    return [{**predictions[k], **observations[k]} for k in sorted(predictions)]


def _ci(gains: Sequence[float], games: Sequence[str], draws: int, seed: int) -> list | None:
    arr = np.asarray(gains, dtype=float)
    group = np.asarray(games).astype(str)
    if arr.shape != group.shape or not len(arr) or not np.isfinite(arr).all():
        raise ValueError("Paired per-play gains must be finite and aligned")
    ids, ix = np.unique(group, return_inverse=True)
    if len(ids) < 2:
        return None
    sums = np.bincount(ix, weights=arr)
    sizes = np.bincount(ix)
    rng = np.random.default_rng(seed)
    output = np.empty(draws)
    for i in range(draws):
        sample = rng.integers(0, len(ids), len(ids))
        output[i] = sums[sample].sum()/sizes[sample].sum()
    return [float(z) for z in np.quantile(output, (.025, .975))]


def _score_group(rows: list[dict], baseline_key: str, contender_key: str,
                 label_key: str, *, draws: int, seed: int) -> dict:
    games = len({r["game_id"] for r in rows})
    if not rows:
        return {"states": 0, "games": 0, "status": "NO_SUPPORT"}
    labels = np.asarray([r[label_key] for r in rows], dtype=float)
    baseline = np.asarray([r[baseline_key] for r in rows], dtype=float)
    challenger = np.asarray([r[contender_key] for r in rows], dtype=float)
    base_loss = (labels-baseline)**2
    new_loss = (labels-challenger)**2
    return {
        "states": len(rows),
        "games": games,
        "factual_outcome_rate": float(labels.mean()),
        "frozen_brier": float(base_loss.mean()),
        "candidate_brier": float(new_loss.mean()),
        "paired_brier_gain_frozen_minus_candidate": float((base_loss-new_loss).mean()),
        "paired_gain_95pct_cluster_ci": _ci(
            base_loss-new_loss, [r["game_id"] for r in rows], draws, seed
        ),
        "status": (
            "SPARSE_DESCRIPTIVE_ONLY" if games < MIN_MEANINGFUL_GAMES
            else "COMPARABLE_FACTUAL_FORECASTS"
        ),
    }


def _fixed_buckets(rows: list[dict], name: str) -> dict[str, list[dict]]:
    """Prespecified subgroup definitions, never chosen after seeing labels."""
    q4 = lambda r: r["quarter"] == 4
    own10 = lambda r: r["yardline_100"] >= 90
    short = lambda r: 1.5 <= r["ydstogo"] <= 3.5
    if name == "wp":
        buckets = {
            "ALL": lambda r: True,
            "Q1_TO_Q3": lambda r: not q4(r),
            "Q4": q4,
            "Q4_FINAL_FIVE": lambda r: q4(r) and 45 <= r["game_seconds_remaining"] <= 300,
            "Q4_FINAL_FIVE_OWN10_LEADING4": lambda r: (
                q4(r) and 45 <= r["game_seconds_remaining"] <= 300
                and own10(r) and r["offense_score_margin"] == 4
            ),
            "HOME": lambda r: r["site"] == "HOME",
            "AWAY": lambda r: r["site"] == "AWAY",
            "NEUTRAL": lambda r: r["site"] == "NEUTRAL",
        }
    elif name == "go":
        buckets = {
            "ALL": lambda r: True,
            "GO_RUN": lambda r: r["observed_go_action"] == "NORMAL_GO_RUN",
            "GO_PASS": lambda r: r["observed_go_action"] == "NORMAL_GO_PASS",
            "FOURTH_2_OR_3": short,
            "OWN10_FOURTH_2_OR_3": lambda r: own10(r) and short(r),
            "Q4": q4,
            "HOME": lambda r: r["site"] == "HOME",
            "AWAY": lambda r: r["site"] == "AWAY",
        }
    else:
        raise ValueError("Component must be wp or go")
    return {key: [r for r in rows if predicate(r)] for key,predicate in buckets.items()}


def evaluate(
    predictions: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *, draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    """Use two distinct files/sections: forecasts FIRST, labels LAST."""
    if type(draws) is not int or not 20 <= draws <= 10000:
        raise ValueError("Game-bootstrap draws must be integer 20..10000")
    if not isinstance(predictions, dict) or set(predictions) != {"wp", "go"}:
        raise ValueError("Need both wp and go prediction lists")
    if not isinstance(outcomes, dict) or set(outcomes) != {"wp", "go"}:
        raise ValueError("Need separately sourced wp/go outcome lists")
    mode = _manifest(manifest, predictions)
    matched = {
        "wp": _match(
            _predictor_rows(predictions["wp"], "wp", WP_KEYS),
            _label_rows(outcomes["wp"], WP_LABEL_KEYS, "wp"),
            "WP",
        ),
        "go": _match(
            _predictor_rows(predictions["go"], "go", GO_KEYS),
            _label_rows(outcomes["go"], GO_LABEL_KEYS, "go"),
            "GO",
        ),
    }
    wp_games = len({r["game_id"] for r in matched["wp"]})
    go_games = len({r["game_id"] for r in matched["go"]})
    if mode == "prospective" and (wp_games < 100 or go_games < 100):
        raise ValueError("Prospective component test needs 100 distinct games per component")
    specs = {
        "wp": ("frozen_wp", "candidate_wp", "offense_win"),
        "go": ("frozen_conversion", "candidate_conversion", "converted"),
    }
    result = {}
    for name in ("wp", "go"):
        b, c, label = specs[name]
        result[name] = {
            subgroup: _score_group(rows, b, c, label,
                                   draws=draws, seed=BOOTSTRAP_SEED)
            for subgroup, rows in _fixed_buckets(matched[name], name).items()
        }
    # This is an evidence REVIEW checklist, not a production action.
    wp_q4 = result["wp"]["Q4"]
    wp_all = result["wp"]["ALL"]
    go_all = result["go"]["ALL"]
    own_go = result["go"]["OWN10_FOURTH_2_OR_3"]
    def positive_ci(r) -> bool:
        ci = r.get("paired_gain_95pct_cluster_ci")
        return ci is not None and ci[0] > 0
    reviewable = bool(
        mode == "prospective"
        and wp_all["games"] >= 100 and positive_ci(wp_all)
        and wp_q4["games"] >= 100 and positive_ci(wp_q4)
        and go_all["games"] >= 100 and positive_ci(go_all)
        and own_go["games"] >= 20
        and result["wp"]["HOME"]["games"] >= 20
        and result["wp"]["AWAY"]["games"] >= 20
    )
    return {
        "study": "2027_component_candidates_vs_frozen_factual_calibration",
        "mode": mode,
        "policy_commit": POLICY_COMMIT,
        "holdout_season": HOLDOUT_SEASON,
        "forecasts_sha256": forecast_sha256(predictions),
        "n_wp_states": len(matched["wp"]),
        "n_go_attempts": len(matched["go"]),
        "wp_calibration": result["wp"],
        "go_conversion_calibration": result["go"],
        "model_review_gate": {
            "passes_prespecified_component_coverage_and_gain": reviewable,
            "automatic_promotion_allowed": False,
            "causal_coaching_claim_allowed": False,
            "interpretation": (
                "Factual forecasts on actually observed choices only. "
                "No counterfactual result under an unchosen GO/FG/PUNT action. "
                "A future independent source/code audit and explicit release "
                "approval are separate prerequisites."
            ),
        },
        "warning": (
            "Synthetic results are fabricated tests, never football evidence. "
            "A provenance manifest cannot prove pre-snap-only extraction, "
            "time-stamped sealing or independence. Prospective use requires "
            "independent audits of data, feature availability, model hashes "
            "and final-game checks. 2026 Step 66/83 remains separate."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Offline only, never fetch NFL outcomes")
    p.add_argument("input_json", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    payload = json.loads(args.input_json.read_text())
    result = evaluate(payload["predictions"], payload["outcomes"], payload["manifest"])
    txt = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(txt)
    else:
        print(txt, end="")


if __name__ == "__main__":
    main()
