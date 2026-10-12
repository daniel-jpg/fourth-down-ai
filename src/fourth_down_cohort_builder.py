"""Synthetic-only dry run of the preregistered 2027 fourth-down cohort join.

Separates PRE-SNAP forecasts from logged play classifications and post-game
win/loss labels. Does NOT retrieve NFL data, execute the frozen engine, fit
nuisance models, or authorize access to any 2026/2027 real-world outcomes.

Real prospective use is intentionally blocked until the extractor, model
provenance and final-season gate receive a separate versioned review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.fourth_down_policy_study import ACTIONS, POLICY_COMMIT

PRE_SNAP_KEYS = frozenset({
    "game_id", "play_id", "season", "week", "quarter", "down",
    "offense_team", "home_team", "away_team", "site",
    "game_seconds_remaining", "yardline_100", "ydstogo",
    "score_differential", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "feature_asof", "policy_commit",
    "policy_action", "eligible_actions", "q_wp_by_action",
    "propensity_policy", "mu_policy", "close_call",
})
OBSERVED_KEYS = frozenset({"game_id", "play_id", "observed_action", "no_play"})
FINAL_KEYS = frozenset({
    "game_id", "season", "week", "home_team", "away_team",
    "home_score_final", "away_score_final", "status", "game_type",
})
REQUIRED_MANIFEST = {
    "mode": "synthetic",
    "source_type": "fabricated_only",
    "policy_commit": POLICY_COMMIT,
    "feature_policy": "pre_snap_only",
    "holdout_season": 2027,
    "nuisance_training_end_season": 2025,
}


def _schema(row: Any, required: frozenset[str], kind: str) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"{kind} must be a JSON object")
    missing, excess = required.difference(row), row.keys() - required
    if missing or excess:
        raise ValueError(
            f"{kind} schema mismatch: missing={sorted(missing)} "
            f"extra={sorted(excess)}; pre-snap cannot contain labels"
        )


def _str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _int(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be integer in [{low}, {high}]")
    return value


def _number(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be finite numeric")
    n = float(value)
    if not math.isfinite(n) or not low <= n <= high:
        raise ValueError(f"{label} must be finite in [{low}, {high}]")
    return n


def _id(row: Mapping[str, Any]) -> tuple[str, str]:
    return (_str(row["game_id"], "game_id"), _str(row["play_id"], "play_id"))


def _checked_rows(records: Any, label: str, schema: frozenset[str], key_fn):
    if not isinstance(records, list):
        raise ValueError(f"{label} must be a list")
    indexed = {}
    for row in records:
        _schema(row, schema, label)
        key = key_fn(row)
        if key in indexed:
            raise ValueError(f"Duplicate {label} identifier {key!r}")
        indexed[key] = row
    return indexed


def _validate_prespnap(row: Mapping[str, Any]) -> dict[str, Any]:
    _int(row["season"], "season", 2027, 2027)
    _int(row["week"], "week", 1, 18)
    quarter = _int(row["quarter"], "quarter", 1, 5)
    down = _int(row["down"], "down", 1, 4)
    offense = _str(row["offense_team"], "offense_team")
    home = _str(row["home_team"], "home_team")
    away = _str(row["away_team"], "away_team")
    if home == away or offense not in (home, away):
        raise ValueError("pre-snap offense must match one of two distinct teams")
    site = row["site"]
    if site not in ("HOME", "AWAY", "NEUTRAL"):
        raise ValueError("site must be HOME/AWAY/NEUTRAL")
    if site == "HOME" and offense != home or site == "AWAY" and offense != away:
        raise ValueError("site must be from current offense's perspective")
    clock = _number(row["game_seconds_remaining"], "game_seconds_remaining", 0, 3600)
    if quarter <= 4:
        boundaries = {1: (2700, 3600), 2: (1800, 2700),
                      3: (900, 1800), 4: (0, 900)}
        lo, hi = boundaries[quarter]
        if not lo <= clock <= hi:
            raise ValueError("game clock does not match regulation quarter")
    yardline = _number(row["yardline_100"], "yardline_100", 0.5, 99.5)
    togo = _number(row["ydstogo"], "ydstogo", 0.0, 100.0)
    if togo > max(1.0, yardline) + 1e-9:
        raise ValueError("ydstogo cannot cross opponent end zone")
    _number(row["score_differential"], "score_differential", -150, 150)
    for label in ("posteam_timeouts_remaining", "defteam_timeouts_remaining"):
        _int(row[label], label, 0, 3)
    if row["feature_asof"] != "pre_snap":
        raise ValueError("pre-snap forecasts must be pre_snap only")
    if row["policy_commit"] != POLICY_COMMIT:
        raise ValueError("pre-snap policy forecasts must use pinned policy commit")
    if type(row["close_call"]) is not bool:
        raise ValueError("close_call must be a pre-snap boolean")
    eligible = row["eligible_actions"]
    wp = row["q_wp_by_action"]
    if (not isinstance(eligible, list) or not eligible
            or len(eligible) != len(set(eligible))
            or any(a not in ACTIONS for a in eligible)):
        raise ValueError("eligible_actions must be distinct normal actions")
    if not isinstance(wp, dict) or set(wp) != set(eligible):
        raise ValueError("pre-snap WP must cover exactly eligible normal actions")
    values = {a: _number(v, f"q_wp_by_action[{a}]", 0, 1) for a, v in wp.items()}
    policy_action = row["policy_action"]
    if policy_action not in values or values[policy_action] < max(values.values()) - 1e-12:
        raise ValueError("policy_action must be a highest-WP eligible normal action")
    mu = _number(row["mu_policy"], "mu_policy", 0, 1)
    propensity = _number(row["propensity_policy"], "propensity_policy", 0, 1)
    return {"quarter": quarter, "down": down, "offense": offense,
            "home": home, "away": away, "site": site,
            "values": values, "mu": mu, "propensity": propensity}


def build_synthetic_cohort(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically join fabricated data, preserving exclusion audit rows.

    The observed play classification and final win label are joined AFTER all
    pre-snap forecasts are validated. Never use them as model features.
    """
    if not isinstance(payload, dict) or set(payload) != {
        "manifest", "pre_snap", "observed_plays", "final_games"
    }:
        raise ValueError("Input requires manifest, pre_snap, observed_plays, final_games")
    manifest = payload["manifest"]
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != expected for key, expected in REQUIRED_MANIFEST.items()
    ):
        raise ValueError("Only fabricated synthetic 2027 data with pinned <=2025 nuisances is allowed")
    # Refuse real prospective evaluation even if a caller adds an enticing flag.
    if set(manifest) != set(REQUIRED_MANIFEST):
        raise ValueError("Unexpected manifest provenance; prospective mode is not implemented")
    pre = _checked_rows(payload["pre_snap"], "pre_snap", PRE_SNAP_KEYS, _id)
    observed = _checked_rows(payload["observed_plays"], "observed_plays", OBSERVED_KEYS, _id)
    finals = _checked_rows(payload["final_games"], "final_games", FINAL_KEYS,
                           lambda x: _str(x["game_id"], "game_id"))
    # Strict all-row checks: invalid feature schema or unsafe split NEVER
    # becomes a silent exclusion that hides label leakage.
    checked = {key: _validate_prespnap(row) for key, row in pre.items()}
    for game_id, game in finals.items():
        _int(game["season"], "final season", 2027, 2027)
        _int(game["week"], "final week", 1, 18)
        _str(game["home_team"], "home_team")
        _str(game["away_team"], "away_team")
        if game["home_team"] == game["away_team"]:
            raise ValueError("Final game teams must differ")
        _int(game["home_score_final"], "home_score_final", 0, 150)
        _int(game["away_score_final"], "away_score_final", 0, 150)
        if game["status"] not in ("final", "pending") or game["game_type"] != "REG":
            raise ValueError("Final-game schema must be regular-season games")
    for row in observed.values():
        if type(row["no_play"]) is not bool:
            raise ValueError("no_play must be Boolean")
        _str(row["observed_action"], "observed_action")

    records, excluded = [], []
    for key in sorted(pre):
        row = pre[key]
        context = checked[key]
        outcome = observed.get(key)
        game = finals.get(key[0])
        reason = None
        if context["quarter"] == 5:
            reason = "overtime"
        elif context["down"] != 4:
            reason = "not_fourth_down"
        elif outcome is None:
            reason = "unmatched_observed_play"
        elif outcome["no_play"]:
            reason = "administrative_no_play"
        elif outcome["observed_action"] not in ACTIONS:
            reason = "fake_or_unclassified_action"
        elif game is None:
            reason = "unmatched_final_game"
        elif game["status"] != "final":
            reason = "game_not_final"
        elif (game["home_team"] != context["home"]
              or game["away_team"] != context["away"]
              or game["week"] != row["week"]):
            raise ValueError(f"Game team/week mismatch for {key!r}")
        elif game["home_score_final"] == game["away_score_final"]:
            reason = "tied_final_game"
        if reason is not None:
            excluded.append({"game_id": key[0], "play_id": key[1], "reason": reason})
            continue
        actual = outcome["observed_action"]
        score = game["home_score_final"] - game["away_score_final"]
        offense_win = int(score > 0) if context["offense"] == context["home"] else int(score < 0)
        records.append({
            "game_id": key[0], "play_id": key[1], "season": 2027,
            "week": row["week"], "quarter": context["quarter"],
            "observed_action": actual, "policy_action": row["policy_action"],
            "eligible_actions": row["eligible_actions"],
            "offense_win": offense_win,
            "q_wp_observed": context["values"].get(actual),
            "q_wp_policy": context["values"][row["policy_action"]],
            "mu_policy": context["mu"],
            "propensity_policy": context["propensity"],
            "feature_asof": "pre_snap",
            "site": context["site"], "close_call": row["close_call"],
            "yardline_100": row["yardline_100"],
            "ydstogo": row["ydstogo"],
        })
    # An observed fourth-down play with no reconstructed pre-snap decision
    # must NOT vanish from the cohort denominator. This is a different
    # extraction failure from having a pre-snap candidate with no logged play.
    for key in sorted(set(observed).difference(pre)):
        excluded.append({
            "game_id": key[0], "play_id": key[1],
            "reason": "unmatched_pre_snap",
        })
    excluded.sort(key=lambda r: (r["game_id"], r["play_id"], r["reason"]))
    exclusions = dict(sorted(Counter(x["reason"] for x in excluded).items()))
    canonical_input = json.dumps(payload, sort_keys=True, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8")
    return {
        "manifest": dict(manifest),
        "records": records,
        "exclusion_counts": exclusions,
        "excluded_decisions": excluded,
        "total_pre_snap_candidates": len(pre),
        "total_observed_plays": len(observed),
        "matched_decision_keys": len(set(pre).intersection(observed)),
        "included_decisions": len(records),
        "factual_unsupported_decisions": sum(
            r["q_wp_observed"] is None for r in records
        ),
        "synthetic_input_sha256": hashlib.sha256(canonical_input).hexdigest(),
        "warning": "SYNTHETIC ONLY. No real 2027 or reserved 2026 game outcomes accessed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fabricated-only 2027 cohort join dry run")
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = build_synthetic_cohort(json.loads(args.input_json.read_text()))
    result = json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result)


if __name__ == "__main__":
    main()
