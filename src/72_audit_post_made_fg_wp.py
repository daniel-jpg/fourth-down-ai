"""Step 72: real made-FG kickoff/first-snap WP attribution and Rams diagnostic.

Frozen production model only. Chronologically completed 2024 and previously
inspected 2025 REG games. 2026 remains unopened, and the live app is unchanged.

Two distinct questions:
1. Exactly why does a made FG's original kickoff-pending WP differ from
   averaging the SAME WP model over all frozen ordinary 2025 kickoff outcomes?
   Decompose by touchback/return without Monte Carlo noise.
2. On observed kickoffs following MADE field goals, how is the frozen
   synthetic kickoff-pending forecast calibrated against final game outcomes?
   Actual first-snap WP is a HINDSIGHT reference, never a pre-kickoff forecast.

Does NOT assess a counterfactual FG-vs-GO decision or predict onside kicks,
return scores and unusual penalty outcomes. Do not auto-promote any model.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kickoff_2025_empirical import load_frozen_pool, validate_frozen_pool

ENGINE_PATH = ROOT / "src/42_build_decision_engine.py"
STEP61 = ROOT / "src/61_audit_observed_kickoff_wp.py"
STEP64 = ROOT / "src/64_validate_2025_kickoff_encoding.py"
REPORT_PATH = ROOT / "artifacts/72_post_made_fg_wp_attribution.json"
SEASONS = (2024, 2025)

RAMS_EXAMPLE = {
    "qtr": 4,
    "game_seconds_remaining": 299.0,
    "yardline_100": 6.0,
    "ydstogo": 4.0,
    "score_differential": -4.0,
    "posteam_timeouts_remaining": 3.0,
    "defteam_timeouts_remaining": 3.0,
    "site": "AWAY",
    "roof": "outdoors",
}

# Actual scrimmage/try rows are barriers between a made FG and the next KO.
SCORING_ACTIONS = {
    "run", "pass", "qb_kneel", "qb_spike", "punt", "field_goal",
    "extra_point",
}


def import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load historical validator {path}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def kickoff_after_made_field_goal(pbp: pd.DataFrame) -> pd.DataFrame:
    """Identify the next kickoff after an actual made FG, within each game.

    Take the PRECEDING action in the same game (ignoring administrative rows)
    rather than grabbing an arbitrary prior made FG. No future events are
    allowed into the state features.
    """
    required = {
        "season", "season_type", "game_id", "play_id", "play_type", "qtr",
        "field_goal_result", "posteam",
    }
    if missing := required.difference(pbp.columns):
        raise ValueError(f"Missing PBP scoring fields: {sorted(missing)}")
    if pbp.empty or not pbp["season"].isin(SEASONS).all():
        raise ValueError("Only 2024-25 completed history may be loaded.")
    df = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    if df.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate play IDs prevent adjacent scoring attribution")
    action = df["play_type"].isin(SCORING_ACTIONS)
    game = df["game_id"]

    def previous_action(series: pd.Series) -> pd.Series:
        return series.where(action).groupby(game).transform(
            lambda group: group.ffill().shift(1)
        )

    previous_type = previous_action(df["play_type"])
    previous_fg_status = previous_action(df["field_goal_result"])
    previous_posteam = previous_action(df["posteam"])
    previous_id = previous_action(df["play_id"])
    kick = df["play_type"].eq("kickoff")
    # A field goal never has a Try; the scoring team's ensuing KO follows.
    made = previous_fg_status.fillna("").astype(str).str.lower().eq("made")
    mask = (
        kick
        & df["season_type"].eq("REG")
        & df["qtr"].eq(4)
        & previous_type.eq("field_goal")
        & made
        & df["posteam"].notna()
        & previous_posteam.notna()
        & df["posteam"].ne(previous_posteam)
    )
    matched = df.loc[mask, ["game_id", "play_id", "season"]].copy()
    matched["preceding_made_fg_play_id"] = previous_id[mask].to_numpy()
    return matched.rename(columns={"play_id": "play_id_kick"}).reset_index(drop=True)


def actual_made_fg_kickoff_pairs(pbp: pd.DataFrame, year: int,
                                 step61, step64) -> pd.DataFrame:
    """Intersect existing strict ordinary kickoff pairs with actual made FGs."""
    if year not in SEASONS or pbp.empty or not pbp["season"].eq(year).all():
        raise ValueError("A single supported completed season is required.")
    # 2024 older-rule matcher; 2025 standard-35 rule-aware matcher.
    if year == 2024:
        eligible = pbp.loc[pbp["season_type"].eq("REG")].copy()
        ordinary = step61.pair_2024_q4(eligible)
    else:
        ordinary = step64.ordinary_2025_q4_pairs(pbp)
    made = kickoff_after_made_field_goal(pbp)
    joined = ordinary.merge(
        made, on=["game_id", "play_id_kick"], how="inner",
        validate="one_to_one", suffixes=("", "_made"),
    )
    if not joined.empty and (
        joined["game_id"].isna().any()
        or joined["play_id_snap"].le(joined["play_id_kick"]).any()
        or joined.duplicated(["game_id", "play_id_kick"]).any()
    ):
        raise RuntimeError("Invalid made-FG-to-next-possession match.")
    return joined


def all_frozen_donor_snapshots(pending: dict, donors: pd.DataFrame) -> list[dict]:
    """Exact deterministic integration of the same ordinary 2025 transitions.

    Use every matched historical kickoff once; no reseeding, tuning, or
    realized future information from the game being predicted.
    """
    validate_frozen_pool(donors)
    if int(pending["qtr"]) != 4 or float(pending["is_kickoff"]) != 1:
        raise ValueError("Need a regulation Q4 kickoff-pending post-score state.")
    if float(pending.get("_score_change_original", 0)) != 3:
        raise ValueError("Requires made FG scoring transition.")
    output = []
    for donor in donors.itertuples(index=False):
        state = dict(pending)
        own = float(donor.own_yardline)
        elapsed = float(donor.elapsed_seconds)
        new_clock = max(0.0, float(pending["game_seconds_remaining"]) - elapsed)
        state.update({
            "is_kickoff": 0.0,
            "yardline_100": 100.0 - own,
            "down": 1.0,
            "ydstogo": min(10.0, 100.0 - own),
            "goal_to_go": float((100.0 - own) <= 10.0),
            "game_seconds_remaining": new_clock,
            "half_seconds_remaining": max(
                0.0, float(pending["half_seconds_remaining"]) - elapsed
            ),
            "final_five_minutes": float(new_clock <= 300.0),
            "_kickoff_transition_type": str(donor.transition_type),
        })
        output.append(state)
    if len(output) != len(donors):
        raise RuntimeError("Donor integration dropped historical outcomes.")
    return output


def exact_made_fg_decomposition(engine: dict, donors: pd.DataFrame,
                                example: dict) -> dict:
    """Attribute Q4 made-FG continuation difference, absent MC noise."""
    base = engine["normalize_state"](example)
    # Fixed five-second illustrative FG execution: isolates kickoff modeling,
    # not a Monte Carlo estimate of the entire fourth-down decision.
    pending = engine["make_wp_state"](
        base, possession_original=False,
        yardline_100=np.nan, down=np.nan, ydstogo=np.nan,
        elapsed_seconds=5.0, score_change_original=3.0, is_kickoff=True,
    )
    states = all_frozen_donor_snapshots(pending, donors)
    baseline = float(engine["evaluate_post_play_states"](base, [pending])[0])
    # 2024-derived/native 2025 kickoff feature encoding. This is a
    # diagnostic intermediate state, NOT a validated alternative forecast.
    aligned_kickoff = dict(pending)
    aligned_kickoff.update({
        "down": np.nan, "ydstogo": 0.0,
        "yardline_100": 35.0, "goal_to_go": 0.0,
    })
    aligned_wp = float(
        engine["evaluate_post_play_states"](base, [aligned_kickoff])[0]
    )
    simulated = np.asarray(
        engine["evaluate_post_play_states"](base, states), dtype=float
    )
    if len(simulated) != len(donors) or not np.isfinite(simulated).all():
        raise ValueError("Invalid modeled post-kickoff WP")
    tb = donors["transition_type"].eq("touchback").to_numpy(bool)
    if tb.sum() == 0 or (~tb).sum() == 0:
        raise RuntimeError("Missing kickoff type for decomposition")
    results = {
        "site": example["site"],
        "seconds_before_fg": float(example["game_seconds_remaining"]),
        "fg_elapsed_assumed": 5.0,
        "post_fg_margin_for_kicker": (
            float(example["score_differential"]) + 3
        ),
        "kickoff_pending_wp": baseline,
        "native_like_pending_wp": aligned_wp,
        "ordinary_kickoff_integrated_wp": float(simulated.mean()),
        # Sequential, path-dependent sensitivity attribution. NOT causal.
        "synthetic_to_native_encoding_difference": float(aligned_wp - baseline),
        "native_encoding_to_post_snap_difference": float(simulated.mean() - aligned_wp),
        "difference_integrated_minus_pending": float(simulated.mean() - baseline),
        "ordinary_donors": len(donors),
        "touchback_rate": float(tb.mean()),
        "touchback_wp": float(simulated[tb].mean()),
        "return_wp": float(simulated[~tb].mean()),
        "mean_receiver_own_yardline": float(donors["own_yardline"].mean()),
        "mean_kickoff_to_snap_seconds": float(donors["elapsed_seconds"].mean()),
        "year_of_donors": 2025,
    }
    reconstructed = (
        results["touchback_rate"] * results["touchback_wp"]
        + (1-results["touchback_rate"]) * results["return_wp"]
    )
    if abs(reconstructed-results["ordinary_kickoff_integrated_wp"]) > 1e-10:
        raise AssertionError("Kickoff outcome mixture failed sum identity.")
    if abs(
        results["synthetic_to_native_encoding_difference"]
        + results["native_encoding_to_post_snap_difference"]
        - results["difference_integrated_minus_pending"]
    ) > 1e-10:
        raise AssertionError("Feature/continuation sensitivity failed sum identity.")
    return results


def grouped_observed_scoring_calibration(
    name: str, rows: pd.DataFrame, *,
    cluster_ci,
) -> dict:
    """Only synthetic PENDING kickoff can be scored as a pre-kickoff forecast.

    The actual first snap is hindsight, so report it as a diagnostic only.
    Never infer what would have happened if the team had NOT kicked a FG.
    """
    n = len(rows)
    result = {"group": name, "n": n,
              "games": int(rows["game_id"].nunique())}
    if n == 0:
        result["status"] = "NO DATA"
        return result
    target = rows["actual_home_win"].to_numpy(float)
    pred = rows["p_kickoff_synthetic"].to_numpy(float)
    native = rows["p_kickoff_native"].to_numpy(float)
    hindsight = rows["p_first_snap"].to_numpy(float)
    if not np.isfinite(np.concatenate([target, pred, native, hindsight])).all():
        raise ValueError("Invalid observed WP scores.")
    residual = target - pred
    ci = cluster_ci(residual, rows["game_id"].to_numpy())
    # Paired scoring outcomes, same plays/games for native-like vs synthetic.
    # This estimates predictive accuracy only, NOT fourth-down policy quality.
    brier_gain = (target - pred) ** 2 - (target - native) ** 2
    gain_ci = cluster_ci(brier_gain, rows["game_id"].to_numpy())
    result.update({
        "status": "LOW SUPPORT" if n < 30 else "DESCRIPTIVE",
        "observed_home_win": float(target.mean()),
        "production_pending_wp": float(pred.mean()),
        "native_pending_wp": float(native.mean()),
        "hindsight_first_snap_wp": float(hindsight.mean()),
        "pending_brier": float(np.mean((target - pred)**2)),
        "native_pending_brier": float(np.mean((target - native)**2)),
        "native_vs_synthetic_brier_improvement": float(brier_gain.mean()),
        "native_brier_improvement_ci95": list(gain_ci),
        "hindsight_first_snap_brier": float(np.mean((target - hindsight)**2)),
        "calibration_gap_actual_minus_pending": float(residual.mean()),
        "calibration_gap_game_cluster_ci95": list(ci),
    })
    return result


def main() -> None:
    import os
    os.chdir(ROOT)
    print("=== STEP 72: POST-MADE-FG Q4 WIN PROBABILITY ===", flush=True)
    print("2024/25 completed REG only. No 2026; no production edits.")
    engine = import_script(ENGINE_PATH, "post_fg_frozen_engine")
    old = import_script(STEP61, "post_fg_2024_helpers")
    newer = import_script(STEP64, "post_fg_2025_helpers")
    pool = load_frozen_pool()

    example = exact_made_fg_decomposition(engine.__dict__, pool, RAMS_EXAMPLE)
    print("\nRAMS-STYLE LATE 24-YD MADE FG, AWAY TEAM DOWN 4:", flush=True)
    print(f"  PENDING WP: {example['kickoff_pending_wp']:.2%}", flush=True)
    print(f"  EXACT ORDINARY POST-KICKOFF MEAN WP: "
          f"{example['ordinary_kickoff_integrated_wp']:.2%}", flush=True)
    print(f"  NATIVE-LIKE PENDING KICKOFF WP: "
          f"{example['native_like_pending_wp']:.2%}", flush=True)
    print(f"  SEQUENTIAL EFFECT: feature representation "
          f"{100*example['synthetic_to_native_encoding_difference']:+.2f} pp, "
          f"kickoff-to-snap continuation "
          f"{100*example['native_encoding_to_post_snap_difference']:+.2f} pp",
          flush=True)
    print(f"  DELTA: {100*example['difference_integrated_minus_pending']:+.2f} pp",
          flush=True)
    print(f"  2025 donors n={example['ordinary_donors']}, "
          f"touchbacks={example['touchback_rate']:.1%}, "
          f"WP TB={example['touchback_wp']:.2%}, "
          f"WP return={example['return_wp']:.2%}", flush=True)

    report = {
        "status": "READ ONLY; no deployment",
        "2026_outcomes": "NOT ACCESSED",
        "2025": "previously inspected; exploratory",
        "donor_scope": "ordinary 2025 REG Q4, selected matchable outcomes",
        "excluded_outcomes": "onside, return TDs, unusual penalties",
        "rams_style_attribution": example,
        "actual_post_made_fg_kickoffs": {},
        "policy_claim": "NONE; unchosen decisions not observed",
    }
    for year in SEASONS:
        print(f"\n=== OBSERVED {year} MADE FG -> KICKOFF -> SNAP ===", flush=True)
        raw = nfl.load_pbp([year]).sort(["game_id", "play_id"]).to_pandas()
        pairs = actual_made_fg_kickoff_pairs(raw, year, old, newer)
        print(f"  Matched ordinary made-FG kickoffs: n={len(pairs)}", flush=True)
        if pairs.empty:
            report["actual_post_made_fg_kickoffs"][str(year)] = {
                "status": "NO ELIGIBLE PAIRS"
            }
            continue
        score = old.compare_actual_pair_predictions(
            pairs, engine.wp_model, list(engine.WP_FEATURES),
        )
        last = score["seconds"].between(40, 300)
        narrow = score["receiver_margin"].between(1, 3)
        group_masks = [
            ("ALL", np.ones(len(score), dtype=bool)),
            ("Q4 40-300 SEC", last),
            ("Q4 40-300 RECEIVER LEADS 1-3", last & narrow),
            ("Q4 40-300 RECEIVER LEADS 1-3 KICKER HOME",
             last & narrow & ~score["home_receiving"]),
            ("Q4 40-300 RECEIVER LEADS 1-3 KICKER AWAY",
             last & narrow & score["home_receiving"]),
        ]
        stats = [
            grouped_observed_scoring_calibration(name, score.loc[mask],
                cluster_ci=newer.game_cluster_ci)
            for name, mask in group_masks
        ]
        report["actual_post_made_fg_kickoffs"][str(year)] = stats
        for item in stats:
            if not item["n"]:
                print(f"  {item['group']}: n=0", flush=True)
                continue
            print(
                f"  {item['group']} n={item['n']} "
                f"games={item['games']} "
                f"observed_home={item['observed_home_win']:.1%} "
                f"pending_model={item['production_pending_wp']:.1%} "
                f"native_model={item['native_pending_wp']:.1%} "
                f"first_snap_hindsight={item['hindsight_first_snap_wp']:.1%} "
                f"Brier pending={item['pending_brier']:.5f} "
                f"native={item['native_pending_brier']:.5f} "
                f"native_gain={item['native_vs_synthetic_brier_improvement']:+.5f} "
                f"gain_CI={item['native_brier_improvement_ci95']} "
                f"gap_CI={item['calibration_gap_game_cluster_ci95']} "
                f"{item['status']}", flush=True,
            )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report: {REPORT_PATH.relative_to(ROOT)}", flush=True)
    print("No validation of counterfactual play decisions. No model promotion.")


if __name__ == "__main__":
    main()
