"""Read-only test of the TRAIN-ALIGNED standard-kickoff feature encoding.

The currently loaded WP model learned nflverse kickoff rows with
  down=NaN, ydstogo=0, yardline_100=35, goal_to_go=0
for ordinary kicks from the 35. The existing engine predicts pending
kickoffs with yardline_100 and ydstogo as NaN. This opt-in experiment:
  (1) checks synthetic vs historical-style kickoff predictions on 2024
      ordinary Q4 pairs, grouped by kicker HOME/AWAY;
  (2) compares the SAME 4th-down action draws for production,
      train-aligned kickoff encoding, explicitly resolved kickoff
      continuation and BOTH changes together.

Nothing is saved. No engine code or model is modified.
2024 is validation/development data already inspected in prior diagnostics;
not a new independent test. Touchback 30% is a WHAT-IF, not a forecast.

Usage:
  ./.venv/bin/python src/63_compare_kickoff_encoding.py
  ./.venv/bin/python src/63_compare_kickoff_encoding.py --scenario rams_home_mirror
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

import nflreadpy as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kickoff_encoding_candidate import use_native_like_q4_kickoff_wp  # noqa: E402
from src.kickoff_scenario import use_q4_kickoff_scenario  # noqa: E402

SCENARIOS_SCRIPT = ROOT / "src/54_compare_kickoff_scenarios.py"
OBSERVED_SCRIPT = ROOT / "src/61_audit_observed_kickoff_wp.py"

TOUCHBACK_RATE = 0.30  # Hypothetical 2025+ Q4 rate, never fitted here.


def grouped_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float))**2))


def paired_game_bootstrap(delta: np.ndarray, game_ids: np.ndarray,
                          *, repeats: int = 1500, seed: int = 63) -> tuple[float, float]:
    """Paired, game-clustered bootstrap for Brier improvement."""
    ids = np.asarray(game_ids)
    errors = np.asarray(delta, dtype=float)
    if ids.shape != errors.shape or not len(ids) or not np.isfinite(errors).all():
        raise ValueError("Invalid game-clustered Brier differences.")
    games = pd.unique(ids)
    by_game = [np.flatnonzero(ids == game) for game in games]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(games), size=len(games))
        indices = np.concatenate([by_game[i] for i in chosen])
        draws.append(float(errors[indices].mean()))
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def observed_2024_validation(model, features: list[str]) -> None:
    helpers = runpy.run_path(str(OBSERVED_SCRIPT))
    pbp = nfl.load_pbp([2024]).sort(["game_id", "play_id"]).to_pandas()
    pairs = helpers["pair_2024_q4"](pbp)
    native = helpers["feature_frame"](pairs, "kick")
    target = pd.to_numeric(pairs["result_kick"], errors="coerce").gt(0).to_numpy(float)
    if not np.isfinite(pd.to_numeric(pairs["result_kick"], errors="coerce")).all():
        raise ValueError("Missing final outcomes.")
    predict = helpers["model_probabilities"]

    engine_style = native.copy()
    for col in ("down", "ydstogo", "yardline_100"):
        engine_style[col] = np.nan
    engine_style["goal_to_go"] = 0.0

    candidate_style = engine_style.copy()
    candidate_style["ydstogo"] = 0.0
    candidate_style["yardline_100"] = 35.0

    raw_p = predict(model, features, native)
    production_p = predict(model, features, engine_style)
    candidate_p = predict(model, features, candidate_style)
    spot35 = native["yardline_100"].eq(35.).to_numpy(dtype=bool)
    if not spot35.any():
        raise ValueError("No standard-kickoff evaluation rows.")
    # Standard kickoff candidate should agree EXACTLY with nflverse encoding.
    matching = np.max(np.abs(candidate_p[spot35] - raw_p[spot35]))
    print("\n=== REAL 2024 Q4 VALIDATION OF ENCODING (NOT PRODUCTION) ===")
    print(f"Pairs: {len(pairs)}; standard kick spot own35: "
          f"{spot35.sum()}; other spots: {(~spot35).sum()}")
    print(f"Max(abs(candidate - native)) on standard-35 kicks: {matching:.8f}")
    if matching > 1e-9:
        raise AssertionError("Candidate does not reproduce native standard kickoff features")

    home_recv = native["is_home_posteam"].to_numpy(dtype=bool)
    for label, m in (
        ("STANDARD-35 ONLY", spot35),
        ("STANDARD-35, KICKER HOME", spot35 & ~home_recv),
        ("STANDARD-35, KICKER AWAY", spot35 & home_recv),
        ("STANDARD-35, LAST 5 MIN, KICKER HOME",
         spot35 & ~home_recv &
         native["game_seconds_remaining"].le(300).to_numpy()),
        ("STANDARD-35, LAST 5 MIN, KICKER AWAY",
         spot35 & home_recv &
         native["game_seconds_remaining"].le(300).to_numpy()),
    ):
        if not m.any():
            continue
        y = target[m]
        baseline_brier = grouped_brier(y, production_p[m])
        candidate_brier = grouped_brier(y, candidate_p[m])
        brier_delta = (y - production_p[m])**2 - (y - candidate_p[m])**2
        ci_lo, ci_hi = paired_game_bootstrap(
            brier_delta, pairs.loc[m, "game_id"].to_numpy(), repeats=1000,
        )
        print(f"{label} (n={m.sum()} games={pairs.loc[m, 'game_id'].nunique()}): "
              f"actual home wins {100*y.mean():.2f}%")
        print(f"  Pred home WP: current {100*production_p[m].mean():.2f}% "
              f"train-aligned {100*candidate_p[m].mean():.2f}%")
        print(f"  Brier: current {baseline_brier:.5f} "
              f"train-aligned {candidate_brier:.5f}; "
              f"improvement {baseline_brier-candidate_brier:+.5f} "
              f"game-bootstrap 95% [{ci_lo:+.5f}, {ci_hi:+.5f}]")


def values(result) -> dict:
    rows = result["results"]
    return {
        a: (float(rows.loc[rows["action"].eq(a) & rows["eligible"],
                            "win_probability_pct"].iloc[0])
            if not rows.loc[rows["action"].eq(a) & rows["eligible"]].empty
            else None)
        for a in ("FIELD_GOAL", "NORMAL_GO_PASS", "NORMAL_GO_RUN")
    }


def print_action(name, result) -> None:
    v = values(result)
    def fmt(x):
        return "NA" if x is None else f"{x:.2f}%"
    print(f"{name:28s} FG={fmt(v['FIELD_GOAL']):>7s} "
          f"PASS={fmt(v['NORMAL_GO_PASS']):>7s} "
          f"RUN={fmt(v['NORMAL_GO_RUN']):>7s} "
          f"Winner={result['recommendation']}")


def decision_sensitivity(engine: dict, scenario_name: str,
                         *, simulations: int, seed: int) -> None:
    scenario_script = runpy.run_path(str(SCENARIOS_SCRIPT))
    base = scenario_script["SCENARIOS"]["rams"] if scenario_name == "rams_home_mirror" \
        else scenario_script["SCENARIOS"][scenario_name]
    state = dict(base)
    if scenario_name == "rams_home_mirror":
        state["site"] = "HOME"

    pool_path = ROOT / "data/kickoff_2024_dev_transition_pool.parquet"
    if not pool_path.exists():
        raise FileNotFoundError(f"Required frozen donor pool missing: {pool_path}")
    donor = pd.read_parquet(pool_path)
    def rec():
        return engine["recommend"](state, n_simulations=simulations, seed=seed)

    print(f"\n=== CASE {scenario_name.upper()}: {simulations} SIMULATIONS PER ACTION ===")
    print("Same action-outcome seed in every run; 30% touchbacks is an assumed scenario.")
    print_action("Unchanged production", rec())
    with use_native_like_q4_kickoff_wp(engine):
        print_action("Train-aligned kickoff only", rec())
    with use_q4_kickoff_scenario(
        engine, donor, touchback_rate=TOUCHBACK_RATE, seed=2026, rules_year=2026,
    ):
        resolved = rec()
        print_action("Post-kickoff resolution only", resolved)
    with use_native_like_q4_kickoff_wp(engine):
        with use_q4_kickoff_scenario(
            engine, donor, touchback_rate=TOUCHBACK_RATE, seed=2026, rules_year=2026,
        ):
            both = rec()
            print_action("Encoding + resolution", both)
    v_a = values(resolved)
    v_b = values(both)
    print("Effect of adding encoding AFTER kickoff resolution, percentage points:")
    for action in ("FIELD_GOAL", "NORMAL_GO_PASS", "NORMAL_GO_RUN"):
        if v_a[action] is not None and v_b[action] is not None:
            print(f"  {action}: {v_b[action]-v_a[action]:+.5f} pp")
    print("Nonzero residual can indicate an unresolved kickoff-pending path.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=("rams", "rams_home_mirror", "late_fg",
                              "fourth_three", "goal_line"),
        default="rams",
    )
    parser.add_argument("--simulations", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.simulations < 200:
        parser.error("At least 200 simulations per action required.")

    os.chdir(ROOT)
    print("=== STEP 63: TRAIN-ALIGNED KICKOFF ENCODING EXPERIMENT ===")
    print("Frozen model; read-only; no production changes; 2025/2026 PBP not loaded.")
    engine = runpy.run_path(str(ROOT / "src/42_build_decision_engine.py"))
    observed_2024_validation(engine["wp_model"], list(engine["WP_FEATURES"]))
    decision_sensitivity(engine, args.scenario, simulations=args.simulations,
                         seed=args.seed)
    print("\nNo models, data, engine, or app changed.")
    print("Historical standard-35 kickoff metadata is not receiver field position.")
    print("2024 observational Brier does not validate 2025+ kickoff outcomes.")


if __name__ == "__main__":
    main()
