"""Run an OPT-IN kickoff scenario against unchanged production decision logic.

From the repository root:
  ./.venv/bin/python src/54_compare_kickoff_scenarios.py --simulations 2400

No files are written; the default production application remains untouched.
All return donors are 2024 development data. Rates are user-supplied what-ifs,
NOT probabilities fitted on the held-out 2025 season.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kickoff_scenario import use_q4_kickoff_scenario  # noqa: E402


SCENARIOS = {
    "rams": {
        "qtr": 4, "game_seconds_remaining": 299,
        "yardline_100": 6, "ydstogo": 4, "score_differential": -4,
        "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
        "site": "AWAY", "roof": "outdoors",
    },
    "late_fg": {
        "qtr": 4, "game_seconds_remaining": 10,
        "yardline_100": 25, "ydstogo": 10, "score_differential": -2,
        "posteam_timeouts_remaining": 1, "defteam_timeouts_remaining": 1,
        "site": "HOME", "roof": "outdoors",
    },
    "fourth_three": {
        "qtr": 4, "game_seconds_remaining": 458,
        "yardline_100": 30, "ydstogo": 3, "score_differential": -3,
        "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 2,
        "site": "AWAY", "roof": "outdoors",
    },
    "goal_line": {
        "qtr": 4, "game_seconds_remaining": 45,
        "yardline_100": 1, "ydstogo": 1, "score_differential": -4,
        "posteam_timeouts_remaining": 2, "defteam_timeouts_remaining": 2,
        "site": "AWAY", "roof": "outdoors",
    },
}


def score(result, action):
    rows = result["results"]
    found = rows.loc[rows["action"].eq(action) & rows["eligible"]]
    return None if found.empty else float(found.iloc[0]["win_probability_pct"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="rams")
    parser.add_argument("--simulations", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kickoff-seed", type=int, default=2026)
    parser.add_argument("--rules-year", type=int, choices=[2024, 2025, 2026], default=2026)
    args = parser.parse_args()

    if args.simulations < 200:
        parser.error("Use at least 200 simulations; 2400+ recommended")

    import os
    os.chdir(ROOT)  # Existing model paths are repository-relative.

    pool_path = ROOT / "data" / "kickoff_2024_dev_transition_pool.parquet"
    if not pool_path.is_file():
        raise SystemExit(
            "Missing 2024 donor pool. Run:\n"
            "./.venv/bin/python src/53_build_kickoff_transition_pool.py"
        )
    pool = pd.read_parquet(pool_path)
    engine = runpy.run_path(str(ROOT / "src" / "42_build_decision_engine.py"))

    state = SCENARIOS[args.scenario]
    def recommend():
        return engine["recommend"](
            state, n_simulations=args.simulations, seed=args.seed
        )

    print(f"\n=== CASE: {args.scenario} ===")
    print("2024 development-only return donors; scenario touchback rates, not calibrated")
    print(f"Simulations per action: {args.simulations}, seed: {args.seed}")
    print(f"Rules year: {args.rules_year}")
    print("\n=== ORIGINAL PRODUCTION BASELINE ===", flush=True)
    baseline = recommend()
    print(
        f"FG={score(baseline, 'FIELD_GOAL')}% "
        f"PASS={score(baseline, 'NORMAL_GO_PASS')}% "
        f"RUN={score(baseline, 'NORMAL_GO_RUN')}% "
        f"RECOMMENDATION={baseline['recommendation']}",
        flush=True,
    )

    print("\n=== OPT-IN POST-KICKOFF CONTINUATION ===", flush=True)
    for rate in (0.20, 0.25, 0.35, 0.50, 0.65):
        # Identical action outcomes from `args.seed` in every case.
        # Same kickoff uniforms from `args.kickoff_seed` at every rate.
        with use_q4_kickoff_scenario(
            engine, pool,
            touchback_rate=rate,
            seed=args.kickoff_seed,
            rules_year=args.rules_year,
        ):
            corrected = recommend()
        values = [
            score(corrected, "FIELD_GOAL"),
            score(corrected, "NORMAL_GO_PASS"),
            score(corrected, "NORMAL_GO_RUN"),
        ]
        print(
            f"Touchbacks={rate:.0%}  "
            f"FG={values[0]}%  PASS={values[1]}%  "
            f"RUN={values[2]}%  RECOMMENDATION={corrected['recommendation']}",
            flush=True,
        )
    print(
        "\nNOTE: Rare kickoff outcomes omitted; 2024 returns may differ from "
        "2025+. This is NOT a validated production correction.",
        flush=True,
    )


if __name__ == "__main__":
    main()
