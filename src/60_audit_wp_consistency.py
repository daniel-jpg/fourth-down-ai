"""Read-only win-probability consistency audit for fourth-down continuations.

Run from the repository root:
    ./.venv/bin/python src/60_audit_wp_consistency.py

Checks all HOME/AWAY, three Q4 clocks and three score situations. Each case
models a MADE FG (+3) after a five-second FG snap, with the opposing team
receiving a normal kickoff. Compare raw kickoff-pending WP to an EXACT
weighted average of post-kickoff receiving-team possession WPs, using 2024 Q4
ordinary return donors and hypothetical 2025+ touchback rates of 20/30/40%.

Also flags field-position reversals as the receiving team advances upfield,
holding possession, score, clock, timeouts and down/distance constant.

No 2025 or 2026 feed is loaded. No data/models/production code is changed.
These are synthetic consistency stress tests, not factual outcome forecasts.
"""

from pathlib import Path
import os
import runpy
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wp_consistency_audit import (  # noqa: E402
    RECEIVER_OWN_GRID, TOUCHBACK_RATE_SCENARIOS,
    validate_2024_q4_donors, evaluate_made_fg_case,
)


def main() -> None:
    os.chdir(ROOT)  # The existing engine loads artifact paths relative to root.
    donor_path = ROOT / "data/kickoff_2024_dev_transition_pool.parquet"
    if not donor_path.is_file():
        raise SystemExit("Missing frozen 2024 donor pool; run src/53_build_kickoff_transition_pool.py")
    donors = validate_2024_q4_donors(pd.read_parquet(donor_path))
    print("\n=== WIN-PROBABILITY CONSISTENCY AUDIT (READ ONLY) ===")
    print(f"2024 Q4 donor sample: {donors.touchback_count} touchbacks, "
          f"{donors.return_count} returns; among touchbacks, "
          f"own-20 share={donors.touchback_own20_probability:.1%}")
    print("Touchback-rate scenarios (not forecasts): 20%, 30%, 40%")
    print("Return field positions + clock costs: empirical 2024 Q4 ordinary returns")
    print("2025+ touchback rules: direct end-zone -> own 35; landing-zone -> own 20")
    print("\nLoading frozen production decision engine...")
    engine = runpy.run_path(str(ROOT / "src/42_build_decision_engine.py"))
    results = []
    for site in ("AWAY", "HOME"):
        for seconds in (120, 299, 600):
            for diff in (-4, -2, 3):
                raw_state = {
                    "qtr": 4, "game_seconds_remaining": seconds,
                    "yardline_100": 6, "ydstogo": 4,
                    "score_differential": diff,
                    "posteam_timeouts_remaining": 3,
                    "defteam_timeouts_remaining": 3,
                    "site": site, "roof": "outdoors",
                }
                result = evaluate_made_fg_case(
                    engine, engine["normalize_state"](raw_state), donors,
                    touchback_rates=TOUCHBACK_RATE_SCENARIOS,
                )
                results.append(result)

    print("\n=== KICKOFF-PENDING VS ACTUAL POSSESSION-STATE WP ===")
    print("All values below are from the ORIGINAL scoring team's perspective.")
    print("Each case: made FG (+3); five-second FG snap; then ordinary kickoff.")
    print("Delta = pending WP - weighted post-kickoff WP, in percentage points.")
    print("Site   Time  ScoreBefore  Pending   Mix20%   Mix30%   Mix40%   Delta30  FieldWarn")
    for r in results:
        mix = r["mixed_wp"]
        delta = 100 * (r["kickoff_pending_wp"] - mix[.30])
        print(f"{r['site']:5s} {r['time_seconds']:5d} "
              f"{r['pre_fg_score_diff']:+11.0f} "
              f"{100*r['kickoff_pending_wp']:8.2f}% "
              f"{100*mix[.20]:7.2f}% "
              f"{100*mix[.30]:7.2f}% "
              f"{100*mix[.40]:7.2f}% "
              f"{delta:+8.2f} "
              f"{len(r['reversals']):9d}")

    outliers = sorted(results, key=lambda r: abs(r["kickoff_pending_wp"] - r["mixed_wp"][.30]), reverse=True)
    more_than_five = [r for r in results if abs(100*(r["kickoff_pending_wp"] - r["mixed_wp"][.30])) > 5]
    print("\n=== INCONSISTENCY SUMMARY ===")
    print(f"Situations audited: {len(results)}")
    print(f"Abs(pending - Mix30%) > 5 percentage points: {len(more_than_five)}")
    print(f"Situations with receiving-yardline monotonicity warning >0.5 pp: "
          f"{sum(bool(r['reversals']) for r in results)}")
    print("Largest kickoff-pending disagreements (top five):")
    for r in outliers[:5]:
        delta = 100 * (r["kickoff_pending_wp"] - r["mixed_wp"][.30])
        print(f"  {r['site']} Q4 {r['time_seconds']}s before FG margin "
              f"{r['pre_fg_score_diff']:+.0f}: pending-minus-continuation={delta:+.2f} pp")

    rams = next(r for r in results if r["site"] == "AWAY" and r["time_seconds"] == 299
                and r["pre_fg_score_diff"] == -4)
    print("\n=== RAMS CASE: OPPONENT FIRST-DOWN FIELD-POSITION SENSITIVITY ===")
    print("Rams made FG, now trailing by 1. Seattle takes possession.")
    print("Receiving team own yard  |  Original Rams WP (fixed post-FG clock)")
    for own, wp in zip(rams["field_positions"], rams["field_wps"]):
        print(f"{int(own):22d}  |  {100*wp:6.2f}%")
    print("Material monotonicity reversals (>0.5 percentage points):")
    if not rams["reversals"]:
        print("  None found at sampled receiving-yardlines.")
    else:
        for item in rams["reversals"]:
            print(f"  Seattle own {item['from_own']:.0f} to "
                  f"{item['to_own']:.0f}: Rams WP increases "
                  f"{item['kicker_wp_increase_pp']:+.2f} pp")

    print("\nNO PRODUCTION CHANGE. This audit does not demonstrate actual WP accuracy.")
    print("Unmodeled kickoff events: onside recoveries, return TDs and unusual penalties.")
    print("Synthetic states use a fixed five-second made-FG cost and 2024 return donors.")
    print("Next: prioritize large pending/continuation inconsistencies and field reversals.")


if __name__ == "__main__":
    main()
