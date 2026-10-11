# Step 72 — post-made-field-goal kickoff WP reliability (October 11, 2026)

This is a **completed, read-only** diagnostic of the existing regulation win-probability engine. It does not retrain or change the Streamlit app or production model. No 2026 games or results were accessed.

Code: [src/72_audit_post_made_fg_wp.py](../src/72_audit_post_made_fg_wp.py) · Tests: [tests/test_post_made_fg_wp.py](../tests/test_post_made_fg_wp.py) · [Completed automated audit](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38173258057) · [157 passing regression tests](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38173258063)

## Why the late Rams example changes so much

In an illustrative **Rams-away fourth-and-4 from opponent 6, trailing 27–31 with 4:59 in Q4**, hold a **successful 24-yard field goal** fixed, with five seconds of elapsed time. Calculate the Rams' WP using the same frozen production WP model for each representation:

| Sequential diagnostic state after the made FG | WP |
|---|---:|
| Synthetic kickoff-pending features in the original engine (down/ydstogo/yardline missing) | 42.44% |
| Native-like kickoff encoding (kickoff flag, down missing, ydstogo=0, yardline_100=35, goal_to_go=0) | 37.43% |
| Exact arithmetic mean after all 488 frozen ordinary 2025 Q4 kickoffs, transitioned to the receiver's first scrimmage state | 34.46% |

**Total shift: -7.98 percentage points**. Sequential sensitivity split: **-5.01 pp** due to replacing the synthetic kickoff feature representation, then **-2.97 pp** due to transitioning to first-snap states. This decomposition is *path-dependent, not uniquely causal*. It does not test different field-goal execution clocks, kicks missed/blocked or subsequent scoring attempts.

The fixed 2025 Q4 ordinary donor pool has 488 matched kickoffs from 251 games, with touchback share **32.0%**. The same WP model yields a **32.52%** Rams WP conditional on the observed touchbacks and **35.38%** conditional on observed returns. These conditional averages exactly reconstruct the **34.46%** overall mean. The computation is deterministic: **no Monte Carlo noise**.

These historical donors are descriptive and are **not** a validated forecast for an NFL postseason game. They omit onside attempts and recoveries, return touchdowns, unusual penalties and cases without a matchable first snap.

The 42.44% and 34.46% are **conditional on a successful field goal**. They are deliberately not the exact fourth-down **action Q-values** of 41.57% and 34.11%, which average all FG outcomes and execution clock possibilities.

## Evaluation on observed post-made-FG possessions

Selected only **real completed 2024/25 regular-season Q4 kickoffs** immediately following a made field goal, and only if the ordinary kickoff matches a receiving-team first scrimmage snap with an unchanged score. The original synthetic pending prediction, native PBP kickoff-feature prediction and *hindsight* actual-first-snap prediction all use the **same frozen 2014–22-trained WP model**.

Brier is measured against the actual final game winner for the **chosen made-field-goal possessions**. Hindsight snap WP cannot be used as a pre-kickoff forecast. Both 2024 and 2025 are examined development/evaluation periods; 2025 had already been analyzed in earlier project experiments.

| Season and observed kickoff subset | Plays | Current pending Brier | Native-like kickoff Brier | Gain (lower is better) |
|---|---:|---:|---:|---:|
| 2024 Q4, all matched made-FG kickoffs | 208 | 0.10786 | **0.10494** | +0.00292 |
| 2025 Q4, all matched made-FG kickoffs | 184 | 0.09814 | **0.09629** | +0.00184 |
| 2024 Q4, 40–300 sec remaining | 72 | 0.11744 | **0.10675** | +0.01068 |
| 2025 Q4, 40–300 sec remaining | 58 | 0.08206 | **0.07431** | +0.00776 |

The **2024 late-Q4** paired Brier improvement had game-cluster 95% CI **[-0.01109, +0.03608]**; **2025 late-Q4** CI **[-0.01247, +0.03338]**. Both contain zero. Similarly, the all-Q4 intervals include zero. These exploratory subgroup comparisons are not multiple-testing-adjusted.

Critically, restricting to **Q4 40–300 seconds, receiving team leading by 1–3, after made FG** yields only **one 2024 play** and **two 2025 plays**. Only one 2024 case had the kicker at home; two 2025 cases had the kicker away. Such tiny groups cannot support meaningful game-context validation, and their bootstrap intervals are not trustworthy indicators of generalization.

## Findings and decision

1. The difference between the legacy kickoff-pending WP and ordinary-kickoff-resolved WP is **reproducible and structural**, not simulation noise.
2. The largest part of that difference is associated with how the **kickoff-pending feature vector** is encoded. A native-like representation removes about five WP points from the Rams-style made-FG state; that is an informative diagnostic, not yet a safe blanket fix.
3. Native-like kickoff encoding shows better Brier point estimates on observed 2024 and 2025 post-made-FG Q4 plays, but **uncertainty includes no improvement**. The rare, directly comparable close-game group is far too small.
4. Neither analysis establishes causal FG-vs-GO policy quality or proves whether 42.44% or 34.46% is the true probability.

**Decision: DO NOT PROMOTE.** Keep the frozen official engine, independent observed-2025 experimental comparison, and model-sensitivity warning. A future version of the kickoff continuation must be validated with adequate independent data and special-kickoff branches, including important rare outcomes. Preserve the original 2026 Weeks 6–18 prospective study and its 100-game evaluation gate; do not alter or peek at it for this experiment.

No local actions required; no production weights, outputs, or API contracts were changed by Step 72.
