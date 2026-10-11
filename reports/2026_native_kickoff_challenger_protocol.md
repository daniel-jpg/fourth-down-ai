# Step 83: Frozen 2026 native kickoff-input challenger

**Registered October 11, 2026, before the Step 66 forward threshold.**
**Status:** prospective design and synthetic tests only; no real 2026 W6+ outcomes have been scored for the companion. This is a preregistered protocol, **not** an evaluation result.

## Purpose and existing study

The frozen original Step 66 kickoff evaluation already compares synthetic kickoff inputs with native kickoff inputs. Step 83 prepares a **separate, explicit native-versus-legacy Brier evaluation with home/away subgroups**, using exactly the original Step 66 eligible game sample. It leaves Step 66 code, schedule, criterion, and output completely untouched.

The native encoding changes precisely two input fields: yardline_100 becomes 35 rather than missing, and ydstogo becomes 0 rather than missing. Down stays missing and goal_to_go stays 0. All scores, timeouts, possession, WP coefficients and other model inputs are identical. This correction improved previously explored 2023-25 pooled Brier reliability but *slightly worsened the away-kicking-team subgroup*, so a separate prospective site check is mandatory.

## Exact no-peeking and one-time gates

1. The original [Step 66 workflow](../.github/workflows/forward_2026_validation_once.yml) must reach at least **100 distinct eligible completed 2026 Weeks 6–18 games** and commit its original once-only research result at reports/2026_forward_kickoff_wp_final.txt.
2. Only after that workflow finishes successfully can the [new companion workflow](../.github/workflows/forward_2026_native_challenger_once.yml) consider scoring. It is triggered by the original workflow's successful completion, **not by a schedule or a manual dispatch**. Weekly incomplete checks skip before loading NFL data.
3. The new scorer's frozen Git blob is **21074cfd74710e738a7b7cd8c4f9ef4cadbfd402**, pinned in the companion workflow. The original Step 66 source, feature helper and native correction source are also pinned. Step 66 enforces the WP model and feature-spec blobs. Any drift fails closed before the real 2026 play-by-play read.
4. The scorer independently rechecks the original once-only report, an absent Step 83 completion marker, the original cohort's Week 6–18 constraints and the **100-distinct-game threshold before computing a forecast**.
5. The companion then commits exactly one immutable summary at reports/2026_native_kickoff_challenger_final.json. Subsequent successful weekly checks take the **skip path**. If the original gate does not complete before its already scheduled cutoff, this companion does not evaluate.

The existing Step 66 source and automatic gate are unchanged. The companion tests use fabricated games only; no real 2026 W6+ outcome access is required to develop or run them.

## Fixed cohort, primary score, and uncertainty

**Cohort:** Same Step 66 ordinary, score-neutral Q4 standard-own-35 matched kickoff/first receiving scrimmage snap cohort, with >40 seconds at kickoff, 2026 REG Weeks 6–18, completed non-tied final game. It excludes onside plays, penalty-marked kicks, return touchdowns, re-kicks and unmatched plays. This cohort is narrower than the broader 2023–25 all-outcomes post-score audit.

**At-kickoff predictors:** Synthetic baseline (down, ydstogo, yardline_100 all missing, goal_to_go=0) versus native-like challenger (down missing, ydstogo=0, yardline_100=35, goal_to_go=0). The final home-game win is the **outcome label only**. The actual return spot and first-snap clock never enter either forecast. Both models use the same frozen WP classifier and all other state fields.

**Primary paired outcome:** Brier gain = mean[(actual_home_win - synthetic_wp)^2 - (actual_home_win - native_wp)^2]. A positive number means the native candidate improves Brier loss. Confidence interval: **95% game-cluster percentile bootstrap, 2,000 replicates, seed 83**, resampling whole games rather than individual kickoffs.

**Predeclared descriptive groups:** overall, original kicker HOME, original kicker AWAY, final five minutes (>40 seconds), and 5–15 minutes remaining. Report each group's native and synthetic Brier, log loss, mean home WP, number of kickoffs, number of games, paired Brier gain and game-cluster interval. No tuned per-site correction is allowed.

**Strict research-screen criterion:** Preliminary support is reported only when the overall **lower 95% interval bound exceeds zero**, BOTH home and away kicker groups have **at least 20 distinct games**, and BOTH site groups have **nonnegative** Brier gains. Otherwise label the result **inconclusive/negative**. A passing result is not permission to promote the official engine; it establishes only selected prospective kickoff reliability.

## Important interpretation constraints

These game outcomes are observational. They cannot identify the counterfactual result of a fourth-down FG that was never tried or a hypothetical GO attempt. A favorable kickoff Brier score is not a causal fourth-down decision advantage, nor is its game bootstrap interval a measure of model-selection or coaching-policy uncertainty.

**No production model, Streamlit default, action probabilities, or pre-existing Step 66 research gate will be changed by this companion.**

Implementation: [fixed scorer](../src/83_forward_2026_native_challenger_once.py), [synthetic-only tests](../tests/test_forward_2026_native_challenger.py), and [gated once-only workflow](../.github/workflows/forward_2026_native_challenger_once.yml).
