# Step 81 — Broader Q4 post-score kickoff WP reliability (2023–2025)

**Completed October 11, 2026.** The original production WP model was trained on 2014–2022 play-by-play. This is an **exploratory retrospective** comparison on previously examined 2023–2025 regular-season data. **No 2026 NFL outcomes were accessed.** The classifier and other model artifacts were not retrained, and this test does not establish the causal value of a field goal versus going for it.

[Step 81 analysis](../src/81_broader_post_score_kickoff_calibration.py) · [Regression tests](../tests/test_broader_q4_kickoff_calibration.py) · [Completed historical calibration](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38179882894) · [243 passing tests](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38179882915)

## Selection known before the kickoff

Use 2023, 2024 and 2025 REG Q4 kickoffs at the standard own-35 spot, with more than 40 seconds left, where the immediately preceding scoring action in the same game was a made FG, extra point or two-point attempt. Require that the original scoring team is not the receiver and that no intervening kickoff occurred. Exclude explicitly declared onside attempts and special kickoff spots. **Do not filter on future outcome, penalty, return touchdown or a matchable first snap.** Final home win is the scoring label only.

| Season | Selected post-score Q4 kicks | Games |
|---|---:|---:|
| 2023 | 525 | 253 |
| 2024 | 549 | 253 |
| 2025 | 543 | 256 |
| **Pooled** | **1,617** | **762** |

Event sources: 599 made FG, 834 after-extra-point and 184 after-two-point kickoffs.

## Comparison

The unchanged original model sees a post-score synthetic kickoff feature vector with down, ydstogo and yardline_100 missing, goal_to_go=0. The Step 80 corrected candidate changes **only two values**: ydstogo=0, yardline_100=35. Score, clock, timeouts, home/away axis and 2014–22-trained classifier stay fixed.

Paired **home-win Brier score** is evaluated against the actual eventual game winner. Lower is better. Gain means original Brier minus candidate Brier. The overall and yearly primary 95% confidence intervals resample whole games, 1,500 bootstrap replicates; all finer subgroups below are descriptive only.

| Sample | Kicks | Original Brier | Candidate Brier | Gain | 95% game-cluster CI for gain |
|---|---:|---:|---:|---:|---|
| **2023–2025 combined** | **1,617** | **0.11561** | **0.10743** | **+0.00818** | **[+0.00264, +0.01398]** |
| 2023 | 525 | 0.11131 | 0.10272 | +0.00859 | [-0.00094, +0.01827] |
| 2024 | 549 | 0.10148 | 0.09282 | +0.00867 | [-0.00076, +0.01747] |
| 2025 | 543 | 0.13406 | 0.12677 | +0.00729 | [-0.00211, +0.01688] |

The combined loss is about **7.1% lower** relative to the old Brier. The combined bootstrap interval excludes zero, but **each individual season CI includes zero**. Because these historical seasons were previously examined, especially 2025, the result is not a pristine prospective test.

## Outcome breakdown

| Subgroup | Kicks | Original Brier | Candidate Brier | Gain |
|---|---:|---:|---:|---:|
| Made FG | 599 | 0.10693 | 0.10085 | +0.00608 |
| After extra point | 834 | 0.10789 | 0.09984 | +0.00805 |
| After two-point attempt | 184 | 0.17889 | 0.16330 | +0.01560 |
| **Kicking team HOME** | **851** | **0.12365** | **0.10704** | **+0.01661** |
| **Kicking team AWAY** | **766** | **0.10668** | **0.10787** | **-0.00119** |
| Last five minutes, >40 seconds | 541 | 0.11823 | 0.10400 | +0.01423 |
| 5–10 minutes remaining | 500 | 0.11365 | 0.10348 | +0.01017 |
| 10–15 minutes remaining | 576 | 0.11486 | 0.11409 | +0.00077 |
| Receiver trailing 1–3 | 169 | 0.25620 | 0.24216 | +0.01404 |
| Receiver tied | 98 | 0.30722 | 0.24843 | +0.05879 |
| Receiver leading 1–3 | 118 | 0.21444 | 0.20087 | +0.01357 |

**Important counterexample:** The candidate performs slightly **worse for away-kicking teams** by Brier point estimate. This subgroup has no independent site-specific CI or multiple-testing correction. The tied-score improvement is descriptive for only 98 kickoffs after prior research had already identified site/tie anomalies; it is not a basis for an additional tuned correction.

## Engineering and promotion decision

[The automated workflow](../.github/workflows/validate_broad_q4_kickoff_encoding.yml) downloads only the 2023–2025 seasons, recomputes the prespecified cohort, scores both representations, validates paired Brier and archives a detailed JSON/CSV. Tests explicitly protect scoring association, receiving-team orientation, absence of intervening re-kicks, and retention of penalty and return-TD outcomes. All **243 project regressions passed** at completion.

This is better reliability evidence for the kickoff-encoding candidate than the narrow 14-state stress test, and strongly supports a structural issue in the original kickoff-pending representation. **It does not prove the complete fourth-down policy is correct.** Actual outcomes of completed post-score kickoffs do not reveal the outcome of a field goal that was never attempted, or of a hypothetical GO attempt.

**No automatic production promotion.** The original model, official fourth-down recommendation, and reserved prospective 2026 evaluation remain unchanged. The next safe product improvement is offering the corrected native-encoding model as a clearly labeled optional research comparison, without replacing the default.
