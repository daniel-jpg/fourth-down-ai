# Fourth-down component reliability audit — 2026-10-11

## Method and provenance

These two development-only evaluations used frozen production event models. They measure observed FG makes and conversions on the actions NFL teams actually took, **not** the outcomes of counterfactual FG-vs-GO choices.

- [Step 69 observed-action calibration](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38171564181)
- [Step 70 fixed Platt calibration](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38171801935)
- [147 passing tests](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38171801854)
- Frozen fourth-down dataset from 2023–25; no 2026 data accessed.
- 2023–24 are development data; previously examined 2025 is exploratory, **not** a pristine independent holdout.
- Step 69 scored 5,605 of 5,640 input rows, excluding 35 invalid states.
- Calibration differences are actual minus predicted. The 95% game-level bootstrap intervals are not adjusted for multiple subgroup comparisons.

## Observed production event-model calibration

| Years | Context | n | Actual | Predicted |
|---|---|---:|---:|---:|
| 2023–24 | FG 18–29 yards | 489 | 98.0% | 97.7% |
| 2025 | FG 18–29 yards | 211 | 98.1% | 97.6% |
| 2023–24 | FG 50+ yards | 456 | 71.5% | 64.3% |
| 2025 | FG 50+ yards | 235 | 70.2% | 62.5% |
| 2025 | All field goals | 1,017 | 86.2% | 83.1% |
| 2025 | FG Q4 final five minutes | 110 | 80.0% | 81.5% |
| 2023–24 | 4th-and-4 passes | 106 | 48.1% | 48.2% |
| 2025 | 4th-and-4 passes | 77 | 59.7% | 47.7% |
| 2023–24 | 4th-and-4 passes at opponent 1–10 | 18 | 50.0% | 41.4% |
| 2025 | 4th-and-4 passes at opponent 1–10 | 13 | 69.2% | 39.3% |

The 50+ yard FG underprediction appears in both periods: 2023–24 actual-minus-predicted bias +7.2 percentage points, 95% game-cluster CI [+2.6,+11.3]; 2025 +7.7 percentage points, CI [+1.6,+13.7].

The 24-yard FG make model appears reasonably calibrated. The Rams Q4 example's 7+ pp shift when changing kickoff continuation appears primarily related to downstream WP rather than the probability of making the FG.

The 2025 4th-and-4 pass gap is not stable across periods and the near-goal sample is tiny. Do not fit to the selected conversion percentages or mistake them for causal go-for-it estimates.

## Step 70: monotone FG recalibration

Platt calibration trained on 2023 and evaluated on 2024; then trained on 2023–24 and evaluated on already-examined 2025. No tuned hyperparameters and no model replacement.

| Later evaluation | Subgroup | Original Brier | Candidate Brier |
|---|---|---:|---:|
| 2024 | All FG | 0.11769 | 0.11723 |
| 2024 | FG 50+ yards | 0.20797 | 0.20460 |
| 2024 | FG Q4 final five | **0.13993** | 0.14200 |
| 2025 exploratory | All FG | 0.10980 | **0.10835** |
| 2025 exploratory | FG 50+ yards | 0.21349 | **0.20889** |
| 2025 exploratory | FG Q4 final five | **0.14766** | 0.14805 |
| 2025 exploratory | FG 18–29 yards | 0.01867 | 0.01867 |

Lower Brier is better. The 2025 overall Brier improvement was +0.00146 with 95% CI [+0.00031,+0.00251], and 50+ yard improvement +0.00461 with CI [+0.00023,+0.00904]. Earlier 2024 overall improvement CI included zero.

**Deployment decision: do not promote.** The candidate does not improve late-game FG scoring consistently and the 2025 data are no longer independently untouched.

## Subsequent engineering target

Focus on the Q4 **post-score possession-value model**, not the short-FG make rate. Test onside recoveries, return touchdowns, penalties and other omitted kickoff branches; repair field-position and score consistency in home/away post-kickoff WP; then assess the end-to-end fourth-down decision gap with honest model uncertainty and independently later evaluation.

The app now flags the default recommendation as **model-sensitive** when the observed 2025 kickoff scenario changes the ranking or yields a near-tie. The frozen 2026 Week 6–18 study runs automatically once enough eligible games exist; it remains read-only and does not authorize a production promotion.

No manual scripts required. No production model change was made by Steps 69–70.
