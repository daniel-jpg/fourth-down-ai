# Issue #9 follow-up — field-position GO and Q4 HOME/AWAY reliability

**Research date:** October 11, 2026 Pacific / October 12 UTC.
**Status:** Two read-only historical-development audits completed. No model promotion, prospective-data peeking, or causal coaching claims.

## GO conversion challenger: no improvement, reject promotion

Sources: [field-position GO helper](../src/field_position_go_research.py), [audit script](../scripts/audit_go_field_position_challenger.py), [passing Actions audit](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38203143916).

A single preregistered research challenger appends continuous starting yardline_100 to the frozen regularized pooled logistic GO model's existing 14 features. The learner and fixed C=0.25 are unchanged. Training was exclusively 2014–2022, and already-examined 2023–2024 was used for exploratory chronological development validation. The fixed 2025 dataset and 2026+ outcomes were not read.

| Metric on 1,598 actual normal GO choices in 2023–24 | Frozen | Challenger |
| --- | ---: | ---: |
| Brier (lower better) | **0.217991** | 0.218233 |
| Log loss (lower better) | **0.624860** | 0.625377 |
| AUC (higher better) | **0.69488** | 0.69282 |

Frozen-minus-challenger Brier gain = -0.000241, paired game-cluster 95% CI [-0.000855, +0.000322]. No evidence the challenger is better overall.

**Comparable fourth-and-2/3 GO support:**

| Location | 2014–22 training | 2023–24 validation |
| --- | ---: | ---: |
| Own 1–5 | **0** | **0** |
| Own 1–10 | **1** | **0** |
| Own 1–20 | 13 | 5 |
| Own 1–35 | 73 | 25 |

At own 5, fourth-and-3 Q1 tied, the model's pass conversion increases from 55.15% frozen to 61.51% challenger. The normal-run estimate increases from 66.38% to 71.80% after applying the same existing short-run logit adjustment to both. These are unsupported model extrapolations, NOT actual conversion rates.

**Action:** The challenger was NOT promoted. The research demo now abstains at the recommendation headline with **INSUFFICIENT DATA** for the very narrow region: regulation fourth-and-2/3 inside own 10, *when the model picks GO*. It still shows all original simulated GO/PUNT win probabilities and ranking, clearly flagged as research-only. No forced PUNT, new conversion coefficients, or eligibility changes.

## Q4 HOME/AWAY model sensitivity: real; local calibration remains sparse

Sources: [WP site helper](../src/q4_site_wp_research.py), [historical 2023–24 site audit](../scripts/audit_q4_site_wp_2023_2024.py), [passing Actions audit](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38203292979).

The audit reuses the original pre-snap WP feature builder and frozen three-class WP model. It uses only 2023–2024 REG non-neutral, non-kickoff regulation states. The final winner is a scoring label ONLY. There were 78,127 analyzed states from 534 games; repeated plays are game-clustered, not independent game results.

- Q4 final-five-minute states with possessing offense at own 1–10: 173 play states in 83 games.
- Same region while possessing offense leads 1–8: **28 play states in 13 games** (HOME 10 states in 4 games, AWAY 18 states in 9).
- Exactly +4 lead in same region: only **7 play states from 3 games, all AWAY**, with **zero HOME-offense examples**.
- These observed scrimmage states are *analogous* to a failed near-goal-line GO continuation, but are not necessarily the results of failed GO plays.

For a mirrored Q4 first-and-10 at receiving offense's own 6, leading by four with 4:50 remaining and three timeouts each, the same frozen WP model gives **84.29% when that offense is HOME**, versus **73.18% when AWAY**: a deterministic **11.11 percentage-point difference** from correctly mirrored possession, score, and timeouts. This is model site sensitivity, NOT an experimentally observed home-field effect.

No data support a tuned correction of the 11pp model difference or an automatic replacement win-probability model. The app now adds a narrow late-Q4 goal-line GO/FG site-support caption, noting the 13-game analog and model-specification uncertainty beyond Monte Carlo sampling error.

## Final decision and next evidence

- **No live conversion or WP weight changes.** The historically trained yardline-only challenger is slightly worse overall and extrapolates even more aggressively from own 5.
- **No overconfident own-field coaching recommendation.** A selective, display-only insufficient-data classification replaces the misleading GO headline in the empirically unobserved region; complete ranking remains visible.
- **No arbitrary home/away correction.** Late-Q4 backed-up leading-by-four situation has only three observed games and no HOME-offense examples in the 2023–24 development sample.
- Independent future WP calibration and causal action-policy studies remain required. Preserve the original 2026 Step 66/83 automatic once-only gates and the separately registered 2027 action-policy study. Do not promote based on retrospective development, modified gates or selected scenario results.

A passing software test verifies behavior, not optimal on-field coaching. This app is a qualified research demo, not a prospectively validated coaching policy.
