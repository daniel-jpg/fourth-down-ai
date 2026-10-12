# Issue #9 — Deep-own-field GO support and Q4 site-value audit

**Investigation:** October 12, 2026.  
**Reproducible historical evidence:** [audit run #2](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38200763389), [source audit](../scripts/audit_issue9_support_and_branches.py), [two-seed scenario QA #253](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38200131029).

**Scope:** Frozen 2014–2022 training artifacts and descriptive 2023–2025 development data only. No 2026 outcomes, no model retraining, no Step 66/83 manual runs, no causal effect claims. All percentages are model estimates, not observed wins under alternate coaching choices.

## Finding A: GO from a team's own 5 is unsupported extrapolation

In `data/fourth_down_modeling_split.parquet` (normal GO = pass/dropback or designed run; fourth-and-2/3 approximated by 1.5–3.5 yards to go):

| Starting area | Normal GO, 2014–2022 **training** | All 2014–2025 GO (already examined; descriptive only) | PUNT 2014–2022 training |
|---|---:|---:|---:|
| Own 1–5 | **0** | **0** | 0 |
| Own 1–10 | **1** (one pass, failed) | **1** | 44 |
| Own 1–20 | **13** (12 passes, 1 run) | **22** | 292 |
| Own 1–35 | **73** (68 passes, 5 runs) | **109** | 1,421 |

The current frozen GO conversion model has only coarse opponent-red-zone and goal-to-go indicators and **omits continuous starting `yardline_100`** from its feature list. Consequently it predicts **PASS conversion 55.15%** and **RUN conversion 66.38%** for the Q1 tied own-5 fourth-and-3 example despite zero comparable own-5 training examples.

The GO simulation preserves actual field geometry. On failed own-5 GO attempts, the opponent's next first-down field position is about **4–5 yards from the scoring end zone**, as expected for turnover on downs. No reversed possession or lost short-field penalty was identified.

However, its top 250 "local" GO transition donors are not truly local in this rare region:

| Donor branch | Comparable own-10, fourth-and-2/3 donors in top 250 | Median donor `yardline_100` | Median absolute field-position gap from own-5 (yd) |
|---|---:|---:|---:|
| PASS failed | 1 | 66 | 29 |
| PASS converted | 0 | 66 | 29 |
| RUN failed | 0 | 52 | 43 |
| RUN converted | 0 | 66 | 29 |

At the tested Q1 own-5 state, the frozen model's simulated continuation estimates were **GO_RUN 45.02%**, **GO_PASS 43.71%**, **PUNT 41.95%** (isolated audit sequence; simulations and action-specific RNG differ from the app's full-recommendation seed consumption). The conditional mean WP after failed GO was ~34%, and after normal PUNT ~42%. The aggressive GO rank is mathematically consistent but **does not have defensible location-specific historical support**. It should **not** be labeled a verified +3pp coaching advantage.

**Safe immediate mitigation:** a **display-only**, narrowly triggered informational notice for GO recommendations on regulation fourth-and-2/3 inside a team's own 10, grounded in the development counts. No action is suppressed, no expected WP is changed, and no hard-coded PUNT recommendation is invented. Removing GO from the ranked action set, retraining the conversion model or recalibrating the simulated policy would be a material modeling-policy change requiring separate versioned validation, not an arbitrary reaction to surprising results.

## Finding B: Q4 HOME/AWAY reversal originates in learned downstream WP

Same quarter 4, 4:59, fourth-and-4 at opponent 6, down 4, three timeouts each, outdoors. Using the corrected native standard-35 kickoff input representation:

- **Immediate PASS conversion probability:** 46.26% for AWAY, NEUTRAL and HOME.
- **Immediate 24-yard FG make probability:** 97.992% for all three sites.
- **No direct site difference in immediate action odds**; the differing continuation WPs drive the rank reversal.

One 1,200-draw branch audit (GO pass, separate action RNG; percentages can differ slightly from the full recommendation because of different RNG consumption):

| Sub-outcome | AWAY original-offense WP | NEUTRAL original-offense WP | HOME original-offense WP |
|---|---:|---:|---:|
| GO produces a score (~17.2% of simulated passes) | 60.2% | 65.3% | 70.4% |
| GO retains possession without score (~28.5%) | 55.6% | 56.1% | 56.7% |
| **GO fails and opponent has ball (~54.2%)** | **15.0%** | **22.4%** | **29.9%** |
| FG made (~98.2% in that sampled run) | 37.4% | 37.2% | 36.9% |
| FG fails (~1.8%) | 16.0% | 20.5% | 25.0% |

The largest site difference is on **GO failure**, which is more than half the simulated pass outcomes. In the full 1,200-action seeded matrix, AWAY FIELD GOAL led **37.00% versus GO 34.34%**, while HOME GO led **44.52% versus FG 36.70%**. The recommendation changed when the site changed, but this is **not caused by an incorrect field-goal conversion probability or a missing turnover-on-downs field-position flip**.

**Mirror check:** The WP model is trained on home-team coordinates. NEUTRAL is defined to average the corresponding HOME and AWAY orientation predictions. The two-seed [site sensitivity CI check](../scripts/qa_site_territory_sensitivity.py) now verifies that the neutral results equal the mean of both orientations for GO and FG, at fixed simulation inputs. This rules out a simple HOME/AWAY sign reversal in these calculations; it does **not** prove that the magnitude of the learned site effect is calibrated.

The historical [2023–2025 Q4 kickoff representation audit](2023_2025_broad_q4_native_kickoff_calibration.md) found pooled Brier improvement from corrected native inputs but a **slight negative subgroup result for away-kicking teams**. That data was already explored and cannot establish a newly valid site correction or causal fourth-down coaching policy.

**Decision:** Do not change classifier coefficients, replace the model's HOME/AWAY effect with a hand-tuned adjustment, select models after peeking at reserved 2026 outcomes, or publish a new stable release. Retain Q4 research-demo limitations. Site-specific **prospective** kickoff reliability is preregistered in the protected [2026 Step 83 protocol](2026_native_kickoff_challenger_protocol.md). Independent future fourth-down action calibration is separately planned for [2027](2027_fourth_down_action_policy_protocol.md).

## Acceptance and remaining evidence

- [x] Real-model 15-scenario QA passed; no fabricated probabilities for unsupported actions.
- [x] Two-seed field/site sensitivity check reproduced the unusual rankings.
- [x] Source historical train/validation counts verified; only **one** GO attempt in own 1–10 fourth-and-2/3, none in own 1–5.
- [x] Deep own-field turnover location and downstream WP decomposition verified.
- [x] HOME/AWAY difference localized to WP continuation; immediate action probabilities unchanged.
- [x] Added a narrow, **display-only** sparse-support notice and unit/Streamlit integration tests; no change to existing model calculations.
- [ ] Assess predictive calibration and policy behavior on a truly independent future dataset as allowed by frozen protocols. **This cannot be completed now.**

An expert coach should not mistake a 3pp lead at own 5 or a large site-specific GO advantage for an empirically established decision-quality improvement. The application remains a **qualified research demo**.
