# Issue #9: Deep-own GO vs PUNT break-even conversion sensitivity

**Research:** October 12, 2026. **Status:** COMPLETED computational sensitivity study, **NOT** new causal or out-of-sample evidence. [Passing CI audit: 49 tests + 16 real-engine modeled action/seed cases](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38278292661). [Reproducible read-only source](../src/deep_own_go_break_even_audit.py) | [Safety and mathematical tests](../tests/test_deep_own_go_break_even_audit.py).

The earlier [deep-own empirical study](issue9_field_position_go_q4_site_followup.md) and [detailed branch diagnosis](issue9_deep_territory_site_audit.md) established why the current frozen model may display GO as attractive from the offense's own five-yard line despite **zero comparable fourth-and-2/3 GO attempts in 2014–2022 training**, and confirmed the simulated failed-GO short field is numerically oriented correctly. The public research demo already abstains with **INSUFFICIENT DATA** when the preferred strategy is GO inside own 10 on fourth-and-2/3.

This additional audit asks a more precise question: **Holding all other parts of the frozen simulator fixed, by how much could the GO conversion estimate be optimistic before a PUNT ranks ahead?** This is a hypothetical calculation, **not** a learned correction or an estimate of actual NFL team behavior.

## Exact scenario and reproducible frozen inputs

- Q1, tied, 05:30 remaining in quarter; fourth-and-3, three timeouts for each team, neutral stadium, outdoors.
- Offense starting at own 5, 10, 20 and 35, defined by original model yardline_100 = 95, 90, 80, 65.
- Existing conversion model and win-probability model, same frozen code/assets and transition pools, pinned by six original Git blobs before opening any model or play data.
- Both hypothetical NORMAL_GO_PASS and NORMAL_GO_RUN, with seed 43 and independent seed 101, **1,200 simulation draws each** of all-convert, all-fail and PUNT branch estimates.
- No Q4/native-kickoff modifications, no 2026/2027 outcome files, no refitted models or app recommendation changes.
- The prior checked-in fourth_down_modeling_split.parquet is SHA-gated before opening; only **2014–2022 training** and **2023–2024 already examined validation** support records are counted. 2025 is present in the locked archive but excluded from this test's counts/metrics.

## Conditional simulator calculation

The simulator supplies three action-conditional estimates for the same fabricated state:

- F = estimated original-team win probability **if every simulated GO attempt failed**, using the exact frozen failure-donor process
- S = estimated original-team win probability **if every simulated GO attempt converted**, using the exact frozen success-donor process
- U = estimated original-team win probability **if choosing PUNT**, from the untouched frozen punt simulation

For an *assumed* conversion probability p, the fixed-branch mixture is:

    GO(p) = (1-p)*F + p*S

Provided S > F, the model-implied break-even against PUNT is:

    p_break_even = (U-F)/(S-F)

That threshold may lie outside [0,1], which is handled explicitly; no fake boundary clipping is used. **This changes only the conversion-probability assumption in a separate cloned function namespace.** The original frozen GO predictor, field-position transitions, resulting original-team score axis, and both models' coefficients remain untouched. Monte Carlo standard errors are recorded per branch, and the scenario's rank can shift under different donors/seeds. This calculation **does not identify true conversion probability, GO-vs-PUNT causal value, or model specification error**; it fixes the conditional donor distributions, which themselves may extrapolate badly from deep own field.

## Results (model-only; percent conversion, not win probability)

The original frozen model's own-5 conversion estimate is **55.15%** for normal PASS and **66.38%** for normal RUN. Because the conversion model omits continuous yardline_100, these estimates are the same across the four tested locations when all other state inputs are fixed.

| Starting position | PASS break-even, seeds 43 / 101 | RUN break-even, seeds 43 / 101 | PASS simulated GO edge over PUNT (pp) | RUN simulated GO edge over PUNT (pp) |
| --- | --- | --- | --- | --- |
| Own 5 | 44.93% / 44.63% | 47.73% / 47.28% | +1.81 / +1.86 | +3.06 / +3.11 |
| Own 10 | 48.05% / 47.98% | 50.91% / 50.40% | +1.29 / +1.29 | +2.61 / +2.69 |
| Own 20 | 51.59% / 50.43% | 55.61% / 54.94% | +0.64 / +0.86 | +1.78 / +1.89 |
| Own 35 | 44.74% / 43.08% | 47.73% / 46.17% | +1.79 / +2.11 | +3.02 / +3.29 |

The GO edges in this table are **interpolated model estimates, not independently measured coaching benefits**. The engine's full-recommendation random draws may yield slightly different action-level estimates because of RNG consumption; this is a dedicated conditional branch sensitivity study. The close agreement between the two seeds reduces concern about a single random draw but does **not** test the correctness of the underlying model.

**Own 5 tipping point:** If the frozen PASS conversion estimate (55.15%) overstates the assumed rate by roughly **10.2–10.5 percentage points**, PUNT overtakes PASS within this fixed-donor simulator. For RUN, the corresponding break-even overstatement is **18.7–19.1 percentage points**. Those are thresholds, **not estimates that the current model is actually off by those amounts**. GO_RUN's original +3pp simulated edge is conditional on unsupported near-own-goal run conversion assumptions.

**Own 20:** The PASS tipping point is more sensitive: the 55.15% model prediction needs to fall only roughly **3.6–4.7 percentage points** for PUNT to overtake PASS in this model setup. This motivates careful robustness reporting even outside the narrow own-10 abstention region; it is **not** by itself grounds to widen the existing UI warning without a separately reviewed user-facing support rule.

## Historical choice-support denominator (not causal propensity)

Fourth-and-2/3 means historical ydstogo 1.5–3.5. These are **normal GO vs ordinary PUNT decisions**, excluding fake plays, FG and no-plays. Observed GO/(GO+PUNT) is descriptive action frequency, not a causal or identifiable propensity to attempt GO given all football context.

| Region | 2014–2022 chosen GO | 2014–2022 chosen PUNT | 2023–2024 previously examined chosen GO | 2023–2024 chosen PUNT |
| --- | ---: | ---: | ---: | ---: |
| Own 1–5 | **0** | **0** | **0** | **0** |
| Own 1–10 | **1** (0 converted) | 44 | 0 | 6 |
| Own 1–20 | 13 (8 converted) | 292 | 5 (5 converted) | 57 |
| Own 1–35 | 73 (38 converted) | 1,421 | 25 (16 converted) | 256 |

All ranges are **nested**, so their counts must not be summed. Only one training normal GO was chosen inside own 10, and it failed. The own-5 region has **no observed GO or PUNT examples in this restricted fourth-and-2/3 decision table**. Inferring a reliable location-specific conversion probability or a local action comparison would be statistically unjustified; the model's predicted 55%/66% conversion rates are extrapolated there.

## Implications and next decision

1. **Keep the existing INSUFFICIENT DATA abstention** for own-10, fourth-and-2/3 when GO is chosen by the model. The narrow display-only mitigation remains supported; the engine's GO/PUNT values should remain inspectable but qualified. Don't silently replace GO with PUNT.
2. **Do not modify frozen weights, conversion/run shift, transitions, or Q4 site assumptions.** Adding only yardline_100 to the conversion model was already found to worsen 2023–2024 development Brier in the previous fixed-spec comparison. A hand-picked penalty learned from this unusual scenario would be post-hoc overfitting.
3. **Retain both kinds of uncertainty** in interpretation: (a) stochastic Monte Carlo error around a fixed simulator; (b) dominant unmeasured structural uncertainty from essentially nonexistent geographically matched actual GO decisions and distant donor branches. Two seeds and 1,200 simulations address only (a).
4. **Require future genuine independent evidence** to evaluate conversion calibration and GO/PUNT recommendation quality. The separately [prespecified 2027 component protocol](2027_component_candidates_prospective_protocol.md) already distinguishes actual normal GO conversion from the causal effects of unchosen choices and requires enough independent own-field game support. The live pre-snap source/witness gate in [Issue #8](../README.md) remains open.

### Reproduction

    PYTHONPATH=. python -m pytest -v tests/test_deep_own_go_break_even_audit.py
    PYTHONPATH=. python -m src.deep_own_go_break_even_audit --simulations 1200 \
       --out artifacts/issue9_deep_own_go_break_even.json

The [passing research workflow](../.github/workflows/issue9_go_break_even_sensitivity.yml) checked **49 targeted tests**, reproduced all 16 action/site/seed comparisons, captured 8 historical support rows, checked original model/source hashes, and archived the JSON + log for 30 days. Its generated data are isolated under ignored artifacts/; **neither original dataset nor production model/application changed**.

**Conclusion:** The unexpected aggressive own-5 GO ranking is a reproducible feature of the frozen simulator, but conversion accuracy in this region is unverified. Our new break-even thresholds measure *which assumptions drive the recommendation*, not whether NFL teams should actually go for it.
