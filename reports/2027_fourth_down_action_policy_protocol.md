# Preregistered 2027 fourth-down action-policy audit (separate from kickoff Steps 66/83)

**Registered:** October 2026, before the 2027 regular season.  
**Status:** protocol and synthetic-only estimator/tests; **no 2027 outcomes inspected**.  
**Fixed target policy:** [commit 31176d50af25db9dd42dae07e9b17cb5c2b930b4](https://github.com/daniel-jpg/fourth-down-ai/commit/31176d50af25db9dd42dae07e9b17cb5c2b930b4), including the Q4 native standard-35 kickoff-input correction and normal-action eligibility rules. All later results use this unchanged snapshot.  
**Protected separate validation:** [2026 Steps 66/83](2026_native_kickoff_challenger_protocol.md) stay fully frozen and closed until their existing automatic once-only gates permit completion.

## The two questions are not the same

1. **Primary: factual action win-probability calibration.** For a future game where a team *actually chose* GO (pass/run), FIELD GOAL or PUNT, did the frozen model's predicted probability of eventually winning under that **chosen action** match observed eventual wins? Evaluate Brier error, calibration bins and action-level support. This is a testable, descriptive prediction claim.
2. **Secondary: observational one-step policy evaluation.** Where actual coach choices have sufficient overlap with the fixed recommended action, compare a hypothetical substitution at **one** fourth-down decision with observed coaching decisions, using pre-snap propensity and outcome regressions. This is a doubly robust (DR) observational diagnostic, **not** randomized evidence, a full-season policy trial or identified causal coaching benefit.

**Neither a low factual Brier score nor a positive observational DR comparison proves the counterfactual outcome of the same fourth down under the unchosen action.**

## Preregistered future cohort and label

- **2027 NFL REGULAR season Weeks 1–18**, only after all included games become final. Eligible decision units are regulation fourth downs (quarters 1–4) with factual normal dropback/pass GO, designed rush GO, regular FIELD GOAL or normal PUNT. Exclude overtime, fake plays, no-play administrative penalties, unclassified actions, unmatched game/play keys and final tied games; document exclusion counts.
- One unit = (game_id, play_id), reconstructed strictly **before the snap**. Repeat fourth downs in a game are allowed but are **clustered by game** in confidence intervals.
- The post-game outcome label is 1 when the offense at the decision eventually won, and 0 when it lost, in a completed non-tied game. Never use this or other post-snap features to form predictions, propensity or reward regression.
- Policy action is the **highest-WP eligible normal action** produced by the fixed policy snapshot. For Q4 it includes the pinned native kickoff-input feature correction; classifier weights are unchanged. Recorded coach action is not randomized. In unsupported model states, report missing factual WP and population coverage; never impute from the final score.

### Data leakage and nuisance estimation

The target policy, the estimated logged-coach propensity e(action|X), and the independent outcome model mu(action,X) must use **decision-time covariates only**. Their training/tuning data must end no later than the **2025 season**. Use chronological evaluation/cross-fitting within historical 2014–2025 data as available; do **not** train, tune, diagnose, or peek at any outcomes from the reserved 2026 Steps 66/83 period. The 2025 policy sample has already been examined, so it is development-only, not an untouched test cohort.

Before any 2027 outcomes are evaluated, pin the nuisance source versions, feature lists, training cutoff, propensity calibration tests, missing-data handling, and source hashes. Independently audit the future 2027 cohort extractor to prove pre-snap feature availability. A JSON manifest is a provenance assertion, **not proof by itself** that the extractor was leakage-free.

## Fixed metrics and methods

### Factual action-conditional calibration

For each model-supported observed action with forecast p and outcome y in {0,1}: Brier = mean[(y-p)^2]. Report total and per-action counts/Brier; observed vs mean predicted win rate in fixed bins [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]. Bins with fewer than 20 plays are descriptive only. Report unsupported factual action counts, not only successful predictions.

For overall Brier, a **95% game-cluster percentile bootstrap** (2,000 resamples, fixed seed 2027) resamples complete games, never plays as if independent. Prespecify descriptive breakouts: Q4 vs Q1–Q3, kicker/home/away context, Close call flagged/not flagged, fourth-down action and situation coverage. The initial synthetic-only evaluator implements overall/action and fixed WP-bin reporting; the pre-snap cohort builder and report renderer must add fixed Q4/site/close-call breakouts **before** 2027 evaluation, without outcome-based subgroup choices.

### Exploratory logged-coach policy comparison

Common-support set: **0.05 <= e(policy action | pre-snap X) <= 0.95**, defined without conditioning on the *observed* action or outcome. For each included fourth-down decision:

> DR_i = mu(policy_i, X_i) + 1{observed_i = policy_i} * [win_i - mu(policy_i, X_i)] / e(policy_i | X_i).

Report DR policy value, observed coaching win rate over that **same support**, their mean difference, and the **95% game-cluster percentile bootstrap CI**, 2,000 draws, fixed seed 2028. Also disclose common-support coverage, number of games, coach/model disagreement count, propensity distribution, and importance-weight effective sample size (ESS). Never silently clip out-of-range DR values; instability is a warning about weak identification.

Before presenting even an **exploratory** policy contrast, require **>=200 supported decisions, >=100 distinct games, ESS >=100, >=40 coach/policy disagreements, >=60% support coverage**. Otherwise mark *insufficient overlap* and do not interpret uplift. In a prospective report, the evaluator now **suppresses** the DR policy value, policy-minus-coach estimate, and game-cluster interval (JSON `null`) if any threshold fails; it preserves the observed-coach win rate, counts, support fraction, ESS, disagreement count and machine-readable failed-criterion codes. **Synthetic-only** CI cases still compute the estimator for arithmetic testing, even with inadequate overlap, and cannot be interpreted as evidence. Passing every support threshold still does NOT establish exchangeability, correct nuisance modeling or causality.

The estimand is **one decision, with other observed/coaching continuation unchanged**, not the value of forcing the same policy at every decision in a game. The historical observed coaching win rate is the comparator on the exact common-support population, not a simulated counterfactual.

### Threats requiring explicit disclosure

- Hidden coaching information (kicker injury, wind, personnel, play design, opponent alignment, weather, field conditions, team-specific style, fatigue) makes observational comparisons confounded.
- Positivity/overlap may be weak, particularly GO_RUN beyond model support and unusual PUNT or FG spots; count unsupported actions and disclose population drift.
- Monte Carlo standard error covers simulation noise only, not action-model misspecification or causal uncertainty.
- Multiple decisions from the same game share terminal labels: use game clusters.
- 2027 rules and coaching may differ from 2014–25. If material rules change, report "incompatible" and write a new separately registered study before evaluating outcome-derived changes. Never tune this locked policy after peeking.
- No repeated testing until results look favorable and no cherry-picked subgroup or propensity cutoff changes.

## Staged work plan

1. **Now:** commit [pure offline study evaluator](../src/fourth_down_policy_study.py) and [synthetic-only CI tests](../tests/test_fourth_down_policy_study.py). No external data retrieval, scheduled future-outcome access, retraining, early Step 66/83 run, automatic app change or new release.
2. **Before 2027 Week 1:** implement a transparent pre-snap cohort builder and independent propensity/outcome-nuisance fit on historical games **through 2025 only**. Lock algorithms, features, handling of unsupported actions, artifact hashes and fixed source policy; test with fabricated data and label any 2023–25 examples *exploratory*.
3. **After all 2027 Week 18 regular-season games are final:** construct the holdout **once** and record the game/play IDs, omissions, cohort-builder commit, final-game checks, policy SHA and nuisance provenance before computing results. Run the offline evaluator, save a reproducible JSON report and review fixed overall/action/site/time/Close call summaries. If data are absent or too sparse, report *not evaluated* or *insufficient support*.
4. **Review with explicit authorization:** consider narrowing prediction-specific language **only** if supported by independent validation. Do not publish a new stable release or infer an actual coaching advantage automatically.

### What would let us change the current app disclaimer?

- The **2026 Step 83** prospective kickoff test can address only the sentence *"corrected Q4 kickoff inputs have not yet been validated in future games"*. Passing its frozen pooled, home and away Brier criteria may justify shortening **that specific qualification**. It does not validate an individual FG-versus-GO edge.
- The **2027 factual WP and observational one-step study** can test action-level calibration and identify overlap problems. Even a positive DR confidence interval does not, on its own, remove unknown-confounder or unchosen-action uncertainty. To claim a *causal* FG-versus-GO edge, a stronger design (e.g. credible natural experiment or ethical randomized prospective collaboration with an NFL team) is needed.
- A close-call or model-assumption notice may therefore remain even when the narrower statement about kickoff **future calibration** becomes outdated.

## Offline evaluator input

A JSON object has keys **manifest** and **records**. Every record supplies game_id, play_id, season=2027, week, quarter 1–4, observed_action, policy_action, eligible_actions, offense_win, q_wp_observed, q_wp_policy, mu_policy, propensity_policy, feature_asof="pre_snap". An unsupported actual action may omit q_wp_observed (null) but remains counted as unsupported. The manifest specifies mode ("synthetic" or "prospective"), policy_commit (the exact pinned SHA above), feature_policy="pre_snap_only", holdout_season=2027 and nuisance_training_end_season<=2025. A prospective run also requires an explicit final_2027_regular_season_complete=true marker, nuisance_provenance and at least 100 distinct games.

Offline invocation, **only when these preconditions are met**:

    python -m src.fourth_down_policy_study sealed_input.json --output audit.json

No code in this study reads from nflverse, fetches the protected 2026 games, trains models or runs the existing Steps 66/83 gates. Passing synthetic tests is *software verification*, not sports-science validation.
