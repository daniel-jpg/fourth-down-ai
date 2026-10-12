# 2027 independent component validation — preregistration and freeze conditions

**Registered October 2026, before the 2027 REG season.** The currently available 2023–24 development tests are not pristine independent tests. The 2027 outcomes are not available in October 2026, so no prospective performance result or coaching benefit is claimed. This component-validation track is SEPARATE from the fixed 2026 kickoff Steps 66/83 and the original 2027 one-decision coaching-policy diagnostic.

## Questions answered, and questions NOT answered

Factual conversion: among actual fourth-down normal GO RUN/PASS attempts, did frozen or field-position challenger better predict conversion? This cannot identify whether an unchosen PUNT or FIELD GOAL would have been better.

Factual win probability: among pre-snap regulation scrimmage states, did frozen or constrained WP better predict the eventual winning offense? This cannot establish the value of alternative unchosen fourth-down decisions or validate the simulation's complete GO/FG action edge.

The source cohort must exclude final tied games and record explicit exclusions. Every row is (game_id, play_id) from a completed 2027 regular-season game, with game-cluster inference for repeated states and labels.

## Frozen historical candidates

- Baseline GO: original regularized pooled logistic conversion, 14 feature contract, trained 2014–2022.
- Development GO challenger: same logistic model with continuous yardline_100 appended, same fixed C=0.25, trained 2014–2022. Already shown slightly worse on 2023–24 development and more aggressive in unsupported own-5 locations, so it is NOT being promoted.
- Baseline WP: unchanged three-class 2014–2022 frozen regulation model.
- Development WP challenger: fixed football-monotonic regulation scrimmage HistGradientBoosting, game-weighted 2014–2022 training, frozen class tie probability. It is not valid for kickoffs and delegates to frozen for kickoffs/OT. Development Q4 gain was inconclusive.
- Pin the actual unmodified artifact SHA-256, reviewed extractor source hash and frozen decision-policy commit 31176d50af25db9dd42dae07e9b17cb5c2b930b4 BEFORE any outcome analysis. Source version alone is not the model artifact hash.
- Record archived pre-snap prediction hashes at the time they are produced. Final outcomes are extracted independently after games become final, never passed into the prediction module.

## Avoid the logged-action lookahead trap

The predictions file has TWO distinct collections: factual-context win-probability predictions and **hypothetical GO RUN/PASS forecasts computed before knowing the coach's actual play choice**. It can record both GO action probabilities for every observed fourth-down situation, even when the coach ultimately chooses PUNT or FIELD GOAL.

The separate outcomes file supplies the observed GO RUN/PASS choice and its actual conversion label. The evaluator matches the factual observed action to one of those previously sealed hypothetical action predictions. Unchosen GO hypotheses are counted as unused, not scored as converted/failed; no final result or actual coach play-type label is permitted inside the pre-snap prediction schema.

This distinction is essential: conditioning the PRE-SNAP predictor file on the run/pass action that happens next would be post-snap leakage. Source manifests do not by themselves prove the extractor or the timestamp was clean. An independent source and feature-availability review is mandatory.

## Locked sample, metrics and support gates

Holdout: 2027 NFL regular season, Q1–Q4, after all games become final. No interim peeking, repeated interim scoring, or selected weeks. Different season/rule changes need a separate preregistered protocol.

For WP and GO factual predictions, compute Brier = mean((observed_binary - forecast_probability)^2). Gain is frozen Brier minus candidate Brier; positive is favorable. Paired 95% percentile intervals resample games (2,000 draws, seed 2027), not individual play states. Report distinct game count for every subgroup; fewer than 20 games means descriptive only.

GO groups: all chosen normal GO, PASS, RUN, fourth-and-2/3, own-10 fourth-and-2/3, own-5 fourth-and-2/3, Q4, HOME and AWAY. WP groups: all regulation, Q1–Q3, Q4, final-five-minute Q4, home/away/neutral, and exact Q4 final-five-minute own-10 with possessing offense leading by four **separately for HOME and AWAY**. This last group is a rough analog of a failed goal-line GO continuation, NOT a matched causal GO outcome.

A candidate only qualifies for **additional human review** if each component has at least 100 distinct games, both components' overall paired Brier CIs have positive lower bounds, Q4 WP paired Brier CI has a positive lower bound, and there are at least 20 independent games in each of own-5 and own-10 GO short-yardage groups, plus 20 games in each of Q4 own-10 leading-four HOME and AWAY groups. These strict target-region support thresholds may never be met. Missing support means evidence is insufficient, not that any one action is worse.

Even if all metrics pass, the evaluator MUST NOT auto-promote models or make a causal coaching claim. Model replacement requires a separate release authorization, artifact integrity audit, and policy-level decision evaluation.

## Executable code and safety

The pure offline component comparator is in src/prospective_component_comparison.py and tests in tests/test_prospective_component_comparison.py. It performs no network access, model fitting, 2026 gate dispatch, live app changes, or GitHub release actions.

The input JSON contains three separate top-level objects: manifest, predictions, outcomes. Predictions and outcomes each contain wp/go arrays. Forecasts have strict known pre-snap schema and season=2027. GO rows include predicted_go_action (hypothetical normal RUN or PASS). Outcomes contain the actually logged observed_go_action and converted flag, joined only AFTER pre-snap predictions are sealed. The WP outcome is offense_win. Invalid probabilities, duplicate IDs, missing chosen-action predictions, future-season violations, unexpected post-snap fields and mismatch of immutable forecast SHA are rejected.

Prospective mode requires a completed 2027 regular-season flag, independently verified and sealed input provenance, fixed artifact/extractor SHA-256, and at least 100 games per component. These are **necessary assertions, not sufficient proof** of independence. The 2027 prospective mode is NOT to be used early.

Development and CI usage (FABRICATED synthetic inputs only):

    PYTHONPATH=. python -m pytest tests/test_prospective_component_comparison.py
    PYTHONPATH=. python -m src.prospective_component_comparison synthetic_input.json --output synthetic_audit.json

No NFL 2026 or 2027 data are fetched by this software. Preserve both existing protected 2026 Steps 66/83 workflows exactly as registered, and preserve the separately registered 2027 fourth-down policy study.

**Bottom line:** As of October 2026, model-validity research and fail-safe display cautions are implemented; independent future football performance remains UNVERIFIED.
