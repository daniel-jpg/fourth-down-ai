# Issue #8 — historical pre-snap cohort and coach/action-win nuisances

**Engineering date:** October 11 Pacific / October 12, 2026 UTC.
**Scope:** Historical 2014–2025 development ONLY. No 2026 protected outcomes or future 2027 games, no model promotion, no automatic release, no changes to public Streamlit recommendations.

## What was actually built and run

- Strict historical state extractor: [src/historical_fourth_down_cohort.py](../src/historical_fourth_down_cohort.py), with fabricated contract tests. It whitelists the exact decision-time state, then materializes separate pre-snap, observed-action, final-winner and row-exclusion tables. Every included decision is identified by unique (game_id, play_id).
- Separate coach choice e(a|X) and logged-action conditional final win mu(a,X) nuisance models: [src/historical_coach_nuisances.py](../src/historical_coach_nuisances.py). Input prediction API **rejects action/outcome fields**; the desired policy action is supplied as a hypothetical action argument, not inferred from what a coach chose afterward.
- [Offline historical training/audit](../scripts/train_2027_historical_nuisances.py) reads only the checked-in labeled fourth-down PBP parquet, pinned to Git blob d7230df63cad1c24579ee40cabaacd0b330e4784. No NFL download is performed. Train 2014–2022, evaluate fixed 2023–2024 retrospective development, then fit separate unpromoted historic nuisance artifacts on all 2014–2025 records.
- [Passing historical job](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38205263693): 21 focused synthetic tests passed, the real historical census ran, the 2014–22/eval-2023–24 models were fitted and scored, and separate 2014–25 e/mu joblib artifacts and a SHA-256 provenance JSON were archived as development-only GitHub Actions artifacts.

## Historical cohort audit (50,095 candidate labeled fourth-down plays)

| Category | Count |
|---|---:|
| Included regulation REG normal GO/FG/PUNT decisions | 44,507 |
| Postseason / non-regular season | 2,153 excluded |
| Administrative no-play | 2,567 excluded |
| Fakes and other unclassified actions | 396 excluded |
| Overtime | 316 excluded |
| Tied final game | 155 excluded |
| Impossible fourth-down distance | 1 excluded |
| Total excluded | **5,588** |
| Games represented by included decisions | **3,140** |

Included factual normal actions: PUNT 26,163; FIELD GOAL 11,089; NORMAL_GO_PASS 4,908; NORMAL_GO_RUN 2,347.

These counts come from the checked-in historical data snapshot, NOT an external audit of coach intentions. The fake/special-teams classifier uses post-play descriptions to classify the action for **training labels only**, never as a prediction-time feature. Rare/hard-to-label plays are exclusions.

## Chronological 2014–22 training to 2023–24 exploratory test

- Training: **32,961** decisions through 2022.
- Retrospective development: **7,794** decisions from 2023–2024 across **544 games**.
- Historical coach-action multiclass log loss: **0.43286**, accuracy **83.74%** (dominated by action frequency; NOT proof of deep-field validity).
- Factual logged-action final-offense-win Brier: **0.15877**, log loss **0.47272**.
- Empirical 2023–24 factual-action propensities had 5th/50th/95th percentiles **0.1132 / 0.9323 / 0.9998**, a reminder of extreme skew and weak overlap.

| Observed 2023–24 action | Decisions | Factual final-win Brier | Factual e between .05 and .95 |
| --- | ---: | ---: | ---: |
| NORMAL_GO_RUN | 497 | 0.16237 | 93.8% |
| NORMAL_GO_PASS | 1,007 | 0.09812 | 84.2% |
| FIELD_GOAL | 1,998 | 0.17368 | 72.8% |
| PUNT | 4,292 | 0.16563 | 28.6% |

These are **factual** action distributions. The preregistered 2027 policy's *common-support set* must use **e(policy action | X)** at the 2027 decision, not the observed coach action. In particular, neither 28.6% punt propensity coverage nor the overall logged-action Brier is a causal policy advantage.

## Model specification and provenance

Fixed pre-snap covariates: quarter, regulation game clock, yardline_100, ydstogo, goal_to_go, score differential from the possessing team's perspective, own/defensive remaining timeouts, and -1/0/1 site advantage. No game final score, play gain, EPA, play description, observed conversion, winner or factual action is included among the input state features. Features are checked for finiteness and team/site consistency; incomplete states are excluded with logged reasons.

Propensity model: StandardScaler + 4-class logistic regression, fixed C=0.35, max_iter=3000, random_state=2027. Conditional reward model: separate weighted binary logistic model with state features, hypothetical-action one-hot indicators and predeclared action-by-state interactions for clock, quarter, distance, location and score differential. It weights repeated observations inversely by the number of fourth downs within the game. Hyperparameters were fixed in code, NOT selected on 2023–2024 outcomes.

The job writes the historical parquet hash, raw/source SHA-256, extractor/trainer source SHA-256, each final artifact SHA-256 and row counts into artifacts/historical_nuisance_development_audit.json. These are independently inspectable **assertions**, not cryptographic signatures or independent source verification. The actual archived joblib files live under the GitHub Actions development-only artifact; they are not the production frozen weights and are not loaded by the current Streamlit app.

## Durable research-model parameter archive

The GitHub Actions binary joblib archive has 21-day retention, so relying solely on that ZIP would risk losing the exact fitted nuisances. To avoid this, the tiny logistic model arrays (classes, standardization means/scales, coefficients and intercepts) have now also been preserved as **version-controlled plain JSON** at [data/historical_nuisance_2014_2025_development_parameters.json](../data/historical_nuisance_2014_2025_development_parameters.json).

[Pickle-free scorer](../src/historical_nuisance_portable.py) recreates the four-action propensity and conditional reward outputs from JSON without unpickling, fitting or looking at coach-observed plays. [Regression tests](../tests/test_historical_nuisance_portable.py) include four full reference predictions calculated directly from the original sklearn 1.9.1 archived models, source-file SHA-256 guards and post-snap input rejection. The model parameters can survive expiration of the temporary Action ZIP and can be independently inspected in ordinary Git diffs.

The archive is explicitly marked **RESEARCH_ONLY_UNAPPROVED**, not a signed production deployment or a 2027 coaching recommendation. Original binary artifact SHA-256s remain in the JSON metadata. The immutable source/trainer/provenance checks require a deliberate new model-review cycle if code changes. A separate signed source-of-truth and verified upstream game-result source are still required for prospective approval.

## Remaining independent evidence and freeze blockers

1. Verify historically labeled final game margins and game status against an **independent** authoritative source (and spot-check ambiguous fake/FG/PUNT classifications), especially neutral venue and no-play edge cases. The current training labels use the same historical PBP source's final result field; feature whitelists protect X from labels but do not prove the source's external correctness.
2. Reconstruct exact policy action estimates for **2027 decision-time states** from immutable commit 31176d50af25db9dd42dae07e9b17cb5c2b930b4, with Q4 native standard-35 kickoff inputs, pre-snap eligibility, action-specific WP and seed lock, and separate, independently sealed forecasts before 2027 outcome availability.
3. Independently review overlap, calibration and the effects of observed-coach selection by site, quarter, action and deep-field region. Substantive rare-action omissions may prevent the preregistered ESS/coverage/disagreement thresholds.
4. Sign/approve the model, source and extractor provenance manifest and secure immutable artifact retention prior to first 2027 REG game. GitHub Actions raw joblib retention is 21 days; plain-text numeric parameters are now version-controlled, but signed custody and external outcome verification remain outstanding. Do not claim pipeline freeze until reviewed provenance and archival conditions are satisfied.
5. 2027 independent action-level results and any credible causal assessment remain FUTURE work. The reserved 2026 Step 66/83 kickoff evaluation is separate and protected.

**Decision:** useful development engineering completed; historical nuisance model artifacts are **not promoted/approved** for prospective 2027 policy inference or production coaching. No causal FG-vs-GO superiority claim is supported by this report.
