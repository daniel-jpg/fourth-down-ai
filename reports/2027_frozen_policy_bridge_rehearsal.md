# Issue #8: frozen 2027 decision-policy bridge — synthetic integration proof

**Prepared October 2026 before the 2027 regular season.**
**Status:** Pinned source and algorithm verified; 2027-shaped fabricated smoke
tests only. NOT an approved 2027 prospective extractor, not a coaching
deployment, and not independent evidence of a real FG-vs-GO effect.

## Critical finding: the registered policy is NOT bare engine.recommend

The registered action-study policy is [source commit 31176d50af25db9dd42dae07e9b17cb5c2b930b4](https://github.com/daniel-jpg/fourth-down-ai/commit/31176d50af25db9dd42dae07e9b17cb5c2b930b4). Its **app orchestration** matters:

- **Q1–Q3:** 300 simulation trials/action, seed 42. If the top-two **normal-action** gap Z is nonfinite or <1.96 combined Monte Carlo SE, rerun all actions from scratch with 1,200 trials/action, seed 43. It is NOT always 4,000.
- **Q4:** use the temporary isolated-model namespace and [native Q4 kickoff-feature candidate](../src/streamlit_kickoff_preview.py) (unresolved NORMAL score-produced kickoffs encoded with ydstogo=0 and yardline_100=35). Always 1,200 trials/action, seed 43. The original engine alone does **not** apply this native adjustment.
- Filter the engine's fake actions out of the four registered normal policy choices; normal GO RUN is unavailable past 3 yards, FG past 70 yards, and punt formation inside the opponent's 20 as enforced in the pinned engine. The highest-WP *eligible action* is the action-policy treatment; RUN/PASS both map to the same user-visible GO FOR IT strategy.
- The study's Close call subgroup uses the pinned app's **strategy-level** GO-versus-FG/PUNT gap Z, NOT merely a near tie between GO-RUN and GO-PASS. The adaptive pilot escalation above remains action-level.

The pinned engine source Git blob is
fa69ee309221d8e25af86026956c082d1862d89a.
The published Q4 orchestration helper and native patch are pinned to Git blobs
c41782a34b1f0e9e75ec812174247d6dae3273fd and
03bca375f042b24fb51099050cc7b3fd9ef796a7.
The original **app.py** at the registered commit has Git blob
b5588fb16bff7bb595155ab5141e946303f56a1f.
Today's app.py has extra, later safety displays, so its Git blob alone
CANNOT serve as the target action-study policy source.

The [bridge module](../src/frozen_2027_policy_bridge.py) explicitly checks
the pinned engine and EVERY frozen input artifact/transition pool for
byte-identical Git blobs before importing the frozen engine. The Q4 patch
loads **no new 2026+ outcomes**; the original 2026 Step 66/83 gates remain
untouched and never dispatched by the bridge.

## Frozen synthetic input state and source separation

The bridge takes a strict, fabricated-only pre-snap record with:

- IDs (season=2027-shaped synthetic only, game_id, play_id, week);
- known offense/home/away teams and offense-perspective HOME/AWAY/NEUTRAL
  plus the -1/0/+1 site coordinate;
- Q1–Q4 game clock, down=4, opponent-distance yardline_100, ydstogo,
  goal_to_go, score differential and both timeout counts;
- roof classification from **actual pre-snap venue**; the engine defaults
  to outdoors if missing, so missing roof is explicitly rejected;
- Q2 requires the pre-known second-half kickoff receiver OFFENSE/DEFENSE,
  needed for end-of-half crossing. A missing receiver is rejected rather
  than silently trusting the engine's fallback;
- source_type=fabricated_only and feature_asof=pre_snap, with EXACT
  whitelist. Any observed_action, converted, result, yards_gained,
  offense_win, final score or arbitrary extra column is an error.

Goal-to-go is checked against line-to-gain and yards-to-opponent end
zone (including exact decimal yard representation). The historical
nuisance model receives ONLY the 9 documented pre-snap state features;
roof and kickoff receiver are passed only to the football simulator.
The absence of individual kicker-strength priors is **explicitly
recorded as use of the pinned engine's default kicker prior**, not
evidence of known actual kicker fitness on a past snap.

The model fits e(action|X) and mu(action,X) were developed using
2014–2025 historical data and are plainly labeled unapproved. The
[portable parameters](../data/historical_nuisance_2014_2025_development_parameters.json)
are SHA-256 pinned to the source and training code, and reject observed
actions or final winners in their prediction API.

The adapter produces exactly the input pre_snap fields used by the
[synthetic cohort builder](../src/fourth_down_cohort_builder.py):
policy_commit, policy_action, eligible_actions, every eligible normal
q_wp_by_action, propensity_policy for the **policy** action, mu_policy
for the same hypothetical action, the pinned app's strategy Close call
flag and all identifying state fields.

**Outcome/action join happens only afterward**, using the independently
supplied fabricated observed play and final-game fields in the cohort
builder. A test verifies a coach choosing GO-PASS when the frozen policy
prefers FG keeps the pre-snap forecast intact, then joins the factual
observed action and eventual win as labels.

## Verification / simulation precision

The synthetic-only [test suite](../tests/test_frozen_2027_policy_bridge.py)
checks frozen file hashes, source/roof/clock/team/goal-to-go guards,
action ranking and eligibility, source X/result separation, the exact
cohort-builder join, individual run/pass versus strategy close-call
separation, and **real pinned engine** replay:

- Q4 at opponent's 6, fourth-and-4, down four: actual Q4 native
  correction path, with a 200-trial synthetic smoke budget.
- Q2 near halftime with explicitly recorded second-half receiving team:
  actual adaptive path with 32-trial synthetic smoke budget.
- Q1 at neutral site: actual adaptive path with 32-trial budget.
- Separate own-5 4th-and-3 synthetic pass/run/FG/PUNT eligibility case.
- No empirical kickoff 2026 results, 2027 outcomes, model training,
  release tag, public app mutation or protected prospective gate run.

**Smoke-test budgets are intentionally lower than registered app
precision**, and are labeled as such. They verify code-path/source parity
and seeded structural behavior; their WP percentages are NOT official
1,200-trial policy forecasts or new validation evidence.

## Remaining freeze blockers before 2027 Week 1

1. Obtain an independent factual pre-snap source for actual venue roof,
   second-half kickoff receiver, offense/home/away identity, line-to-gain
   and kicker context as of each fourth-down decision, with independently
   verified timestamps. The synthetic contract does NOT prove real
   feature availability, and the real historical nuisance table alone
   lacks roof/receiver. Final game-score verification and ambiguous
   fake-action labels still need an independent source.
2. Confirm the real prospective extraction and prediction pipeline
   produces sealed forecasts **at decision time**, before the coach's
   actual action and any observed yards/points/wins are available.
   Offline manifest claims cannot replace independently verified seals.
3. Agree on and retain signed immutable model and data provenance.
   The versioned numerical nuisance coefficients are preserved, but
   signed approval and forensic source verification are still missing.
4. Run candidate/production action-policy evaluation only when the
   preregistered **2027 REG season is complete** and the separate gates
   permit it. Use Q4 native inputs and original pinned 2027 policy
   exactly, without tuning on future outcomes.

**Decision:** This completes the **synthetic integration rehearsal**;
Issue #8 stays open until actual real-data/time-of-forecast provenance
is reviewed and the 2027 procedure is securely frozen. No claim of
causal coaching benefit, reliable unusual GO, or calibrated Q4 site
effects has been established.
