# 2027 fourth-down evaluation using the **existing nflverse / nflreadpy data source**

**Recorded:** October 12, 2026, before the 2027 NFL regular season.  
**Owner preference:** Reuse the existing data-acquisition pipeline. **No paid vendor, Sportradar email, new subscription, or live receipt infrastructure is needed for this objective.**  
**Status:** Future held-out **retrospective reconstruction / out-of-time** evaluation plan. It has not been run, no 2027 rows exist yet, and the protected 2026 Steps 66/83 must remain untouched.

## Evidence that the project already has this pipeline

- [Original dataset builder](../src/01_build_fourth_down_dataset.py) calls `nflreadpy.load_pbp(SEASONS)` for 2014–2025 and extracts fourth-down observations from the resulting Polars data.
- [Existing 2026 kickoff validation code](../src/66_forward_2026_continuation_wp.py) already calls `nflreadpy.load_pbp([2026])` under the unchanged, once-only research gate. Its [weekly GitHub workflow](../.github/workflows/forward_2026_validation_once.yml) is **already scheduled**; never accelerate it or open its reserved 2026 evaluation data early.
- The [2014–2025 historical cohort extractor](../src/historical_fourth_down_cohort.py) already separates game-state fields from **observed action** and **final winning outcome**. [Historical snapshot audit](shadow_historical_development_intake.md) verified 44,507 supported historical normal-action decisions with explicit exclusions. It deliberately refuses future data now and must not be bypassed.

## What we can test **without** a live feed

When the 2027 regular season is over and final data are available from the **same nflverse/nflreadpy source**, reconstruct the **pre-play game state** of each observed eligible fourth down. Apply the preregistered frozen decision engine and historical nuisance estimators without retraining or choosing features based on results, then compare:

1. For the **action the team actually took**, predicted win probability versus the team's eventual actual game win/loss: per-action coverage, Brier score, calibration, fixed subgroups and game-cluster intervals.
2. How often the model and actual coach agree, and the model's *estimated* action advantages. These are model outputs, not observed gains from a choice the coach did **not** make.
3. An exploratory observational one-step policy contrast **only** with the existing prespecified propensity/overlap, ESS, disagreement and cluster-bootstrap safeguards; never claim it is causal.

This is genuine **future-season, out-of-time testing of a frozen model**, but the individual forecasts would be **computed retrospectively after games**, rather than captured and externally witnessed in real time. Call the study “retrospective reconstruction on a prospectively reserved 2027 season,” **not** an independent pre-snap recorded/live deployment. The source `nflverse` alone cannot certify the *timestamp when the model's forecast was generated*.

## Leakage, data-quality, and future-holdout conditions

- Lock the source/version contract, exact allowed feature whitelist, model/artifact commit (`31176d50af25db9dd42dae07e9b17cb5c2b930b4`), rules for exclusions, thresholds and 2027 scoring code **before** inspecting the evaluation season's game outcomes.
- Start from source-reported **pre-play fields only** (down, time, distance, field location, current scores and timeouts as of that play, offense/defense and supported venue context). Treat `play_type`, `desc`, yards gained, conversion, touchdowns, `epa`, later play state and final scores as **forbidden forecast inputs**. The subsequently observed action and final winner belong in separately keyed **labels**.
- A source column existing in completed-game PBP does **not** prove it would have been available pre-snap in real time. Screen venue/roof corrections, kickers known only because they attempted a FG, unknown Q2 second-half receiver, penalties/no-plays, clock states, unusual relocations and post-snap corrections. Where trustworthy pre-play reconstruction is impossible, record the missing/unsupported decision and abstain rather than infer the answer from what happened later.
- Use exactly one (game_id, play_id) per decision; retain a denominator of all fourth-down opportunities with row-level exclusion reasons, source hashes, source acquisition date/time and version, and game count. Group uncertainty by game.
- Keep the 2026 Step 66/83 one-time analysis separate from this study: **no code reuse that bypasses its gates, no early 2026 Week 6–18 outcome reads**. Preserve the original 2027 strictly **live, independently witnessed** protocol as an optional stronger *different* experiment; do not silently represent retrospective reconstruction as having passed that provenance requirement.
- If the 2027 data schema changes, missing pre-play fields prevent reconciliation, or a rule change invalidates model assumptions, **do not silently re-fit or patch against held-out outcomes**. Publish coverage/failure limitations and, if needed, a separate new protocol rather than adjusting results post hoc.

## Workflow when the season is over

1. Keep the 2027 dataset held out. During the season, optional *metadata-only* availability checks can see whether nflverse has released season/week files, without running a scorer or inspecting fourth-down outcomes. There is **no need** for a paid live feed.
2. After **all** eligible 2027 regular-season games are final and the study gate is met, load 2027 using the existing source API, e.g. `nflreadpy.load_pbp([2027])`, in a **new isolated 2027 artifact directory**. Never overwrite the checked-in 2014–2025 historical Parquet, frozen model artifacts, or original 2026 files.
3. Verify source season, game completeness, expected columns, unique game/play keys, end-of-game labels and the independent final-game result reference; record a SHA-256 source manifest and omissions.
4. Generate frozen action predictions **from only the allowed reconstructed pre-play columns**, without passing eventual action/result to the forecast call. Join observed action and actual winner *after* predictions are written.
5. Use the fixed evaluator, preserving low-overlap DR suppression and preregistered descriptive site/Q4/close-call/own-territory subgroups; report any differences from the original strict prospective witness study explicitly.

**Current engineering action:** Do **not** download 2027 records or change the registered 2026 workflows today. The existing nflreadpy source works for offline future-season evaluation. A separate retrospective-cohort extractor/scoring workflow must be finished and synthetically tested before admitting real 2027 results. The costly externally witnessed live-feed path is **optional**, not the prerequisite for this source-based study.

## Bottom line

- **Existing source sufficient:** Historical data, post-game fourth-down analysis, out-of-time 2027 calibration/reliability research and limited observational policy diagnostics.
- **Additional real-time source needed only for stronger claim:** Proof that each actual prediction was collected, signed, independently witnessed, and issued before its specific live NFL snap.
- **No claim either way:** Observational differences in model-estimated GO/FG/PUNT choices cannot establish the true win outcome of an action that wasn't played.
