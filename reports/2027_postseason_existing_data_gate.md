# 2027 NFLverse postseason gate: full schedule, independent scores, one-shot evaluation

**Prepared 2026-10-12, before any genuine 2027 NFL games.**
**Chosen existing provider: nflverse / nflreadpy.**
**Status: FABRICATED-ONLY engineering rehearsal.**
**No Sportradar, subscription, vendor outreach, model retraining, or live timestamp service.**

## What has been implemented

The project now includes:

- [Full-season admission contract](../src/postseason_2027_admission.py): exactly **272** 2027 REG games and **32** NFL franchises playing **17** games apiece; weeks 1–18 present, **16 Week-18 finals**, no duplicate game IDs or teams playing twice in a week. This matches the expected 17-game NFL schedule, and intentionally fails for future rule changes until separately reviewed.
- **Cross-source score verification:** one set of game-level scores shaped like NFLverse PBP, and another *separately supplied* set shaped like ESPN final scoreboards. Every game must match on season, week and normalized home/away teams, and the NFLverse home final margin must equal independently formatted home-minus-away points. An extra game, absent final, duplicate provider event, mismatched pairing, inconsistent score, missing Week 18 or unknown team fails completely; individual errors never silently become missing values.
- [Strict one-time local evaluation rehearsal](../src/postseason_2027_once_only.py): validate the label-free forecast set and full candidate-plus-abstention denominator, create a NEW output directory exclusively, write a content-hash **forecast_seal.json**, and **only then** touch the observed coach calls and independently crosschecked postgame final scores. Build the cohort with the existing [separate cohort builder](../src/fourth_down_cohort_builder.py), evaluate using the [existing offline study evaluator](../src/fourth_down_policy_study.py), and exclusively write a report plus done.json hash receipt. A second run or partial failed run at that path is rejected.
- [Entire-season fabricated fixture and CLI](../scripts/rehearse_2027_postseason_once.py): a 32-team circular rotation creates 272 fictitious games over weeks 1–18, with 17 appearances per team, a second independent-looking **invented** scoreboard, one fake forecast and an explicit missing-receiver abstention. It tests label joins and failure gates rather than the model's empirical accuracy.
- [Negative regression tests](../tests/test_postseason_2027_once_only.py), including incomplete 272-game coverage, incorrect finals, duplicate team-week entries, wrong providers, malformed IDs, bypass attempts and a test that specifically checks the forecast seal exists **before** any final-score or observed-action list is inspected.
- [CI](../.github/workflows/2027_postseason_fabricated_gate.yml) with **read-only GitHub permissions**, no cron, no real 2027 loader, no hidden schedule, and a deliberate failed second execution check. As another independent baseline, the CI replays the previously published 2014–2025 ESPN archive against **3,151** already checked-in historical REG game margins; no new historical or future games are downloaded.

### A critical evidentiary distinction

The new 272-game fabricated score crosscheck tests **code structure**, **not** independent ESPN data authenticity. The invented provider literal is intentionally named FABRICATED_ESPN_SCOREBOARD_FIXTURE. Its audit output always sets authentic_espn_response_verified=false, external_source_independence_certified=false, 2027_season_currently_complete=false, and future_evaluation_authorized=false. Merely passing this fixture cannot unlock real NFL games.

The previous 2014–2025 score archive was genuinely retrieved from ESPN separately and is a useful *historical* integration test. It cannot establish that the 2027 source is complete or verify its future scores.

## Current no-bypass gate

Real 2027 mode is **unconditionally disabled**, even if a JSON caller sets season-complete=true or supplies fabricated 272-game evidence.

- Existing production app and frozen models are not changed.
- The 2026 Step 66/83 once-only test remains untouched and is never invoked from these modules or workflows.
- The synthetic-only scorer cannot read a real NFLverse 2027 game-ID; it will not download nflreadpy data, access any ESPN endpoint, or call an external witness.
- A forecast marked pre_snap inside the *fabricated-only* schema must not be represented as a real forecast issued before an actual football snap. The 2027 future-season study chosen by the project owner is **postgame reconstruction on a held-out season**.

## What must actually happen after the 2027 REG season ends

This is a **future release checklist**, not a setting to flip today:

1. In early **2028**, confirm independently that every game in the 2027 18-week REG schedule has reached final status, including the actual recorded Week 18, cancellations, postponements or potential ties. Validate team counts and source coverage against the official schedule; this year's real game counts are not established by mock fixtures.
2. Obtain and archive nflreadpy 2027 data **only at that post-season gate**, with original file bytes, retrieval time, precise dataset/schema version and SHA-256. Separately obtain full original ESPN final scoreboard responses for *all* 18 weeks, each with its own response hash, source URL, retrieval time, unique event IDs and nonfinal exclusion accounting. Preserve a reviewed game-ID/matchup crosswalk.
3. Re-run the source and independent-final comparison on actual authorized game records. Record missing fourth-down opportunities and unsupported pre-play context (notably Q2 receiving team, roof/venue, fake or botched plays). No game should be marked final merely from repeated PBP result columns.
4. Use the **unchanged pinned 2027 fourth-down policy**, 2014–2025 nuisance parameters, proper Q1–Q3 adaptive 300/1200 and Q4 1200 native kickoff procedure to compute forecasts on postgame-reconstructed preplay states. Write a content-addressed forecast table **before** joining observed play calls or outcomes, with exact denominators/abstentions.
5. Install and approve a **separate real-data admission release**. It must enforce the reviewed ESPN/PBP raw source hashes, 2028 temporal eligibility, completed-season manifest, source-schema compatibility, unchanged model/feature rules, non-rerunnable durable Git tracked lock, and one authorized final score. **Do not weaken** the existing fabricated module to do this.
6. Run factual-action Brier/calibration and the locked cautious overlap/DR diagnostics once. The existing evaluator's prospective mode is not a truthful label for retrospectively generated predictions; a separate reviewed reporting mode must explicitly state **future-season retrospective reconstruction**, and not claim before-snap publication or causal benefits.
7. Archive versioned reports and hashes for reproducibility, retaining any null/unsupported decisions in the denominator. Do not promote a model, change the Streamlit recommendation, or erase uncertainty notices from observational results alone.

### Dry run today (fabricated-only)

From the repository root:

    PYTHONPATH=. python -m pytest -q tests/test_postseason_2027_once_only.py
    PYTHONPATH=. python -m scripts.rehearse_2027_postseason_once \
      --fabricated-only --outdir artifacts/fabricated_2027_postseason_once

The output directory must **not already exist**. Its files are a local-only one-shot rehearsal. A fresh GitHub Actions runner has a fresh filesystem; an artifact stored on such a runner **is not a global or permanent once-only execution lock**. The genuinely held-out future study will require a repository-level durable lock and independent release approval.

## Next implementable work before the 2027 season

Test/adapt the historical source and scoreboard *schema contracts* on data **only through 2025**, and develop a code-reviewed genuine post-season source collector that remains unable to access 2027 outcomes before the future release. There is **no need** to subscribe to Sportradar, contact vendors or modify the protected 2026 kickoff workflows.
