# 2027 fourth-down action-policy cohort: synthetic-only builder contract

**Status (October 2026):** Engineering dry run. This builder is NOT a live or prospective NFL data importer, does not execute the fourth-down engine, and cannot access 2026 or 2027 observed records. There is no new official model or coaching recommendation. This is preparatory work for Issue #8.

## Separating decision-time inputs from final-game labels

The cohort builder in src/fourth_down_cohort_builder.py accepts one JSON object with four fields:

1. **manifest**: exact synthetic-only, fabricated-only source contract; policy pinned to commit 31176d50af25db9dd42dae07e9b17cb5c2b930b4; nuisance training cutoff 2025.
2. **pre_snap**: fourth-down decision-time state, original offense, game/site, candidate eligibility, frozen action-specific forecasts, the engine's highest-WP strategy and historically trained nuisance estimates. No observed action, post-play state or final-score field is permitted here.
3. **observed_plays**: factual RUN GO, PASS GO, FIELD GOAL, PUNT, fake/unclassified or no-play classification joined separately by (game_id, play_id).
4. **final_games**: final/nonfinal REG game team scores and teams, joined last, solely to derive the original offense's eventual win/loss label. Final scores must never appear in model features.

The strict pre-snap schema rejects unexpected keys, including post-snap labels. It verifies quarter/clock, down, field position, score/timeouts, team/site orientation, frozen policy commit, action eligibility, probability bounds and that the declared policy action has highest estimated WP among eligible normal actions. A normal factual action outside model eligibility is retained with null factual WP and counted as **unsupported**, rather than converted to a fabricated prediction.

The join excludes overtime, non-fourth downs, administrative no-plays, fake/unclassified observed actions, unmatched keys, unfinished games and tied finals. All exclusions retain a game/play ID and explicit reason; duplicate identifiers and conflicting teams/weeks fail closed instead of disappearing from the audit.

Outputs include deterministic sorted decision/exclusion records, an input SHA-256 fingerprint, exclusion counts and manifest plus records consumable by src/fourth_down_policy_study.py. Input list permutations can have distinct hashes even when sorted results match.

## Local dry run, fabricated records ONLY

    PYTHONPATH=. python -m pytest tests/test_fourth_down_cohort_builder.py
    PYTHONPATH=. python -m src.fourth_down_cohort_builder synthetic_input.json --output synthetic_cohort.json
    PYTHONPATH=. python -m src.fourth_down_policy_study synthetic_cohort.json --output synthetic_eval.json

Tests cover label orientation, unsupported action accounting, leakage, protected-data barriers, excluded cases, duplicate IDs, fingerprints and evaluator compatibility. This does **not** verify predictive or causal accuracy.

## Unfinished work before 2027 Week 1

- Build an independently reviewed real historical pre-snap extractor with provenance. Source-manifest assertions are not proof of leakage-free extraction.
- Independently train, calibrate, document and freeze coach-propensity and conditional-reward nuisance models on 2014–2025 development data, with chronological tests and signed model/source hashes. Their scores are placeholders in the dry run.
- Recompute frozen policy outputs from the immutable pinned snapshot and verify Q4 native kickoff state parity. This builder only checks supplied forecasts.
- Preregister/freeze Q4, HOME/AWAY/NEUTRAL, close-call and deep-own-territory descriptive subgroups before collecting future outcomes.
- Following 2027 REG Week 18, separately authorize and implement final-game provenance and sealed prospective pipeline. Prospective mode is intentionally rejected today. Do not fetch reserved 2026 Week 6–18 events or manually dispatch kickoff Steps 66/83.

See reports/2027_fourth_down_action_policy_protocol.md. This cohort software cannot establish what would have happened under an unchosen action.
