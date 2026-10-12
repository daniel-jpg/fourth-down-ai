# Historical fourth-down source-field coverage is NOT snap-time provenance

**Run:** [Historical pre-snap evidence inventory](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38209839017), October 12, 2026 UTC.
**Frozen input:** data/fourth_downs_raw.parquet, Git blob SHA1 41ae23de12913f0bf06ac608ad4716ae48d75df8.
**Policy:** No 2026/2027 NFL outcome downloads, no model fitting, no protected Steps 66/83, no live app changes.

[Auditable code](../src/historical_prespnap_evidence_inventory.py) projects whitelisted historical 2014–2025 regular-season fourth-down data and reports whether columns/values EXIST in the retrospective archive, **never that the program obtained those values prior to an actual decision**.

## Coverage

| Historical population | Count |
| --- | ---: |
| Complete REG game scores independently verified against ESPN | 3,151 games |
| All REG fourth-down source rows, including overtime | 47,942 |
| Regulation fourth-down source rows (Q1–Q4, down 4) inspected here | **47,626** |
| Difference: nonregulation / overtime fourth-down source rows | 316 |

The 47,626 regulation fourth downs are source candidates, **not** the restricted 44,507 normal-action nuisance-training sample. These populations differ due to exclusions, eligibility and other modeling requirements.

| Historical source field category | Nonmissing 2014–2025 REG regulation fourth downs | As-of-snap independently certified? |
| --- | ---: | --- |
| Game/play IDs, possession, designated home/away, week, down, quarter | 47,626 / 47,626 (100%) | No |
| Game-seconds remaining, field position, distance to go, goal-to-go, score and both timeouts | 47,626 / 47,626 (100%) | No |
| PBP location label, roof, stadium name, stadium ID | 47,626 / 47,626 (100%) | No |
| Playing surface | 46,971 / 47,626 (98.6%; 655 missing) | No |
| Direct second-half receiving-team / coin-toss winner/choice fields | **No such columns in raw source** | No |
| Independent source observed/ingested UTC timestamps | **No such columns in raw source** | No |
| Independent forecast issuance/witness UTC timestamp, snap-time attestation | **No such columns in raw source** | No |
| Kicker player identifier (retrospective event field) | 11,117 / 47,626 (23.34%) | **No — potentially post-play** |

A populated historic game-date or play-time string is **not** an independent real-time observation timestamp. Even though roof, stadium and field state have 100% retrospective coverage, the archive does not attest that the values were available to the policy BEFORE the snap, that a movable roof had that state at that exact moment, or that scores/timeout updates had reached the collector before the coach's call.

The historical kicker-player ID is largely associated with the subsequently **observed attempted kick** and is prohibited as an automatic prior for choosing FIELD GOAL. The engine currently uses the pinned default kicker prior when no independently available individual pre-snap record is supplied.

### Rows by season

| Season | Q1–Q4 REG fourth downs |
| --- | ---: |
| 2014 | 3,942 |
| 2015 | 3,969 |
| 2016 | 3,899 |
| 2017 | 4,044 |
| 2018 | 3,758 |
| 2019 | 3,872 |
| 2020 | 3,617 |
| 2021 | 4,021 |
| 2022 | 4,084 |
| 2023 | 4,284 |
| 2024 | 4,078 |
| 2025 | 4,058 |
| **Total** | **47,626** |

## What is now executable

- [Retrospective field inventory](../src/historical_prespnap_evidence_inventory.py) verifies the original pinned parquet Git object hash, limits the source to 2014–2025, projects explicit fields, distinguishes nonmissing from snap-time-certified evidence, reports per-year coverage and explicitly segregates action/result/next-play fields. It never fetches historical or future games.
- [Synthetic provenance/Ed25519 sealing rehearsal](../src/synthetic_decision_time_seals.py) requires three explicitly fabricated source feeds, checks data/hash consistency and freshness limits, binds the frozen model forecast, verifies separate collector and witness keys plus timestamp ordering, and refuses label joins until verification. The [sealing report](2027_decision_time_provenance_and_witness_rehearsal.md) describes the trust limitations.
- Neither historical completeness nor synthetic signatures create trustworthy live source timestamps. The real-time capture and independent witness service remain unimplemented and unapproved.

## Blocking missing external evidence (before 2027 Week 1)

1. Licensed/approved real-time source with actual provider publication time AND trusted collector receipt time for play state; data transport delays measured against independent snap timestamps.
2. Physical stadium designation plus independent dynamically updated roof-open/closed evidence, not just backfilled PBP game metadata.
3. Pregame coin toss / decision of which team will receive at the second-half kickoff, mapped to **the possessing offense**; not inferred from a later kickoff play.
4. Real kicker availability and identity before any fourth-down decision, or an explicit commitment to the pinned default prior and documented confounding limitations.
5. Separate, externally trusted signing and timestamp witness, physically independent credential custody, durable read-only receipt archive and independently witnessed snap times.
6. A reviewer-approved source/model provenance manifest, plus prospective no-coverage/exclusion accounting and pre-season lock. No early 2026 holdout evaluation and no 2027 game-outcome testing.

**Readiness: REAL PROSPECTIVE FORECAST ATTESTATION BLOCKED.**
This report distinguishes retrospective column completeness, the narrow 2014–2025 ESPN final-score cross-check, and a 2027-shaped synthetic signature rehearsal. None of these establishes that a real observed NFL fourth-down decision was measured and sealed in real time.
