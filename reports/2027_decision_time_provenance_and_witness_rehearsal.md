# 2027 decision-time evidence and forecast sealing — fabricated-only rehearsal

**Built:** October 2026, before the 2027 regular season.
**Status:** Executable synthetic rehearsal and cryptographic tamper checking, **NOT** trusted real-time independent NFL data collection, actual third-party timestamp attestation or authorization of prospective scoring.
**Registered policy:** [31176d50af25db9dd42dae07e9b17cb5c2b930b4](https://github.com/daniel-jpg/fourth-down-ai/commit/31176d50af25db9dd42dae07e9b17cb5c2b930b4), including original Q1–Q3 adaptive budgets and Q4 native kickoff adjustment.
**Separate protected review:** 2026 reserved kickoff Steps 66/83 must not run early or be altered by this work.

## What is actually implemented

The [signed evidence/rehearsal library](../src/synthetic_decision_time_seals.py) consumes a **fabricated 2027-shaped pre-snap snapshot**, an already-computed [registered-engine action forecast](../src/frozen_2027_policy_bridge.py), and three separate, fabricated source-event payloads. It refuses any real/prospective source mode, forbids unexpected fields including actual coach choice, final winner, converted flag, gained yards and other post-play information, and requires the snapshot and prediction input to be consistent with all source payload values.

| Source role | Required facts | Max declared age at capture | Synthetic identity |
| --- | --- | ---: | --- |
| Play/clock feed | Game/play key, year/week, offense, Q1–Q4, fourth down, score margin, absolute regulation clock, opponent-distance yardline, distance, goal-to-go, both remaining timeout counts | 40 sec | fabricated-playfeed |
| Physical venue feed | Designated HOME/AWAY, roof status, physical stadium ID, nominal home stadium ID, explicit physical-neutral flag | 60 min | fabricated-venuefeed |
| Coin-toss / second-half receiving feed | Explicit OFFENSE/DEFENSE half receiver in Q2; nullable otherwise | 6 hours | fabricated-kickofffeed |

Source payload SHA-256 digests, the claimed provider observation and receipt times, and the exact source references all become part of the signed forecast bundle. For these synthetic fixtures the physical-neutral flag is checked against whether the recorded stadium differs from the nominal home stadium. Designated HOME and physical home field are **different facts**, as independently documented by four 2020–2021 historical relocated games; a model's site effect cannot be inferred automatically from a provider's designated-home flag.

The [synthetic snapshot validator](../src/frozen_2027_policy_bridge.py) additionally enforces the exact football quarter/clock range, opponent-distance yardline, goal-to-go, Q2 second-half receiver, approved roof values, offense team, physical home/away/neutral orientation, and frozen eligible normal-action estimates. No live source data are ever read by the sealing code.

### Seal life cycle (separate stages; labels joined afterward)

1. Acquire all three source events and build a complete snapshot from their values **before** the play. Compute the frozen policy and development-only nuisance forecasts. The bundle binds all inputs plus forecast, source event hashes and the asserted capture/forecast issuance UTC times. Source-event timestamps must be nonfuture, no later than capture, and within per-role freshness windows.
2. A **collector** with its own Ed25519 private key signs the canonical bundle SHA-256 together with its asserted sealing time.
3. A distinct **independent witness** verifies that collector's public signature and signs the hash of the entire collected packet (bundle plus collector signature) with a separately controlled Ed25519 private key and an asserted witness receipt time. Its signature and key fingerprint are part of the witness packet.
4. **After** the snap, the verifier requires the signed sequence source-observed <= source-received <= captured <= forecast-issued <= collector-sealed <= witness-received < independently sourced snap. Both signatures are checked against externally provided **trusted public keys**, not keys supplied by the packet itself. A late seal, changed score/state/roof/forecast, forged key/signature, incorrect source, missing Q2 receiver or post-snap field invalidates verification.
5. Only after receipt checks, [join observed play and final score as labels](../src/fourth_down_cohort_builder.py). The label data are NEVER an input to the decision state or the signed policy prediction. This remains a synthetic-only join.

No private keys, real ESPN/NFL events, 2026 reserved data, 2027 game outcomes, model refitting, app writes, release tagging or automatic future tasks are added to this implementation.

## Test evidence

Run locally in a development environment:

    python -m pip install -r requirements-dev.txt
    PYTHONPATH=. python -m pytest -v tests/test_synthetic_decision_time_seals.py

The [dedicated CI](../.github/workflows/2027_synthetic_decision_time_seals.yml) uses new **in-memory ephemeral independent signing keys** for each test, and exercises source age limits, clock/venue/half-receiver consistency, post-snap contamination, hash tampering, swapped/missing public keys, altered signed audit metadata, missing sources, receipt timing, and the full real pinned Q1 engine forecast (on a reduced fabricated-only simulation budget). A signed case is then joined to a separately supplied **fabricated** observed coach action and final winner after verification.

The function verify_witnessed_before_snap returns real_independent_provenance_certified=False **even if all cryptographic checks pass**. An Ed25519 signature proves only that a key signed a particular byte string. It does not independently prove the issuer's identity, the truth of the source facts, or the claimed timestamp, which can be backdated by an uncontrolled signer. In the tests, **all timestamps and snap times are fabricated**. A pair of simulated keys held by one development process does not constitute a genuine independent witness.

The pinned app's full numerical precision is **300/1,200 MC trials (Q1–Q3 adaptive) or 1,200 MC trials with the native Q4 kickoff correction**. The signing smoke tests use cheaper fabricated budgets and cannot be used as validated real-world prediction probabilities.

## Real deployment blockers before prospective 2027

The following are not solved by signatures on synthetic data:

1. **Reviewed real-time football event provider:** A service that can supply game/play identifier, clock, scores, field position, timeout counts, and each field's availability time *before the snap*, with immutable original observations and reconciliation of delayed/out-of-order updates. Retrospective nflverse play-by-play rows are not an equivalent decision-time capture.
2. **Physical venue / movable roof records:** Independent stadium identity and contemporaneous *open versus closed* roof state as it was at the decision. Do not use the nominal home-team label to substitute for physical site or assume every retractable roof is closed.
3. **Kickoff-direction evidence:** Pregame coin-toss source that identifies the known second-half receiver from the possessing offense's perspective. Deriving the receiver from the actual second-half kickoff after it happened would leak future information.
4. **Kicker context:** Confirm pre-snap kicker availability, identity and historical prior or explicitly record the original pinned model's **default kicker prior**. This baseline cannot be retrospectively upgraded using who actually attempted the later FG.
5. **Truly independent timestamp and keys:** A separate pre-snap collector runtime and a separately administered, externally trusted receipt/timestamp service with trustworthy clock, secure private-key custody (KMS/HSM recommended), published public-key fingerprints, unmodifiable retained receipts and verifiable source observation times. The after-snap actual snap timestamp must come from an independently reviewed authoritative source.
6. **Signed archival and freeze review:** Immutable manifest of original source bytes, event IDs, timestamps, receiver/roof evidence, exact model artifacts, pinned code SHA, signed receipt chain, failure/no-coverage counts and read-only storage. An independent reviewer must approve the freeze before operational collection, not after peeking at outcomes. No auto-promotion is allowed.
7. **Run 2027 REG once afterward:** Do not evaluate held-out games until all 2027 REG finals exist and separately registered gates permit the one-time evaluation. Even that analysis will not automatically identify causal benefits from unchosen GO/FG/PUNT actions.

**Readiness decision: BLOCKED FOR REAL PROSPECTIVE INFERENCE.**
This rehearsal proves source consistency, signature verification and timing ordering **conditional on fabricated asserted timestamps and signing keys**. It provides an engineering foundation, not an external time attestation, complete real NFL source coverage or proof of no data leakage in a real-time future service.
