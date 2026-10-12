# 2027 prospective fourth-down live-data provider evaluation and offline Pulse adapter

**Prepared October 12, 2026 before the 2027 NFL regular season.**
**Status: PROVIDER PROCUREMENT AND EXTERNAL CLOCK TRUST REQUIRED — NOT PROSPECTIVE READY.**
**Frozen decision policy:** 31176d50af25db9dd42dae07e9b17cb5c2b930b4. Protected 2026 kickoff Steps 66/83 untouched. No prospective game outcomes inspected or models retrained.

## Recommendation: investigate Sportradar Official NFL Push Pulse first

Primary documentation: [NFL Push Pulse](https://developer.sportradar.com/football/reference/nfl-push-pulse) and [vendor sample stream](https://api-docs.sportradar.us/NFL/NFL_v7_Push_Pulse_Example.json). Sportradar explicitly describes messages before/during/after each play, including players lining up, snap/ball start, game clock, down, location and possession. **Push Pulse requires a paid Realtime plan with Pulse entitlement; it is NOT currently connected.** The official live play-by-play REST endpoint has a 3-second in-progress TTL, but a play result can be published after the snap, so it is insufficient on its own for proving true pre-snap information.

Secondary option: [SportsDataIO NFL API](https://sportsdata.io/developers/api-documentation/nfl) documents live play-by-play, a one-minute game-play endpoint, and a 3-second Play By Play Delta endpoint. Delta updates are **plays changed** in the last 1–2 minutes and may reflect post-snap or corrected outcomes; do not infer pre-snap availability or independently attested snap timing from update cadence. Subscription and legal storage rights must be checked. Historical ESPN/nflverse data remain adequate for retrospective outcome checks, **not** contemporaneous witnesses.

### Critical wire-format trap found in official Sportradar Pulse example

A Pulse event with event_type=players_lining_up can carry a field named wall_clock that Sportradar documents as **the start of the preceding play**, refreshed when an event message is sent. The event's created_at is the provider message creation time; neither automatically establishes when the *next* snap occurred, nor when our application received the update. Do NOT treat wall_clock as current snap time or backdate predictions. The sample payload also contains start_situation and end_situation; use **start_situation ONLY** for decision-state location and down, never current play results. Supplier game/play IDs need a formal mapping to the later independent final-results source.

## Implemented: safe documentation-shaped adapter, not network integration

[Offline synthetic Pulse adapter](../src/sportradar_pulse_synthetic_adapter.py) translates **only explicitly fabricated, pre-snap v7-shaped players_lining_up messages** into the exact [frozen 2027 snapshot contract](../src/frozen_2027_policy_bridge.py). It uses only a conservative whitelist of pre-play starting-situation, game-clock, score and timeout fields, plus separately provided fabricated physical stadium/roof and pregame coin-toss evidence.

- Requires inprogress/regular game, pre-play players_lining_up pulse type, v7 metadata and proper game/team/play identities. Rejects actual play/result fields, unknown phases, post-play metadata, nonempty play details, contradictory score/clock/field/possession, Q5/OT, undefined roof, timeouts and kickoff receiver.
- Converts provider field side and yard number to opponent-endzone yardline_100; uses the possessing offense's **own** score differential, timeouts and site. A physically relocated designated-home game is marked NEUTRAL rather than assuming home stadium.
- Uses provider created_at and independent **claimed collector receipt and capture times**, not pulse wall_clock. Requires a documented freshness ceiling (currently 40 sec play; 60 min venue; 6 hours coin-toss in synthetic tests). Those thresholds are **engineering trial heuristics**, not externally validated or a guarantee the opportunity occurs before snap.
- Converts observed source and receipt timestamps and separate fabricated venue/coin-toss records into the existing [three-source provenance and Ed25519 signing rehearsal](../src/synthetic_decision_time_seals.py). The exact native Q4 kickoff-policy orchestration remains frozen and unchanged.
- The [test suite](../tests/test_sportradar_pulse_synthetic_adapter.py) replays a fabricated pre-play lineup through the registered frozen engine (at reduced smoke-test precision), synthetic pre-snap source validation, separate collector and witness signing, mandatory before-snap timestamp comparison, then a **later** fabricated observed action/final game result join. No real game, provider key, payment, live session or trusted timestamp is involved.

The adapter's mode check **always rejects real/prospective data**. Replacing it with a live connector requires review, entitlement, source license, stream-by-stream recorded timing validation and a separately deployed independent witness. Never set trusted=true because a message is labeled pre_snap.

### Prevent selective fourth-down omission

[Offline independent-opportunity coverage audit](../src/sportradar_pulse_synthetic_adapter.py) accepts a separate fabricated roster of expected (game, play, snap_at_utc) opportunities and counts **every one**, even if missing Pulse, stale or malformed. Results distinguish READY_FOR_SYNTHETIC_FORECAST, MISSING_PULSE_EVENT, PLAY_IDENTIFIER_MISMATCH and INCOMPLETE_OR_INVALID_PRESNAP_SOURCE, with explicit missing/invalid totals and a locally SHA-256-chained ordered omission ledger.

**Limit:** That ledger is only locally chained and not externally append-only; a malicious sole operator could rewrite and rehash it. Independent signed receipt of the running ledger root, external opportunity census, secure read-only storage and reviewed public keys are needed before authenticating prospective no-coverage counts.

## Measured registered-policy computation cost (synthetic, CI hardware only)

The [full registered-budget benchmark](../src/frozen_2027_policy_latency_rehearsal.py), [tests](../tests/test_frozen_2027_policy_latency_rehearsal.py) and [passing CI run](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38273657698) used three repeats per scenario on a GitHub-hosted Linux runner on October 12, 2026 UTC. Model load + frozen asset checks took **125.85 ms** in that process. Actual machine timing varies.

| FABRICATED state | Exact registered budget | Median compute only (3 runs) | Maximum of 3 runs |
| --- | ---: | ---: | ---: |
| Q1 fourth-and-4 at opponent 6 | Adaptive 300 | 126.960 ms | 157.073 ms |
| Q2 known halftime receiver, midfield-ish territory | Adaptive escalated 1,200 | 888.261 ms | 925.121 ms |
| Late Q4 fourth-and-4 at opponent 6 | Native kickoff correction 1,200 | 408.092 ms | 415.398 ms |
| Late Q4 fourth-and-3 at own 5 | Native kickoff correction 1,200 | 416.011 ms | 419.517 ms |

**Interpretation:** Compute time itself appears sub-second for these four sampled states on this particular runner. These tiny samples are **not** an empirical 95th/99th percentile. They exclude Sportradar operator delay and message transport, pre-forecast source reconciliation, live traffic queueing, distributed witness network/signature latency, independent snap timing and server load. A 40-second maximum source-age heuristic does not imply the play is still live or that a forecast can be sealed before the snap. Before even considering a live trial, benchmark a realistic multi-game concurrent feed replay with externally time-referenced snap events and account for ALL missed calls in the denominator. The timing code collects **no** live or historical holdout outcomes and changes no registered model.

## Owner-ready procurement and witness acceptance packet

The [vendor RFI, licensed Pulse evaluation request, independent timestamp/snap evidence contract, and prospective go/no-go checklist](2027_live_feed_vendor_and_witness_procurement_packet.md) were added on October 12, 2026. These are procurement/review materials only: no vendor was contacted, no subscription or TSA was activated, and no real pre-snap forecast has been independently witnessed. The repository must remain **NO-GO** for real prospective collection until an external provider, independent witness and snap reference are demonstrated.

## Blocking real-2027 requirement matrix

| Requirement | Today | Required evidence before real prospective start |
| --- | --- | --- |
| Subscribed pre-play lineup stream | NOT CONNECTED | Sportradar Push Pulse Realtime entitlement, contractual right to store source bytes, v7 payload samples and an approved client |
| Empirical decision-time timeliness | NOT MEASURED | A complete vendor-controlled simulated replay or separately permitted prospective dry run measuring per-snap event creation, transport receipt, forecast computation and witness timestamp, including tail latencies |
| External snap timestamp | NOT TRUSTED | Genuine independently witnessed snap event mapped to the exact provider game/play ID; Pulse wall_clock of previous play is not acceptable |
| Provider receipt and timestamp custody | TEST-ONLY | Actual independently administered witness clock/signing keys, KMS/HSM or equivalent, archived external log receipt BEFORE each snap |
| Physical game venue and roof | HISTORICAL ONLY | Pregame stadium identity and contemporaneous dynamic roof state; do not infer from designated HOME field |
| Known Q2 halftime receiver | NOT AVAILABLE | Pregame coin toss result from a separate legitimate feed, mapped to offense perspective, no inference from second-half kickoffs |
| Kicker availability and prior | PINNED DEFAULT | Independent pre-snap identity/fitness information or explicitly lock original engine default; never backfill actual kicker from attempted FG |
| Missing source/latency/stream gaps | FABRICATED AUDIT | Independent per-game expected fourth-down census, no silent deletions, authenticated append-only exclusion record |
| Sealed frozen model outputs | SYNTHETIC ONLY | Proof of exact registered app's Q1–Q3 300/1,200 and native Q4 1,200 path, source/asset SHA custody and forecast receipt before the live snap |
| Independent review + permission | NOT APPROVED | Written preseason signoff, source-provider approval, externally timestamped hashes and separate release authorization |

### Questions to confirm with Sportradar before subscribing

1. Is the precise **players_lining_up** event always emitted for every REG fourth down, before the ball is snapped? What is the observed median/95th/99th-percentile separation of *provider creation* and *client receipt* from actual snap by venue?
2. Which licensed product supplies time-stamped pre-play score, yardline, distance, both timeouts and possession across corrections, and how can we detect duplicate/out-of-order or missing stream events after reconnecting?
3. Does the pregame feed expose the independently determined **second-half receiver** (including deferrals), and can we retain the raw coin-toss record as evidence?
4. Does stadium_status report a roof **open/closed at snap** (not just nominal stadium classification), with timestamped changes? How are relocated games and designated home clubs represented?
5. Can we obtain trustworthy and stable provider play IDs and separate actual-snap event/time references, for use in a third-party witness/receipt system?
6. What data retention, academic research, rate limit, replay/simulation, historical sample and derivative model-use rights apply? Can we inspect a documented fabricated v7 sample offline before purchasing?
7. Can the provider issue independent source signed receipts or verified server timestamps; if not, which independently operated witness service and storage mechanism will be contractually allowed?

## Proposed go/no-go gate, before 2027 Week 1

**No-go** until we can show, on a fully enumerated and independently witnessed (not opportunistically filtered) trial, that expected fourth-down opportunities are either sealed strictly before their actual snap or excluded with explicit reasons, with no unreviewed source field, key or missing-data override. Preserve the failed/late examples. Approve the exact contract/manifest publicly or with a signed private archive **before** first 2027 REG game; do not adjust thresholds after seeing 2027 outcomes.

**Current result:** Documentation feasibility is identified and an offline adapter with strict source-availability defenses is implemented. **Real provider access, timeliness, external witness and snap-time validation remain UNVERIFIED.** This is an architecture/procurement milestone, not a validated coaching product or causal GO-vs-FG effect.
