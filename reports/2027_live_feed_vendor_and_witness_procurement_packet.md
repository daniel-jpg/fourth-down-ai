# 2027 fourth-down pre-snap data: vendor and independent-witness procurement packet

**Prepared:** October 12, 2026. **Decision:** NO-GO for actual prospective collection.
**Status:** Technical RFI/acceptance contract only; no purchase, vendor trial, key, live source, third-party timestamp authority or genuine snap witness is connected.
**Immutable target policy:** `31176d50af25db9dd42dae07e9b17cb5c2b930b4`.
**Protected studies:** 2026 Steps 66/83 remain sealed; no early runs, previews, or consumption of reserved 2026 Week 6-18 outcomes. The independent 2027 action-policy outcomes cannot be read until after the completed 2027 REG season under the locked protocol.

## Decision: pursue a narrow Pulse-enabled evaluation, not a generic NFL trial

1. **Preferred candidate: Sportradar NFL Official Push Pulse.** Official [Push Pulse documentation](https://developer.sportradar.com/football/reference/nfl-push-pulse) describes pre-play `players_lining_up` and snap-related stream messages, but explicitly requires a **Realtime plan with Pulse enabled**. The standard NFL REST product/trial is *not evidence* of a Pulse entitlement. Request a separately authorized evaluation (ideally documented 2014-2025 archived or fabricated pre-snap stream) and written permission to retain source receipts/hashes before considering a paid plan.
2. **Conditional fallback: SportsDataIO.** Their [NFL Play By Play Delta](https://sportsdata.io/developers/api-documentation/nfl) is called on a 3-second interval, but reports *changed play-by-play*, not guaranteed lineup-before-snap messages. Their [NFL workflow guide](https://sportsdata.io/developers/workflow-guide/nfl) says live PBP typically updates 15-20 seconds behind TV. Do **not** equate short polling interval with pre-snap observation. Ask sales whether they have a *different*, documented pre-play source with externally checkable snap ordering. Absent proof, reject as the primary witness feed.
3. **Do not buy the wrong inexpensive tier.** [SportsDataIO's developer access comparison](https://sportsdata.io/developers) says their free trial is scrambled integration-only data and their Discovery Lab is next-day delayed. These may help parser tests or retrospective data discovery, but **cannot** prove true live pre-snap prediction. Vendor quotes for appropriate live NFL access are currently unknown.

Provider selection is conditional, **not** confirmation of sufficient fourth-down coverage, delivery lead time, license, or data validity.

## Exact provider questions (written answers required)

| Test area | Required vendor evidence | Stop condition |
| --- | --- | --- |
| Rights and cost | Pulse-specific trial/subscription quote and expiration; noncommercial/student research license; cost of NFL REG including concurrent Sunday games; rights to retain source bytes, timestamps, SHA-256 digests, replay samples and publish aggregate error/omission statistics | No written retention/research rights, unaffordable price, or wrong entitlement |
| Pre-play trigger | Exact `players_lining_up` event semantics and event schema, plus historical/fabricated example of all normal fourth downs including GO, FG and PUNT; expected rate of missing lineup events | A post-play update being mislabeled pre-snap |
| Clock proof | Provider creation/receipt reference, independently referenceable actual snap event and source IDs, time accuracy and source corrections. Supply per-event `created_at` versus later snap time for a whole trial, not handpicked examples | `wall_clock` or a self-asserted creation time treated as sufficient proof: Pulse `wall_clock` may refer to the **previous** play |
| Required game state | Game/play IDs; down/distance; field side and yardline; quarter and regulation clock; offense/defense; scoreboard; both teams' timeouts; time-effective corrections | Unrecorded values filled from post-snap play result or game-final data |
| Separate context streams | Pregame second-half receiver including deferrals, actual physical stadium/neutral relocation, timestamped current roof open/closed status, explicit original model's kicker-default handling | Nominal HOME used as physical venue; receiver guessed from future kickoff; kicker copied from resulting FG |
| Delivery and repair | Subscription example, transport and reconnect behavior, event IDs, duplicate/out-of-order/update/delete semantics, heartbeat, gap detection and catch-up REST endpoint | Silent omissions, backdated corrections accepted as earlier predictions, or no reproducible source chronology |
| External timestamp | Permission to promptly transmit only forecast/source cryptographic hashes to independent RFC 3161 or independently administered timestamp witness; source timestamps and external actual-snap reference | Witness issued after snap or self-managed keys represented as independent |
| Data provenance | Which feed owns each fact, signed/licensed original sample, allowed archive location, field-level as-of guarantees and historical-version migration notice | Vendor contract prevents independent retrospective inspection of which facts existed before snap |

### First contact: Sportradar sales (ready-to-send request)

**Subject:** NFL Push Pulse pre-snap research pilot / Realtime entitlement inquiry

Hello,

I am building an independent, noncommercial NFL fourth-down win-probability research project. We are preparing a preregistered 2027 decision-time validation study, and we need to record fourth-down *pre-snap* game states and sealed prediction timestamps before the actual snap. We are not requesting betting or wagering functionality.

Does your NFL Official API offer a trial or small research license with **Push Pulse Realtime/Pulse entitlement** (not just a normal REST NFL trial)? Could you share a documented archived 2014-2025 or fabricated `players_lining_up` / snap example with stable game/play identifiers and creation timestamps, and clarify complete REG fourth-down coverage and measured lineup-to-snap lead times?

We also need to establish available pre-snap timeouts, scoreboard, field position, independently known second-half receiver, physical stadium/roof, reconnect/correction semantics, and whether we can retain raw source events plus independent signed hash/timestamp receipts. Please provide research/student licensing eligibility, trial duration, data storage/public aggregate reporting rights and pricing for concurrent Sunday NFL games.

I can send a compact input schema and acceptance-test plan if helpful. No access to current 2026 reserved validation outcomes is needed for the evaluation.

Thank you.

### Alternative first contact: SportsDataIO sales (only if a dedicated pre-play feed exists)

**Subject:** Do you offer a pre-snap NFL fourth-down state/event feed?

Hello,

For a prospective noncommercial 2027 NFL fourth-down prediction study, we need provider-observed *pre-snap* fourth-down situations and a separately checkable snap event, not delayed results of the already completed play. I see the three-second Play By Play Delta endpoint and the published description of live play-by-play delivery delays.

Do you offer another NFL endpoint or subscription that emits lineup/before-snap messages with stable game/play IDs, full game state and documented source timestamps early enough for a third-party witness to attest our forecast before the snap? If so, could we evaluate a 2014-2025 historical Replay or fabricated stream, and could you provide rights, price and academic/noncommercial trial terms? If not, please confirm that limitation.

Thank you.

Contact paths: [Sportradar Pulse documentation / sales direction](https://developer.sportradar.com/football/reference/nfl-push-pulse); [SportsDataIO contact form](https://sportsdata.io/contact-us); [SportsDataIO testing options](https://sportsdata.io/developers). Contact forms/subscriptions require the human account owner; **no outreach has been sent**.

## Third-party witness requirement: what a signed hash can and cannot establish

An RFC 3161 [Time-Stamp Authority](https://www.rfc-editor.org/info/rfc3161/) can sign evidence that a particular **hash** existed by a claimed trusted time. We should request a quote/implementation proof for a TSP service that can return signed response tokens fast enough in a football pre-snap window, preserves certificate trust chains, and echoes nonce plus message imprint with correctly verifiable token signatures. Ideally the externally administered service archives tokens, and a separate independently sourced snap marker exists for the *same* provider game/play ID.

**Not sufficient:** our own Ed25519 signature, computer `datetime.utcnow()`, Git commit author date, vendor's time for an old play, a manually entered snap time, delayed batch anchoring, or after-game hash submission. None proves a forecast was registered by an independent party before a real snap. RFC 3161 alone proves neither that source football facts were true nor when the **football snap actually occurred**. This is a two-evidence problem: externally timestamp the forecast, and independently validate the snap reference.

The current [fabricated signing rehearsal](../src/synthetic_decision_time_seals.py) correctly labels everything synthetic and returns `real_independent_provenance_certified=False`. Do **not** turn this Boolean on based only on a contract or assumed TSA URL. A genuine provider adapter, secure third-party key custody and verifier require another separately reviewed implementation.

## Dry-run acceptance protocol before any real 2027 collection

Run **only** on vendor-authorized fabricated or previously explored 2014-2025 development material. Keep all future 2026/2027 outcomes and protected workflows unopened.

1. Freeze vendor product, exact feed version, endpoint list, rights, and a content-hashed documented original input schema. Log authentic provider `created_at`, independent collector receipt, captured state, forecast issued, collector seal, external witness token time and independent **later** snap source. Never treat the old `pulse.wall_clock` as current snap.
2. Enumerate every expected fourth down from an **independently obtained opportunity roster**, including records missing from Pulse. Record missing, duplicate, inconsistent, out-of-order, stale, post-snap, timeouts/roof/coin-toss unavailable, compute deadline miss and witness miss as **explicit outcomes**; never filter to only successfully captured plays. Keep hashed original bytes and the signed independent roster.
3. Replay at least two concurrent fabricated/previous-development games, including intentionally simultaneous fourth downs, reconnects, duplicate/out-of-order messages, clock corrections, a relocated HOME designation, retractable-roof status change, Q2 known receiver, Q4 native kickoff inputs and an unavailable/unreliable long FG. Use actual registered `300/1200` Q1-Q3 and `1200` Q4 simulation policy (no cheaper test precision in a latency acceptance report). Verify the unchanged model/policy SHA.
4. Produce a **full denominator** table: all expected fourth-down opportunities, valid delivered pre-snap candidates, forecasts computed, hashes externally acknowledged before independent snap, omitted with reason, and reasons-by-game/site/quarter. Report provider-to-collector, compute, witness and end-to-end p50/p95/p99 latency with sample counts. Keep examples of late/missing failures in the audit.
5. Before the future season, write down the minimum acceptable coverage and timeliness, source outage handling, whether unsupported states abstain, archival retention, and independent signoff. Do **not** choose thresholds or exclusions after reading future winning outcomes.
6. Require an **independent reviewer** to confirm the vendor contract, source/receipt provenance, external witness/time authority, and actual-snap reference before a live pilot can start. An optional API subscription or valid Python fixture does not override this gate.

**Acceptance invariant:** an included, potentially scoreable play must have exactly one frozen forecast, a complete verified as-of source bundle and an authentic independently timestamped forecast **strictly earlier** than an independently validated snap for that same game/play key. Everything else stays in the omission denominator. Delayed/insufficient samples are not silently corrected with post-snap data.

### Current gate status (as of 2026-10-12)

| Gate | State |
| --- | --- |
| Freeze target 2027 policy and offline synthetic tests | DONE (existing repo) |
| Historical 2014-2025 exploratory source inventory | DONE (existing repo) |
| Real pre-play provider subscription and legal rights | **NO-GO: needs owner/vendor** |
| Pulse-entitled historical/fabricated evaluation replay | **NO-GO: not obtained** |
| Independent trusted forecast timestamp receipts | **NO-GO: service/operator not provisioned** |
| Independent actual snap times and ID mapping | **NO-GO: not established** |
| Authentic per-game expected-opportunity roster | **NO-GO: not established** |
| Real concurrent full-chain latency, omissions and 2027 freeze signoff | **NO-GO: requires preceding external gates** |
| Protected 2026 Step 66/83 outcomes used here | **NO** (must remain separate) |

**Next actual owner step:** send the Sportradar trial-and-license RFI and obtain an **explicit Pulse entitlement + full 2014-2025/fabricated replay offer**; in parallel source an independently administered timestamp authority and a trustworthy snap reference. If those services are unavailable or unaffordable, leave 2027 pre-snap policy validation marked **not evaluable** rather than manufacture a prospective result. No real vendor outreach or purchase was performed while preparing this packet.
