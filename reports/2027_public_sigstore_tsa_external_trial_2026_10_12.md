# Real Sigstore timestamp authority proof for a *fabricated* fourth-down packet

**Observed:** 2026-10-12 17:03:52 UTC.
**Workflow:** [one-shot public TSA CI #38281700754](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38281700754) — **SUCCESS**.
**Source commit:** [9abc9f65d1fd948b187cd3676040dcaa6d3c8d69](https://github.com/daniel-jpg/fourth-down-ai/commit/9abc9f65d1fd948b187cd3676040dcaa6d3c8d69).
**Result type:** external timestamp signature/nonce/commitment verification on INVENTED source bytes. **Not** a real NFL snap/forecast attestation, not 2027 study admission, not a production model or app change.

## What was actually independently demonstrated

A single explicit Github Actions workflow obtained **one** real RFC 3161 token from `https://timestamp.sigstore.dev/api/v1/timestamp`, using a synthetic-only domain-separated SHA-256 commitment. The job first installed a released Sigstore Cosign client (`sigstore/cosign-installer@v3.10.0`) and ran `cosign initialize`; the resulting verified TUF target `trusted_root.json` provided the Sigstore public TSA's certificate chain. The script **did not download an unverified root as its trust anchor**.

The unchanged [RFC3161 synthetic verification module](../src/rfc3161_synthetic_tsa_trial.py) then checked the real token **against both its own original nonce-bearing SHA-256 timestamp query and the exact canonical fabricated data commitment**, using `openssl ts -verify`, the TSA's leaf certificate and the TUF-authenticated root. CI finished successfully. `openssl ts -reply -text` showed the issued timestamp (UTC).

Proof metadata from the successful job's printed JSON:

| Field | Observed value |
| --- | --- |
| token issuance | **Oct 12 17:03:52 2026 GMT** |
| status | `SYNTHETIC_RFC3161_TOKEN_VERIFIED_WITH_CALLER_TRUST` |
| commitment SHA-256 | `6f65599688bd41fe424b7fb2b8785df302239e0cf6d2cece4f9e3a882330c37b` |
| response SHA-256 | `263d18813b9bd5113112e7ade2da08d8c6972f38b145bc343339b50aae463c24` |
| TUF verified trusted-root JSON SHA-256 | `6494e21ea73fa7ee769f85f57d5a3e6a08725eae1e38c755fc3517c9e6bc0b66` |
| TSA root certificate DER SHA-256 | `2aca8fea5d3ce48b01cc77076293c280e6c23ffe44034757ee7833ca9f45d633` |
| TSA leaf certificate DER SHA-256 | `85f927bc07ab62cac3b44356c10efc81b2c6883fda7ab9e6d870d9d13acd05b7` |
| token nonce, original data, and TSA signature | **verified true** |
| `trusted_root_from_cosign_verified_tuf` | true |
| `trust_anchor_externally_approved` | **false** (independent 2027 study review has not occurred) |
| `independent_snap_timestamp_verified` | **false** |
| `real_2027_prospective_collection_approved` | **false** |
| `real_provider_connected` | false |

**Reproducibility limit:** Only the verification report and hashes were archived in GitHub Actions. The raw RFC 3161 response, query, signed source bytes and TUF cert PEM were *not* retained as portable artifacts by this first one-time run. Therefore the token cannot be independently reverified outside the logged job from these hashes alone. A future separately authorized receipt-custody trial would need to store the complete public synthetic proof package. Do not claim this log equals immutable independent storage.

## What this does and does not solve

- **DONE:** Existence of a responding external public TSA and functioning authenticated TUF certificate retrieval, nonce/imprint/signature checking, on one fabricated example in Github Actions.
- **NOT DONE:** Vendor-supplied fourth-down pre-snap updates; per-snap source events; cross-provider game/play ID mapping; validated independent real snap timestamps; signed independent *opportunity census*; quantile latency/coverage statistics; approvals for actual 2027 collection. A TSA's token time says nothing about whether a game clock or score was truly available before a real snap.
- **NOT DONE:** Academic/research licensing, cost, rights to retain source events/receipts or any vendor outreach. Sportradar [NFL Push Pulse](https://developer.sportradar.com/football/reference/nfl-push-pulse) still requires a Realtime plan with Pulse entitlement.
- **OUT OF SCOPE:** Repeated third-party requests, unsupervised live feed polling, coaching recommendation changes, 2026 Step 66/83 evaluation, checking previously reserved outcomes or accessing genuine 2027 game events. The workflow has no schedule or future-data dependencies; it is triggered by an explicit file creation/edit or a human-triggered GitHub workflow dispatch.

**Next decision:** seek licensed vendor simulation/archival sample and **independently review** both forecast trust root and a genuine snap timing source. Even a successfully timestamped fabricated forecast is only an engineering test. Keep [Issue #8](https://github.com/daniel-jpg/fourth-down-ai/issues/8) in **NO-GO** state for real prospective collection.
