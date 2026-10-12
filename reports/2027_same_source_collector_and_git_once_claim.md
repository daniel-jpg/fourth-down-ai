# Existing NFLverse + ESPN collector and remote once-only lock

**Prepared:** 2026-10-12. Existing NFLverse/nflreadpy source only. No 2026/2027 outcomes opened. No paid vendor or retraining.

## Implemented

- [Offline evidence collector](../src/nflverse_espn_historical_collection.py) validates the Git blob/SHA256 of original 2014-2025 PBP before parsing, replays the independently retrieved 3,151-game ESPN archive, audits the complete **2025 272-game REG schedule** (18 weeks, 32 teams, 17 games per team, 16 Week-18 games), and records exact source/feature schema hashes. This uses REAL approved historical 2025 data and cannot be invoked on 2026 or 2027.
- [Existing-source collection CLI](../scripts/collect_nflverse_espn_2025.py) has a **no-network historical archive** mode and a separately explicit **2025-only ESPN online refresh** mode. The latter reuses the established [ESPN week collector](../scripts/audit_independent_espn_historical_scores.py) with per-week response SHA256, URLs, independent source labels, completeness checks, historical year guards and a final-score sidecar. It never automatically runs online from CI.
- [Durable Git ref-lock algorithm](../src/repository_once_claim.py) creates a fixed annotated Git TAG on a shared remote with a **non-force** push. Unlike runner-local files, a remote ref is shared across independent Git clones. If another process claimed it first, the second fails even in a race. This implementation only allows a local bare Git remote and a **fabricated test tag**; it has no network GitHub access or real 2027 unlock path.
- [Historical source tests](../tests/test_nflverse_espn_historical_collection.py) check original files, real 2025 coverage, independent ESPN hashes, deterministic source manifests and forbidden future year/symlink/tamper cases.
- [Remote claim tests](../tests/test_repository_once_claim.py) simulate two separate Git runner clones, both sequential and simultaneous, and verify a single winner. Real/network remotes fail closed.
- [Read-only CI](../.github/workflows/2027_retro_existing_source_and_lock.yml) runs both test suites, generates the real-historical-only 2025 manifest, and archives it. Actions permissions are contents:read. Local test pushes cannot write to GitHub.

### Offline commands (historical only)

    PYTHONPATH=. python -m pytest -q tests/test_nflverse_espn_historical_collection.py tests/test_repository_once_claim.py
    PYTHONPATH=. python -m scripts.collect_nflverse_espn_2025 --historical-offline-only --output artifacts/nflverse_espn_2025.json

### Optional, manually requested real 2025 ESPN refresh (not CI)

    PYTHONPATH=. python -m scripts.collect_nflverse_espn_2025 --historical-espn-online-only --output artifacts/fresh_espn_2025.json

This option performs HTTP requests only for historical 2025 ESPN final-score events. The ESPN score source is never used in any model forecast features.

## What is NOT authorized yet

**No real 2027 collector has been activated** and no real 2027 tag was created. The fake source test and fabricated lock cannot certify actual future games, real independent ESPN 2027 scores or external pre-snap timestamps.

In **early 2028**, after all REG 2027 games become final, a reviewed release must:

1. Obtain independent final game/schedule proof for 2027 and separately archive real NFLverse and ESPN source response bytes, versions, timing, exact game ID crosswalks and SHA256 hashes; verify all four-down opportunities and exclusions.
2. Pin original policy/nuisance hashes and record postgame-reconstructed forecasts BEFORE joining observed coach choices and independent final outcomes; be explicit that this is an **out-of-time retrospective** study, not independently witnessed live prediction.
3. Put a reviewed **real** Git ref claim in place BEFORE any final outcome evaluation, with source, forecast, independent-score hashes and a human-authorized permanent run record. Do not simply set a boolean on the fabricated code.
4. Protect the real tag namespace from deletion/force updates with GitHub repository rulesets and restrict the release to reviewer-approved GitHub Actions permissions. A remote Git tag is durable only as long as privileged users cannot erase it; current tests do not configure protections.
5. Preserve the protected 2026 Step 66/83 evaluations, the full denominator, locked calibration/DR thresholds and production app. No causal FG/GO claims or automatic model promotions.

The [2027 post-season synthetic gate](2027_postseason_existing_data_gate.md) remains the separate future-evaluation protocol.
