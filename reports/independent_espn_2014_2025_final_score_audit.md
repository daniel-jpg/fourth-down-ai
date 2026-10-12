# Independent historical NFL final-score verification — complete for 2014–2025

**Completed:** October 12, 2026 UTC (October 11 Pacific)
**Provider:** ESPN Site API v2, independently retrieved historical REG scoreboards, not the nflverse play-by-play source used to train the models.
**GitHub Actions verified run:** [2014–2025 full score audit](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38207988956)
**Scope restriction:** ONLY the completed historical 2014–2025 NFL regular seasons. No 2026 reserved outcome data, no 2027 outcomes, no action-policy scoring, no live model promotion.

## Result: all available historical finals agree

| Independent cross-source check | Result |
| --- | ---: |
| Completed regular-season years | 12 (2014–2025) |
| ESPN weekly feeds downloaded | 209 |
| Completed ESPN regular-season games | 3,151 |
| Distinct PBP source game results compared | 3,151 |
| Matched final home-score margins | **3,151 / 3,151** |
| Final-score mismatches | **0** |
| Unmatched games | **0** |
| Missing source final margins | **0** |
| Historical REG fourth-down rows in matched games | **47,942** |
| Dedicated parser/mismatch tests | 25 passed in the source audit |
| Frozen PBP Git blob SHA-1 | d7230df63cad1c24579ee40cabaacd0b330e4784 |
| PBP file SHA-256 | 07fb573961131e02340b1403e9b03edec306523d528136e438f269447dbba903 |
| Full ESPN final-score record snapshot canonical SHA-256 | bf0523ae05abb872fa8ac44e1cb44e6a91d171039ee702b265382b343c3c8ceb |

**Denominator clarification:** 47,942 counts all sourced regular-season fourth-down rows with a verified completed game's final margin. The earlier historical nuisance cohort contains **44,507** usable normal fourth-down action decisions after excluding fakes, no-plays, OT and other unsupported states. These are different denominators; the score check has not newly verified those action classifications.

### Archived provider exceptions — explicitly NOT treated as final scores

| Season/week | ESPN event | Disposition |
| --- | --- | --- |
| 2014 W12, NYJ at BUF | 400554331 | Postponed placeholder; actual completed game matched separately |
| 2017 W1, TB at MIA | 400951581 | Postponed placeholder; completed game later matched |
| 2022 W17, BUF at CIN | 401437947 | Game canceled without a final score; **never used as a terminal game result** |

The independent parser rejects partial, live, scheduled, ambiguous, duplicate or otherwise malformed game records. Only historical STATUS_POSTPONED placeholders are skipped, and the canceled BUF-CIN game has a specifically reviewed exception. A separate final-game count gate demands 256 games/year before the 17th-game era, 272 games/year from 2021 except the 271-game 2022 season. Every game in the historical PBP source must additionally find an **exact** ESPN final-score match on season/week and normalized home/away franchise identity.

## Reproducible, long-lived archive

The original [full data-audit workflow](../.github/workflows/independent_espn_historical_score_audit.yml) stores its source-result report and a separate, complete ESPN final-score sidecar with per-week raw response hashes as GitHub Actions artifacts. That ZIP has temporary retention: the October 12 source run's artifact expires **November 11, 2026 UTC**.

To avoid losing the central independent label evidence, we also committed the **3151-game score-margin vector** to [compact source archive](../data/independent_espn_2014_2025_margin_archive.json). The archive contains:

- The separate ESPN home-minus-away final score for every completed historical game, compressed as signed 16-bit integers and encoded as base64, sorted by (season, week, normalized home, away).
- SHA-256 of the full ordered game matchup key list and separately SHA-256 of raw uncompressed margin bytes; these hashes prevent a changed source, game order or outcome from being silently reinterpreted.
- The frozen PBP source hashes, original run ID, 2014–2025 scope, nonfinal provider event IDs and the canonical checksum of the complete full-score event sidecar.

The [offline archive verifier](../src/independent_espn_margin_archive.py) rebuilds the 3,151-game PBP census from the pinned parquet file and compares every final score to that independently archived vector **with no ESPN call whatsoever**. The [tampering tests](../tests/test_independent_espn_margin_archive.py) reject a changed game identity, missing game, changed final winner or manipulated vector. New live historical audits also require **exact match** to the versioned score archive. No model fitting or decision-time feature computation depends on the ESPN data.

The full 209 raw weekly responses and rich ESPN event sidecar are held separately in the temporary audit ZIP rather than embedded in the compact permanent archive; only the final-score vector, key hashes and provenance digests are preserved in permanent version control. The permanent archive protects labels from silent drift but is not a third-party signed attestation.

## Independent ESPN versus nflverse game-site check

The separate [3,151-game venue comparison](historical_relocated_neutral_site_provider_disagreements.md) found **49 ESPN neutral-site flags versus 53 nflverse physical-neutral flags**. Exactly four are different: three San Francisco 49ers designated-home games actually played at **State Farm Stadium in Arizona** during 2020, and the 2021 Saints designated-home opener actually played at **TIAA Bank Field in Jacksonville** due to Hurricane Ida. Official 49ers, Saints and NFL stadium sources independently document the relocations. The frozen source is not automatically relabeled; the raw provider mismatch remains **4** in the audit, with **0 unexplained differences** based on the reviewed stadium evidence. A new/changed provider mismatch fails the future archive audit. Physical relocation does not establish a zero home-advantage coefficient or certify game-snap roof and kickoff-receiver inputs.

## What this does and does not close

**Closed:** The earlier issue that the nflverse-derived 2014–2025 final winners were not checked against an independent outcome provider. Every completed REG game included in the project now has a matching independently sourced final home-score margin.

**Still open (before calling the 2027 policy study frozen):** Independent proof that stadium roof state, possession/site orientation, second-half receiving direction, distance, clock, historical play formation/coach intent, kicker prior and other input facts are known *before* the relevant decision; trusted forecast timestamps and immutable prediction seals; signed approval and external reviewer of the model/source manifest; and real prospective 2027 calibration/overlap and causal limitations. ESPN post-game final scores cannot solve these.

The action-policy recommendation engine and all production model weights were **untouched**, and the preregistered 2026 kickoff Steps 66/83 were never invoked. **No causal GO-versus-FG benefit is established** by 3,151 matching final score labels.
