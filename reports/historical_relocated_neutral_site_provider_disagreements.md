# Four ESPN-vs-nflverse neutral-site disagreements are relocated venues

**Scope:** 2014–2025 historical REG games. This is an independent *retrospective stadium verification*, not proof of calibrated home-field advantage or as-of-snap model feature timestamps.

An [offline 3,151-game provider flag comparison](../src/independent_espn_neutral_site_audit.py) found:

| Historical category | Games |
| --- | ---: |
| ESPN neutralSite=true | 49 |
| nflverse fourth-down PBP location=Neutral | 53 |
| Provider site-flag disagreements | **4** |
| PBP site values missing/unrecognized | **0** |
| Differences explained by documented physical relocations | **4** |
| Unexpected/unreviewed differences | **0** |

Neither provider necessarily has a "bad stadium" record: **ESPN neutralSite=false appears to preserve the designated home-club convention** for these games, even when the game physically occurred outside the normal home stadium. The historical PBP "Neutral" label captures the physical relocation. These fields are not perfect synonyms.

## Official venue evidence for each disagreement

| PBP game ID | Nominal home | Actual game stadium | Cause and contemporaneous/official evidence |
| --- | --- | --- | --- |
| 2020_13_BUF_SF | San Francisco | State Farm Stadium, Glendale, AZ | The [49ers' Nov 30 relocation announcement](https://www.49ers.com/news/statement-from-the-san-francisco-49ers-arizona-week-13-week-14) confirms Bills Week 13 moved to Arizona due to Santa Clara COVID restrictions |
| 2020_14_WAS_SF | San Francisco | State Farm Stadium, Glendale, AZ | The same [49ers official announcement](https://www.49ers.com/news/statement-from-the-san-francisco-49ers-arizona-week-13-week-14) specifically confirms Washington Week 14 moved |
| 2020_17_SEA_SF | San Francisco | State Farm Stadium, Glendale, AZ | [NFL official game center](https://www.nfl.com/games/seahawks-at-49ers-2020-reg-17) records Glendale AZ, State Farm Stadium. Also see [49ers official 2020 full schedule](https://www.49ers.com/schedule/2020/) |
| 2021_01_GB_NO | New Orleans | TIAA Bank Field, Jacksonville, FL | [Saints Sept 1 announcement](https://www.neworleanssaints.com/news/september-12-new-orleans-saints-green-bay-packers-jacksonville-tiaa-bank-field) confirms the relocation from New Orleans due to Hurricane Ida |

**Comparison result:** For all four, ESPN's "neutralSite" is false while nflverse's "location" is "Neutral". This is **not** evidence that the teams played at their normal home stadiums, and must not cause us to overwrite PBP's location.

## Reproducibility and strict anomaly controls

The physical venue findings, game IDs, stadiums, documented official sources and reason for relocation are stored in [source-linked frozen review data](../data/historical_relocated_venue_evidence_2014_2025.json).

The [offline audit](../src/independent_espn_neutral_site_audit.py) compares every archived ESPN venue flag against all 3,151 PBP games, records the four disagreements **without changing them**, and checks the four against the exact reviewed IDs, season/week, home/away and official evidence. It must refuse to certify if any reviewed anomaly disappears unexpectedly, another case appears, site values are missing, or opposite-direction flags occur. The CI audit raises an error for such unexpected divergence and [tests](../tests/test_independent_espn_neutral_site_audit.py) check source integrity and drift.

The official-stadium verification **does not prove** that the home-field advantage in those relocated games was zero, how neutral and nominal home supporters behaved, what roof/weather/kicker inputs were available at individual decisions, or any future 2027 timestamp. These important predictive and causal matters remain open and outside these historical scoreboard/site data. We do **not** edit the action-policy state builder, historical nuisance feature coefficients, public Streamlit model, or protected 2026 kickoff evaluations.

## Remaining data provenance

The [independent final-score audit](independent_espn_2014_2025_final_score_audit.md) matched all 3,151 historical final game margins with no mismatches. The four differences concern game-site **label semantics only**, not the score or winner. Going forward, a prospective decision-time record should keep **both** designated home team and exact stadium/geolocation/roof state (and timestamp), rather than infer location solely from ESPN neutralSite. Source verification should be made independently of the final-game record; the evidence of a public announcement before game time does not itself make our program's stored event snapshot tamper-proof.
