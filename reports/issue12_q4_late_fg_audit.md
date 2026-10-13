# Issue #12: Q4 final-two-minute made-FG support and expired-clock audit

**Research date:** October 12, 2026 Pacific (October 13 UTC).
**Status:** Read-only numerical and historical audits passed. Display-only support guard added; **frozen model weights and protected prospective evaluators NOT changed**. Not validated coaching advice.

## Confirmed reproducible problem

At Q4 0:45, fourth-and-1 at opponent 1, down 4, AWAY, two timeouts each:
- Frozen demo native-standard-35: **GO 57.20%, FIELD GOAL 23.72%**, 1,200 simulations/action, seed 43.
- Different seed 101: GO 58.24%, FG 23.68%, so the high FG estimate is not a lucky Monte Carlo fluctuation.
- The frozen WP model assigns **23.91%** to the original team's win **conditional on a made FG** leaving it **down by 1 with 40 seconds left**. This is a model forecast, not evidence of a recover-and-score success rate.
- For a made FG evaluated with regulation time fully exhausted, the **unmodified frozen model predicts 17.10% rather than the rule-determined 0%** in the AWAY -4 example. All **48 of 48** synthetic clock/margin/site/seed combinations disagreed with an unambiguous terminal 0%/100% game result when using the raw frozen WP model at 0:00.
- The Q4 official demo model represents standard pending kickoffs rather than explicitly sampling future onside decisions and possession changes. It cannot be assumed to calibrate improbable end-game comeback conditions.

Sensitivity with AWAY, original trailing 4, seed 43:

| Q4 remaining before decision | GO modeled WP | FG modeled WP | Native-encoded WP after made FG, 5s kick |
| --- | ---: | ---: | ---: |
| 0:20 | 59.31% | 15.87% | 16.00% |
| 0:45 | 57.20% | 23.72% | 23.91% |
| 1:00 | 55.21% | 23.82% | 24.02% |
| 2:00 | 51.12% | 33.34% | 33.55% |

See [the isolated-engine diagnostic](../scripts/audit_issue12_q4_late_fg.py) and [passing run](https://github.com/daniel-jpg/fourth-down-ai-nfl/actions/runs/38335603388) (36 pure unit tests and 48 model scenarios). Its artifact contains each simulated FG outcome contribution and Monte Carlo SE. **Monte Carlo SE measures simulation variance, not specification error or validation accuracy.**

## Completed historical support census

The separate [2014–2025 read-only historical selector](../scripts/audit_issue12_q4_short_fg_history.py) examines all first standard-own-35 REG Q4 kickoffs within 120s following a verified made FG; declared onsides are **retained**, identified only by description text. Cohorts are selected from pre-kickoff play data; no final outcomes or future-season scores are used.

| Historical era | All qualifying after-FG Q4 kickoffs, 0–40s | All qualifying after-FG Q4 kickoffs, 41–120s | Receiving team leads exactly 1 after FG, 0–120s |
| --- | ---: | ---: | ---: |
| 2014–2022 training era | 144 | 222 | **0** |
| 2023–2025 previously examined development era | 54 | 85 | **0** |

That is **zero** comparable receiving-team-leading-one post-FG kickoffs among the selected 12 seasons; this does *not* establish that a comeback has literal zero probability. The cohort is broadly selected by clock and score, not identical fourth-down state, coaching decision, weather, possession history or all special-kick mechanisms. No causal GO-vs-FG estimate can be extracted from these counts.

See [historical-count workflow](https://github.com/daniel-jpg/fourth-down-ai-nfl/actions/runs/38335692672) (2 synthetic selector tests plus completed-year downloads and counts).

## Scope of mitigation

An **isolated demo-only exact terminal-game rule** is applied when a Q4 simulated post-play state reaches 0:00 and the final score unequivocally determines the winner. It is layered around, not inside, the registered native-35 representation correction. It never overwrites the published frozen engine or modifies the 2026/2027 prospectively registered source or training weights. Tied 0:00 states retain the separate overtime-model treatment, and touchdown Try evaluation is preserved.

A **presentation-only reliability warning** now appears when the demo is in Q4 with 0:01–2:00 left, trailing by four, and FG is an available alternative. It explicitly states the lack of matching late-FG support and warns that raw WP estimates at 0:00 are nonterminal. If the model chooses FG here, its headline becomes **INSUFFICIENT DATA** rather than falsely presenting FG as supported. Otherwise the GO headline, action values, rankings and probabilities are unchanged.

Do **not** silently set displayed FG WP to 0%, override GO with PUNT or insert arbitrary onside assumptions. The deterministic demo-only terminal rule can alter some late-Q4 action values and must be validated before any claim of empirical action-value accuracy. The unmodified frozen prospective evaluators must remain entirely separate. **The registered native kickoff representation module, Step 66/83 gates, stable model files, and 2027 prospective action-policy study remain unchanged.**

This mitigation protects interpretation rather than claiming numerical calibration. Future independently validated late-game continuation rules and better-specified onside data are still needed before calling these probabilities reliable.
