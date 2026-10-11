# 2025 kickoff penalty and return-TD modeling — Steps 76–77

**Research date:** October 11, 2026. **Status:** experimental Streamlit preview updated, official production decision model unchanged.

[Step 76 exception reconstruction](../src/76_freeze_penalty_return_continuation.py) · [Step 77 Rams scenario comparison](../src/77_compare_penalty_resolved_wp.py) · [Step 77 automated run](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38176422593) · [NFL 2026 official rulebook](https://operations.nfl.com/rules-officiating/2026-nfl-rulebook).

The underlying 2025 REG Q4 standard-own-35 kickoff population consists of 584 events with more than 40 seconds remaining. 2025 has already been inspected repeatedly; none of this evidence is an untouched independent holdout or a calibrated 2026 forecast.

## Reconstruct actual first possessions rather than inventing penalties

The frozen 2025 kickoff coverage census previously classified 52 events as penalty/re-kick. Step 76 identified **50 actual, score-neutral, first receiving-team scrimmage snaps** with no intervening second kickoff. The other **two were re-kicks**, requiring separate free-kick logic. The 50 verified penalty-affected first snaps span **43 games**, with mean receiving-team starting field position **own 26.18 yards** and mean observed kickoff-to-first-snap elapsed time **4.34 seconds**.

All 50 now join the original 488 ordinary 2025 donors in a new, validated **538-play experimental default**. For each modeled Q4 kickoff, the receiving-team first-snap field position and elapsed seconds come from the **same historical play**. The application samples no fabricated "average penalty yards", and does not label the two re-kicks as normal returns. Verified penalty data are not extrapolated to a special kickoff from midfield.

| Historical 2025 Q4 sample | Kickoffs modeled | Fraction of 584 eligible Q4 | Fraction of 218 in last five minutes (>40s) |
|---|---:|---:|---:|
| Original ordinary-only | 488 | 83.6% | 171/218 = 78.4% |
| New resolved nonscoring ordinary + penalties | **538** | **92.1%** | **185/218 = 84.9%** |

**Still outside the nonscoring pool:** 41 declared onside kick attempts (available only in the user's separate opt-in what-if), two actual re-kicks, one return touchdown, and two unmatched kickoffs. These 46 cases must not be silently mapped to ordinary returns. The matched onside donors include only **four** kicker-recovered first snaps.

Frozen evidence: [exception coverage JSON](../data/kickoff_2025_penalty_return_coverage.json), [penalty first-snap CSV](../data/kickoff_2025_penalty_first_snaps.csv), [ordinary 2025 donors](../data/kickoff_2025_reg_q4_ordinary.csv), and [onside first-snap observations](../data/kickoff_2025_onside_first_snaps.csv).

## Rams decision: same frozen model, only kickoff transition sample changed

Away offense Rams trailing 27–31, Q4 4:59, fourth-and-4 on the Seattle six. 2,400 simulations per action, seed 42. The frozen action/continuation model has not been trained or altered.

| Experimental kickoff assumption | FG action WP | GO/pass WP | GO minus FG | Monte Carlo near tie |
|---|---:|---:|---:|---|
| 488 ordinary donors only | 34.11% | 34.49% | +0.38 pp | Yes |
| **538 ordinary + verified penalty donors** | **34.08%** | **34.47%** | **+0.39 pp** | **Yes** |
| 538 donors plus *assumed* 25% onside attempt and 10% conditional kicking-team recovery chance after made FG | 32.64% | 34.47% | +1.83 pp | No (MC-only) |

**Interpretation:** including penalty-adjusted first possessions makes almost no difference in the Rams case. The original versus modern-kickoff model discrepancy is still chiefly due to the kickoff-pending feature representation and post-score WP model, not omitted clean penalty continuations. The 0.39 pp point gap is too close to call even at the Monte Carlo uncertainty level. A Monte Carlo-separated what-if comparison is still **not** proof of the right real-world play.

## Return touchdown: explicit risk limit, not invented scoring branch

In these 584 2025 Q4 eligible kicks, **one** kickoff was returned for a touchdown: [2025 Week 2, New England versus Miami](https://www.nfl.com/scores/). It occurred at **7:18 remaining in Q4**, outside the final-five-minute sample. A return touchdown changes the score and requires a Try and a new kickoff; it is not a nonscoring first snap and is **not** included in the 538-play resolver.

One occurrence is far too little data to infer a game-specific 2026 return-TD frequency or build an evidence-backed conditional score/clock simulation. Step 77 therefore supplies a **probability bound** rather than forcing it into the sampler: if the probability of a newly added, unmodeled branch is hypothetically at most \`p\`, then the absolute win-probability change for **one action** cannot exceed \`p\` in probability units (at most 0.5 percentage points for an assumed 0.5% event probability). This is a worst-case identity; the hypothetical probability is **not an empirical rate or forecast**, and this does not bound the difference of two actions jointly.

## Deployment decision and future work

The **experimental Streamlit comparison** now defaults to 538 verified nonscoring 2025 kickoffs; it preserves the previous ordinary-only 488-play sample as an Advanced reproducibility option and keeps the user's optional onside sensitivity. It runs in its own engine namespace and cannot mutate the production recommendation. No user needs to download NFL data or run any scripts.

**Do not promote** the experimental 538-sample policy ranking to production without independent, prospective outcome validation. Future work includes robust score-changing return-TD and penalty re-kick state machines, better 2026 conditional coaching decisions, and especially the remaining original-model kickoff-state encoding/calibration problem. The pre-registered 2026 Week 6+ prospective check remains unchanged and runs automatically at its 100-eligible-game threshold.
