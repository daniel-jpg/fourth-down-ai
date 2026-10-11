# 2025 onside recoveries and optional Rams what-if (Step 74–75)

**Research date:** 2026-10-11. **Deployment decision:** opt-in experimental preview only; leave the official fourth-down recommendation unchanged.

## Provenance and data scope

2025 NFL regular-season Q4 standard-own-35 kickoffs with >40 seconds left, the same 584-play universe as [the frozen coverage census](../data/kickoff_2025_q4_coverage.json). The 2025 season has already been examined in previous project research and is not an untouched validation set.

[Step 74 code](../src/74_freeze_2025_onside_outcomes.py) verifies onside recoveries using nflverse's **own_kickoff_recovery** indicator and checks the next possession using the team at the first scrimmage snap. Penalty-affected plays cannot be silently turned into ordinary transitions.

| Historical outcome | Number |
|---|---:|
| Eligible 2025 Q4 declared onside attempts | 41 in 39 games |
| Official kicking-team recoveries | 4 (9.8% observed among 41) |
| Verified onside-to-first-snap transitions | 37 |
| Verified kicker-recovered first snaps | 4 |
| Verified receiver-recovered first snaps | 33 |
| Onside attempts affected by penalties | 4 |

In verified 2025 first-snap outcomes, the kicking team began at its **own 46.0-yard line** on average after recovering and the receiver began at its **own 54.4-yard line** on average after a failed attempt. Average kickoff-to-first-snap clock differences were **2.25s** for successes and **1.42s** for failures. These are descriptive, selected observations with very small support, NOT stable game-specific forecasts.

Frozen data: [recovery audit JSON](../data/kickoff_2025_onside_outcomes.json) and [verified first-snap transitions CSV](../data/kickoff_2025_onside_first_snaps.csv).

## Hypothetical mixed kickoff simulation

[Step 75 script](../src/75_rams_onside_sensitivity_audit.py) uses the existing 488 ordinary 2025 donors and 37 verified onside first-snap transitions in a **private engine instance**.

- Only a fourth-quarter **made FG where the original kicking team remains behind** is eligible for this particular prototype. It does not claim that other onside strategies are illegal: [the 2026 NFL rulebook](https://operations.nfl.com/rules-officiating/2026-nfl-rulebook) allows a team to declare an onside kick at any time.
- Users **choose** both the probability of declaring an onside attempt and the kicking team's conditional recovery probability. Neither is fitted to 2025 or claimed as a 2026/team-specific forecast.
- An unsuccessful onside kick transfers to the receiving team's **observed post-onside first-snap state**, NOT an ordinary kickoff return; a successful kick **flips possession back to the kicking team** with an observed first-snap spot and elapsed clock.
- Scores, timeouts, home/away semantics and the fourth-down model are preserved.
- Zero assumed onside attempts produces exactly the same empirical 2025 preview as before. Simulations use separate reproducible random streams for ordinary and special kickoffs.

## Rams-style illustration

Hypothetical Rams away, Q4 4:59, 4th-and-4 at Seattle's six, Rams 27–31, three timeouts each. 2,400 simulations per action. **These are not causal comparisons of an unchosen action.** They are model predictions under explicit alternative special-teams assumptions.

| User-assumed onside attempt rate | User-assumed recovery chance GIVEN attempt | FG expected WP | GO/pass expected WP | GO–FG difference |
|---:|---:|---:|---:|---:|
| 0% | 10% (irrelevant with 0 attempts) | 34.11% | 34.49% | +0.38 pp |
| 10% | 10% | 33.41% | 34.49% | +1.07 pp |
| 25% | 5% | 32.15% | 34.49% | +2.34 pp |
| 25% | 10% | 32.60% | 34.49% | +1.88 pp |
| 25% | 20% | 33.59% | 34.49% | +0.89 pp |
| 50% | 10% | 31.33% | 34.49% | +3.16 pp |
| 100% | 0% | 25.01% | 34.49% | +9.47 pp |
| 100% | 100% | 61.71% | 34.49% | −27.22 pp |

The extremes are deliberately artificial checks of transition behavior. They are not realistic empirical distributions. Scenarios with nonzero assumed onside frequencies **cannot** be interpreted as proof that going for it has a higher real-world win probability. Differences may reflect poor or mismatched downstream WP calibration, tiny onside success support, and other missing kickoff outcomes. Monte Carlo standard error does not cover these sources of uncertainty.

The example is a hypothetical playoff-like matchup evaluated with **2025 regular-season** donors and unchanged frozen WP models. No 2026 game outcome data were read and no independent causal decision-level validation was performed.

## Production decision and what remains

**DO NOT PROMOTE** hypothetical assumptions into the official engine. The optional Streamlit **Advanced onside what-if** is isolated and off by default; all official recommendations and original scoring models remain unchanged.

Next improvements require more trustworthy conditional coaching choice data, penalty/re-kick and return-TD branches, calibration against later unseen games, and explicit sensitivity bounds for small n=4 recoveries. Continue preserving the automatic preregistered 2026 forward-evaluation gate, and never tune against it before its scheduled one-time run.
