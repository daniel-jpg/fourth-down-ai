# Fourth Down AI

[![Tests](https://github.com/daniel-jpg/fourth-down-ai/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/daniel-jpg/fourth-down-ai/actions/workflows/tests.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://fourth-down-ai.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An NFL fourth-down decision engine that compares **go for it, field goal, and punt** decisions using machine learning, empirical transition models, and Monte Carlo simulation.

The project supports regulation, 2025+ NFL regular-season overtime, and postseason overtime, with an interactive Streamlit interface.

## Live Demo

[Launch the NFL Fourth-Down Decision AI](https://fourth-down-ai.streamlit.app/)

## Demo

![NFL Fourth-Down Decision AI](assets/fourth-down-ai-demo.png)

## Features

- Compares go-for-it, field-goal, and punt decisions
- Separately evaluates pass/dropback and run attempts
- Estimates downstream win probability in regulation and game value in overtime
- Uses action-specific conversion, field-goal, punt, and win-probability models
- Uses empirical transition pools for realistic post-play states
- Supports adaptive Monte Carlo evaluation for close decisions
- Preserves punt touchbacks as a discrete receiving-team own-20 outcome
- Supports field-goal attempts through 70 yards with explicit sparse-support warnings
- Supports regular-season and postseason overtime
- Keeps fake punts and fake field goals experimental and out of production recommendations
- Includes automated regression and policy smoke tests
- Includes a Streamlit web app for interactive use

## How It Works

For each fourth-down situation, the engine:

1. **Normalizes the game state**
   - Quarter and clock
   - Score
   - Field position
   - Yards to go
   - Timeouts
   - Home / away / neutral site
   - Stadium / roof
   - Overtime format and possession phase when applicable

2. **Checks action eligibility and support**
   - Unsupported or out-of-range actions are marked unavailable
   - Long field goals receive explicit support warnings

3. **Models action outcomes**
   - GO decisions use action-specific conversion models
   - Field goals model makes, misses, blocks, and broken plays
   - Punts model field position, clock runoff, touchbacks, and rare transitions

4. **Simulates post-play game states**
   - Each eligible action produces a distribution of possible next states

5. **Evaluates continuation value**
   - Regulation uses win probability
   - Overtime uses game value with rule-aware terminal handling

6. **Compares actions with Monte Carlo simulation**
   - Production begins with 300 simulations per eligible action
   - If the top two actions are not clearly separated by the approximate Monte Carlo standard-error threshold, the app reruns from scratch with 1,200 simulations per action

The displayed Monte Carlo standard error measures simulation noise only. It does not represent total model uncertainty.

## Overtime Support

The engine supports both **regular-season** and **postseason** overtime.

The app lets the user choose:

- **Overtime format**
  - Regular season
  - Playoffs
- **OT possession**
  - Opening possession
  - Response possession
  - Sudden death

The UI applies format-specific clock and timeout rules, and the engine handles possession requirements, scoring transitions, terminal outcomes, clock expiration, and postseason period continuation.

Overtime decisions are evaluated using **game value**:

- Win = 1.0
- Tie at expiration = 0.5
- Loss = 0.0

The overtime rule layer has been stress-tested with direct rule checks, large state sweeps, timeout combinations, regulation regression states, and held-out overtime plays.

## Interactive App

Run the Streamlit interface locally:

```bash
streamlit run app.py
```

The app allows you to specify:

- Quarter and game clock
- Yards to go
- Offense and defense score
- Offense and defense timeouts
- Line of scrimmage
- Home / Away / Neutral site
- Stadium / roof
- Overtime format and possession phase

It displays:

- Recommended strategy
- Expected win probability or overtime game value
- Advantage over the next-best strategy
- PASS vs RUN guidance when GO FOR IT is recommended
- Monte Carlo uncertainty messaging for close decisions
- Action eligibility explanations
- Long-field-goal support warnings
- Detailed model output and comparison charts

## Experimental 2025+ Kickoff Comparison

For Q4 situations, the Streamlit app offers a clearly labeled, optional
**kickoff-adjusted scenario** alongside the unchanged production recommendation.

- The **current experimental default** samples whole kickoff-to-first-snap
  outcomes from **538 observed 2025 regular-season Q4 standard-35 kickoffs**:
  488 ordinary kicks plus 50 penalty-affected kicks with independently
  verified, score-neutral receiving-team first snaps. The fields (touchback,
  receiving position, clock consumed) are sampled together from actual plays.
- The frozen [2025 Q4 kickoff coverage census](data/kickoff_2025_q4_coverage.json)
  includes **584** eligible kickoffs, so the improved experimental resolver
  represents **538/584 (92.1%)** of Q4 kickoffs, versus **488/584 (83.6%)**
  in the original ordinary-only preview. In the final five minutes and with
  more than 40 seconds remaining, coverage rises from **171/218 (78.4%)**
  to **185/218 (84.9%)**. Neither fraction is a validated 2026 forecast.
- The remaining **46 unsupported kickoff outcomes** are 41 declared onside
  attempts, two actual re-kicks, one kickoff return TD and two unmatched
  outcomes. The [penalty/re-kick and TD audit](data/kickoff_2025_penalty_return_coverage.json)
  and [reconstructed historical first snaps](data/kickoff_2025_penalty_first_snaps.csv)
  are frozen and checked. We never replace a re-kick or scoring return with
  an ordinary return.
- The original **488-only** empirical comparison is preserved under
  Advanced for reproducibility. Its historical touchback share is **31.97%**;
  this descriptive rate is not an assumed future 2026 touchback probability.
- In the selected last-five-minute **receiver-leading-by-1-to-3** group
  relevant after a Rams-style made FG, **15 of 20** observed 2025 kicks
  were ordinary and five were onside attempts. **Twenty kicks is not
  enough to calibrate a future game-specific onside strategy.** The
  Streamlit experimental section explicitly shows these limitations.
- All underlying historical data are versioned: the
  [488 frozen ordinary kicks](data/kickoff_2025_reg_q4_ordinary.csv),
  the [penalty-affected kickoff observations](data/kickoff_2025_penalty_first_snaps.csv),
  and the [2025 onside transition samples](data/kickoff_2025_onside_first_snaps.csv).
  Visitors do **not** download live NFL play-by-play to use Streamlit.
- An Advanced option preserves the earlier 2024-return sensitivity study with
  an explicitly assumed touchback rate.
- A separate, **opt-in 2025 onside what-if** tests two *user-chosen*
  parameters: chance of choosing an onside kick after a made Q4 field goal
  while still trailing, and chance of the kicking team recovering if attempted.
  This does not estimate a real team's policy.
- The 2025 REG Q4 coverage census has **41** eligible declared onside kicks
  across 39 games. The official `own_kickoff_recovery` field identifies **four**
  kicking-team recoveries (4/41, 9.8% observed); **37** onside kicks were
  matched to a verified subsequent scrimmage snap, with **four** penalty
  cases excluded from the transition donor pool.
- Field position and elapsed clock in either what-if onside outcome are
  sampled together from these 37 actual possessions. A recovered kick
  correctly transfers possession **back to the kicking team**; a failed
  attempt leaves the receiver in possession. **Only four verified
  success donors** means substantial unmodeled uncertainty.
- The 2026 NFL rules allow declaring an onside kick regardless of score,
  but this particular prototype is intentionally scoped to a made Q4 FG
  leaving the original kicking team trailing (the Rams-style case). It does
  not attempt to simulate all legally permitted onside choices.
  See [the official 2026 NFL rulebook](https://operations.nfl.com/rules-officiating/2026-nfl-rulebook).
- The [Rams sensitivity script](src/75_rams_onside_sensitivity_audit.py)
  and [2025 onside recovery audit](src/74_freeze_2025_onside_outcomes.py)
  run automatically in GitHub Actions. No local data downloads are
  required for Streamlit visitors.
- The scenario runs in an isolated engine instance and never overwrites the
  official recommendation, models, or shared Streamlit cache.
- The expanded preview still cannot establish a causal FG/GO
  **counterfactual** or model return touchdowns, actual re-kicks and
  ambiguous kickoffs. A single 2025 Q4 return touchdown at 7:18 remaining
  cannot estimate a 2026 late-game risk; [Step 77](src/77_compare_penalty_resolved_wp.py)
  reports explicit, hypothetical worst-case probability bounds rather than
  inventing a scoring-return probability. An experimental winner separated
  by less than Monte Carlo noise is reported as **too close to call**.

The original 2025 evaluation was held out for the v1.1.0 release. Subsequent
exploratory studies have **examined 2025**; those results must not be described
as an untouched independent holdout for future model selection. 2026 Week 6+
is reserved for a future prospective benchmark.

## Q4 Kickoff Feature-Encoding Audit (Research)

A [14-situation Q4 robustness audit](reports/2025_q4_kickoff_site_model_defect.md)
identified a material issue in the **original** (still official) win-probability
engine: changing only HOME versus AWAY or NEUTRAL can reverse GO versus
FIELD GOAL near 4th-and-3 at opponent 30, down three with 7:38 remaining.
The historical kickoff-pending state uses missing `yardline_100` and
`ydstogo`, unlike the native standard-35 kickoff encoding.

A **private, reversible candidate**, using `yardline_100=35`,
`ydstogo=0`, `down=NaN` for unresolved normal Q4 scoring kickoffs,
eliminated both reversals on the fixed 14-state grid while preserving
the same frozen WP coefficients. The native-only candidate and the
538-transition comparison both favor GO in 14/14 cases; the original
engine favors GO in 12/14. This is a modeling defect investigation,
**not** a validated causal decision rule. Historical 2024/25 made-FG
calibration point estimates improved with a similar encoding, but
their game-cluster uncertainty intervals include no improvement.

The broader [Step 81 retrospective calibration](reports/2023_2025_broad_q4_native_kickoff_calibration.md)
then compared the SAME frozen WP model on **1,617 actual 2023–2025 Q4
post-score kickoffs** across **762 games**, retaining penalty outcomes and
return touchdowns. The native input encoding had lower pooled Brier loss
(**0.11561 to 0.10743**, gain +0.00818, paired whole-game 95% bootstrap
CI **[+0.00264,+0.01398]**). All three years had favorable point
estimates, but individual-year CIs included zero. Crucially, the
**away-kicking-team subgroup was slightly worse** (gain -0.00119).
These seasons were already examined and this does not validate the
counterfactual fourth-down play selection.

The candidate remains **research-only**, without a production promotion
or 2026 outcome access. In the Q4 Streamlit experimental panel, you can
now optionally enable **Advanced: also show native kickoff-input
correction (research-only)**, which compares its strategy/action WPs
against the untouched official model and the separate 538-kickoff
scenario. The candidate is off by default, runs in a private engine,
and never changes the official recommendation.

**Automatic Q4 reliability warning:** Because the original official WP
engine retains a demonstrated post-score kickoff input mismatch, the main
Streamlit decision screen now performs a **two-state deterministic**
sensitivity check whenever a normal FG is eligible in Q4 with **more
than 45 seconds before the kick** (so a hypothetical 5-second made FG
leaves more than 40 seconds at kickoff). It holds a hypothetical **made FG** fixed
with an assumed **5-second execution clock** and predicts the original
offense's WP with the legacy and native input encodings. If those
*conditional-on-a-made-FG* WPs differ by **at least five percentage
points**, Streamlit displays an automatic caution beneath the official
recommendation. This five-point cutoff is a **display heuristic, not a
confidence interval or a validated strategy-ranking criterion**. The
conditional made-FG WPs are **not** the full FG action WP and cannot by
themselves imply a GO-vs-FG decision flip. The guard never retrains a
model, uses 2026 outcomes, or replaces an official recommendation.
See [the isolated guard](src/kickoff_q4_reliability_guard.py) and its
[regression tests](tests/test_kickoff_q4_reliability_guard.py).

**One-time 2026 native-candidate validation is now precommitted.**
The original [Step 66 automatic 2026 Week 6–18 gate](.github/workflows/forward_2026_validation_once.yml)
remains completely unchanged: it evaluates only after at least 100 eligible
completed games and records its frozen research result once. A new
[post-gate companion](.github/workflows/forward_2026_native_challenger_once.yml)
will then **automatically** score the native input encoding directly against
the frozen baseline, including *home-kicker versus away-kicker* paired Brier
differences and game-cluster uncertainty. It will **not fetch 2026 outcomes
until the original report exists**, and it never promotes a model. The entire
scope, source hashes, site-specific support conditions and no-peeking rules
were committed ahead of time in the
[Step 83 validation protocol](reports/2026_native_kickoff_challenger_protocol.md).
No manual script runs are necessary.

See also
[`src/kickoff_q4_native_features_candidate.py`](src/kickoff_q4_native_features_candidate.py),
[Step 80 evaluation](src/80_evaluate_q4_native_kickoff_candidate.py),
[Step 81 analysis](src/81_broader_post_score_kickoff_calibration.py),
and the [initial site-defect report](reports/2025_q4_kickoff_site_model_defect.md).

## Installation

Clone the repository:

```bash
git clone git@github.com:daniel-jpg/fourth-down-ai.git
cd fourth-down-ai
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

Run the app:

```bash
streamlit run app.py
```

Run the test suite:

```bash
python -m pytest -q
```

## Project Structure

```text
fourth-down-ai/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── assets/
│   └── fourth-down-ai-demo.png
├── data/
├── models/
├── tests/
│   ├── test_engine_regressions.py
│   └── test_policy_smoke.py
└── src/
    ├── 01_build_fourth_down_dataset.py
    ├── ...
    ├── 39_build_punt_model.py
    ├── 40_build_fake_models.py
    ├── 41_train_win_probability_model.py
    ├── 42_build_decision_engine.py
    ├── 43_evaluate_2025.py
    ├── 46_audit_overtime_engine.py
    ├── 47_evaluate_validation_calibration.py
    ├── 48_evaluate_policy_simulation_stability.py
    ├── 49_evaluate_strategy_simulation_stability.py
    ├── 50_evaluate_adaptive_vs_4000.py
    ├── 51_evaluate_punt_distribution.py
    └── 52_evaluate_punt_own20_atom.py
```

The numbered scripts document the modeling, auditing, calibration, and validation pipeline used to build the production engine.

## Main Components

### Decision Engine

`src/42_build_decision_engine.py`

Loads the trained models and empirical transition pools, simulates each eligible action, and compares expected game value.

### Web App

`app.py`

Streamlit interface for entering a game situation and viewing the recommended fourth-down decision.

### Held-Out Evaluation

`src/43_evaluate_2025.py`

Reports final 2025 benchmark performance. The 2025 season is reserved for held-out evaluation rather than production tuning.

### Overtime Audit

`src/46_audit_overtime_engine.py`

Tests overtime rules, state validity, timeout behavior, terminal scoring logic, postseason continuation, and regulation regressions.

### Simulation Stability

`src/48_evaluate_policy_simulation_stability.py`

`src/49_evaluate_strategy_simulation_stability.py`

`src/50_evaluate_adaptive_vs_4000.py`

Evaluate Monte Carlo stability and the production adaptive simulation policy.

### Punt Validation

`src/51_evaluate_punt_distribution.py`

`src/52_evaluate_punt_own20_atom.py`

Validate the production punt field-position distribution and the discrete own-20 touchback mass.

## Data and Evaluation Design

The project uses a chronological development and evaluation setup.

- **2014-2022:** primary model-training period
- **2023-2024:** development validation and calibration analysis
- **2025:** held-out benchmark for frozen v1.1.0; later examined in labeled exploratory experiments

The production engine combines:

- Win-probability modeling
- Normal GO conversion modeling
- Field-goal outcome modeling
- Punt field-position and transition modeling
- Empirical transition pools
- Monte Carlo simulation

The 2025 season is intentionally kept out of production calibration and model-selection decisions.

## Held-Out 2025 Snapshot

| Metric | Result |
| --- | ---: |
| WP log loss — all states | 0.53652 |
| WP AUC — all states | 0.81781 |
| GO conversion log loss | 0.64830 |
| GO conversion Brier score | 0.22817 |
| GO conversion AUC | 0.65750 |
| Field-goal log loss | 0.35617 |
| Field-goal Brier score | 0.10931 |
| Field-goal AUC | 0.74792 |

On the fixed 1,000-play held-out policy sample:

- Exact agreement with the historical action: **53.20%**
- Mean model-estimated edge over the historical action: **+1.412 percentage points**
- Decisions with more than a 1-point model-estimated edge: **31.29%**
- Fake-play recommendations: **0**

The policy edge is model-estimated, not an observed causal improvement in win probability.

## Punt Validation

On 2023-2024 validation punts, the production simulator matched the overall inside-the-20 rate closely:

- Observed: **40.15%**
- Simulated: **40.17%**

The touchback-preservation update also restores a distinct receiving-team own-20 mass instead of smearing touchbacks through the continuous residual distribution.

## Testing

The frozen v1.1.0 release originally had **26 automated tests**. The current
`main` branch has additional regression coverage for kickoff and model
research, including empirical 2025 scenario isolation and future-data safeguards.

The v1.1.0 release tests cover:

- Engine regressions
- Overtime and terminal-state behavior
- Field-goal behavior
- Punt touchback preservation
- Policy-level smoke scenarios

GitHub Actions runs the suite automatically on pushes and pull requests to `main`.

## Tech Stack

- Python 3.12
- Streamlit
- NumPy
- pandas
- Polars
- scikit-learn
- XGBoost
- Altair
- nflreadpy
- PyArrow
- joblib
- pytest

## Limitations

- This is a predictive decision-support model, not a causal guarantee of the optimal football decision.
- Monte Carlo standard error reflects simulation noise, not total model uncertainty.
- The model cannot capture every relevant football factor, including exact personnel, injuries, play design, coaching tendencies, wind, and other unobserved context.
- Normal-run recommendations are intentionally restricted to the development-supported short-yardage range.
- Very long field goals are data-sparse. Attempts from 61-67 yards use a sparse-data tail; 68-70 yard attempts are explicitly marked outside observed development support.
- Attempts beyond 70 yards are outside the model range.
- Kicker-specific context defaults to the development-data median unless supplied programmatically.
- Fake punt and fake field-goal actions are experimental and are not eligible for production recommendations.
- The 2025 season is held out for evaluation and should not be used to retroactively tune v1.1.0.

## Release

**Current stable release: v1.1.0**

See [CHANGELOG.md](CHANGELOG.md) for release notes.

v1.1.0 is treated as a frozen release. Future feature work should target a later release rather than changing the tagged version.
