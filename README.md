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

## At a glance

Fourth Down AI asks a practical question: **given the score, clock, field position and timeouts right now, which available fourth-down action has the highest *modeled* chance of winning?** It estimates each eligible action's immediate outcome, simulates the resulting game states, and compares their downstream win probabilities (or overtime game values). The model is decision-support research, **not a causal proof** that a coach should follow a particular play.

**Try a representative scenario:** Q4, 7:38 remaining, trailing 24–27, fourth-and-3 at the opponent's 30-yard line, with three timeouts for each team. In a verified live-app run, the frozen official model displayed **GO FOR IT: 41.44%** (run), **PASS: 40.86%**, **FIELD GOAL: 34.08%**, and **PUNT: 22.05%**. That is a **+7.36 percentage-point *model-estimated* edge** for GO versus FG; run versus pass was too close to distinguish confidently. These are an illustrative snapshot for those exact settings, not observed game outcomes or guaranteed stable values under different inputs.

**What makes the project distinctive:** action-specific GO/FG/punt outcome models, realistic sampled next-state transitions, adaptive Monte Carlo comparison, rule-aware overtime handling, and separate, reproducible 2025 kickoff research in the repository (not the public app). Open **Why this recommendation? Immediate action odds** in the app to see conversion and kick-outcome assumptions separately from overall game value.

**Evidence at a glance:** The frozen v1.1.0 2025 benchmark includes WP AUC **0.81781**, GO conversion Brier **0.22817**, and FG Brier **0.10931** (see [full benchmark](#held-out-2025-snapshot)). [GitHub Actions CI](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38182568146) passed **339 tests** (293 regression + 46 Streamlit integration); test success is not a substitute for external calibration or causal policy evaluation. Later 2025 exploratory analyses are no longer untouched holdouts; a precommitted [2026 prospective check](reports/2026_native_kickoff_challenger_protocol.md) is still gated.

**Explore:** [Live demo](https://fourth-down-ai.streamlit.app/) · [How it works](#how-it-works) · [Evaluation](#data-and-evaluation-design) · [Limitations](#limitations) · [Research-demo readiness](reports/88_research_demo_release_readiness.md)

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

## 2025 Kickoff Research (repository only)

The live Streamlit app deliberately has **no 2025 kickoff comparison
panel, experimental strategy toggle, or separate historical-scenario
results**. It focuses on the frozen official GO (pass/run), FIELD GOAL
and PUNT choices. The automatic Q4 post-made-FG model-reliability warning
remains available when warranted, because it flags uncertainty in the
official model rather than promoting another strategy.

The independent historical 2025 research remains reproducible in the
repository. Its selected 2025 regular-season Q4 nonscoring kickoff
pool contains 538 verified first possessions (488 ordinary and 50 clean
penalty-affected) out of 584 eligible standard-35 events (92.1%).
The other 46 include 41 onside attempts, two actual re-kicks, one
kickoff return touchdown and two unmatched plays; they are not
silently treated as ordinary kickoffs. See the
[coverage census](data/kickoff_2025_q4_coverage.json),
[ordinary kickoff pool](data/kickoff_2025_reg_q4_ordinary.csv),
[verified penalty-affected first possessions](data/kickoff_2025_penalty_first_snaps.csv)
and [onside outcomes](data/kickoff_2025_onside_first_snaps.csv).

Research-only native kickoff-input, historical-2024, ordinary-only,
onside what-if and calibration studies remain in the source and reports
**without public UI controls**. Retrospective 2025 results are not a
calibrated 2026+ forecast, causal coaching advantage, or fresh untouched
holdout. The precommitted Step 66/83 future-data gates remain frozen;
none of this promotes a challenger model. Fake punts and fake field
goals are also excluded from public action rankings.

## Readable explanation of the fourth-down recommendation

The decision page now provides a collapsed **"Why this recommendation?
Immediate action odds"** section. It displays the existing engine's
fourth-down pass/run conversion probabilities, FG make/miss/block rates,
and punt regular/block/broken-play probabilities alongside each normal
action's expected game-winning probability and Monte Carlo standard
error. There are no new model estimates or simulation calls: the
explainer parses the frozen engine's returned action details, excludes
ineligible and experimental fake plays, and fails closed on malformed or
non-normalized probabilities.

**Important:** a successful conversion or made kick is an immediate
outcome, **not** the probability of winning the game. Win probabilities
also depend on remaining time, score, field position, possession and
the frozen continuation model. Kicker-specific field-goal strength
defaults to the training-data median unless supplied in the engine.
See [the presentation-only explainer](src/fourth_down_action_explainer.py)
and [its tests](tests/test_fourth_down_action_explainer.py).

**Updated Q4 calculations (research demo):** The public Streamlit app now
uses the existing native standard-35 kickoff-input correction automatically
for regulation Q4, with the *same frozen model weights* and a fixed
1,200-simulation/action budget. Per-request Python function namespaces
share the cached model/transition artifacts read-only, avoiding a second
large engine load while keeping Streamlit sessions isolated. The app recomputes the **action win
probabilities and recommended play**; it does not merely replace a
disclaimer while leaving the older Q4 scores unchanged. No separate
kickoff checkbox or research panel is presented.

When the old-vs-native made-FG input check is materially sensitive, a
short **Close call** note accompanies the model recommendation. These are
still estimated probabilities, not prospectively validated 2026+ coaching
effects. The published stable v1.1.0 model and original precommitted
forward-evaluation protocols remain unchanged.

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

The native-input candidate remains **research-only**, without production
promotion or 2026 outcome access. Its historical comparisons are preserved
in the source code and audit reports; the advanced candidate toggle was
removed from the public app to keep its interface focused. The official
decision engine is unchanged.

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

### Streamlit integration coverage (Step 85)

The deployed `app.py` is exercised through `streamlit.testing.v1.AppTest`:

- Regulation inputs, halftime receiver selection, clock validation and action metrics
- Immediate outcome explanations and agreement with the engine's returned probabilities
- Unavailable actions, experimental fake exclusion and malformed explanation fallback
- Adaptive simulation budgets and Q4 reliability warnings, including unavailable checks
- Regular-season/playoff overtime controls, possession validation and game-value labels
- Removal of public kickoff-research controls without affecting official model output or the automatic Q4 reliability check
- Three smoke cases using the real frozen engine: regulation, Q4 and overtime

The UI cases block network connections and use frozen local artifacts. Most
tests replace only the engine response and optional simulation boundaries with
fixed fixtures; the three smoke cases use the actual production engine. They
do not assert exact Monte Carlo winners or access future validation outcomes.

```bash
python -m pytest -v tests/test_streamlit_app_integration.py
```

CI runs this as a separately named step and retains its JUnit report for
14 days. AppTest tests Python-side UI logic; it does not execute custom field
JavaScript or certify responsive layout. Separately, [Playwright live-browser check #9](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193289281)
**passed** against the hosted Streamlit app after the kickoff-panel removal:
the old panel and checkbox were absent, and **Analyze Fourth Down**
rendered the official recommendation. This functional check does not
independently validate predictive calibration or causal impact. See the
[Step 85 readiness review](reports/85_streamlit_ui_release_readiness.md).

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
- The v1.1.0 evaluation originally held out 2025. Later 2025 exploratory research means it is no longer untouched evidence for selecting a future challenger.

## Research-demo readiness

The public app is suitable for a **qualified technical research demo**,
not a validated real-world coaching policy or a new stable release.
[GitHub Actions Tests #233](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193413685)
on `85f0874` checks the complete regression suite, Streamlit AppTest,
and frozen artifact preflight after removal of the experimental panel.
[Live browser smoke #9](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193289281)
passed on app commit `053e4a5`: the live panel was gone and the
official Analyze Fourth Down result rendered. The only changes between
these two commits are source-level test corrections, not application logic.

[Release-QA Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5)
remains closed. The user accepted the mobile layout. The exact Streamlit
Cloud deployed SHA has not been cryptographically attested, and the
user-visible build-ID widget will **not** be reintroduced. The frozen
2026 Step 66/83 prospective research gates remain untouched.

See the [research readiness report](reports/88_research_demo_release_readiness.md)
and [release candidate checklist](reports/90_release_candidate_checklist.md).
A passing smoke test and CI do not establish causal coaching impact.

## Release

**Current stable release: v1.1.0**

See [CHANGELOG.md](CHANGELOG.md) for release notes.

v1.1.0 is treated as a frozen release. Future feature work should target a later release rather than changing the tagged version.
