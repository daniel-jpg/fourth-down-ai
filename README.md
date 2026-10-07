# Fourth Down AI

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
- **2025:** held-out final benchmark and reporting only

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

The v1.1.0 release has **26 automated tests** covering:

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
