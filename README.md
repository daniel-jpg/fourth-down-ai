# Fourth Down AI

An NFL fourth-down decision engine that compares **go for it, field goal, and punt** decisions using machine learning, empirical transition models, and Monte Carlo simulation.

The project supports both regulation and current NFL regular-season overtime situations and includes an interactive Streamlit interface.

## Features

- Compares go-for-it, field-goal, and punt decisions
- Separately evaluates run and pass attempts
- Estimates downstream win probability / game value
- Models field-goal outcomes and post-kick transitions
- Models punt field position and rare punt outcomes
- Uses empirical fourth-down transition pools
- Supports regular-season overtime:
  - Opening possession
  - Response possession
  - Sudden death
  - 10-minute overtime period
  - 2 timeouts per team
  - Tie value at expiration
- Interactive Streamlit app

## How It Works

The decision engine combines several components:

1. **Fourth-down conversion modeling**
   - Estimates conversion probability
   - Separately evaluates run and pass attempts

2. **Field-goal modeling**
   - Estimates made, missed, blocked, and other outcomes
   - Simulates resulting possession, field position, and clock state

3. **Punt modeling**
   - Estimates resulting field position
   - Includes ordinary and rare punt outcomes

4. **Win-probability modeling**
   - Estimates the value of resulting game states

5. **Monte Carlo simulation**
   - Simulates possible outcomes for each available action
   - Recommends the highest-value decision

## Overtime Support

The engine supports current NFL regular-season overtime decision states.

It distinguishes among:

- **Opening possession**
- **Response possession**
- **Sudden death**

Overtime logic handles possession requirements, terminal scoring events, clock expiration, and tied-game value.

The OT implementation was tested with:

- 12 direct rule-layer checks
- 1,330 valid overtime states
- 27 timeout combinations
- 144 regulation regression states
- Held-out 2025 regular-season overtime plays

Final audit results:

```text
HARD FAILURES: 0
REVIEW FLAGS: 0
Regulation mismatches: 0