# Changelog

Notable changes to Fourth Down AI are documented here.

## v1.1.0 — 2026-10-06

v1.1.0 is the hardened release of the fourth-down decision engine. It contains 25 commits beyond v1.0.2 and focuses on overtime rules, simulation stability, field-goal and punt behavior, validation, and automated regression coverage.

### Decision engine

- Added adaptive Monte Carlo evaluation: eligible actions are evaluated with 300 simulations initially, then rerun with 1,200 simulations when the top two actions are not clearly separated by the approximate Monte Carlo standard-error threshold.
- Preserved action-level uncertainty so close PASS/RUN decisions can be surfaced instead of overstated.
- Added validated short-yardage calibration for normal runs through 4th-and-3.
- Kept 2025 as a held-out benchmark rather than a tuning source.

### Overtime

- Added explicit regular-season and postseason overtime formats.
- Added opening-possession, response-possession, and sudden-death state handling.
- Added format-specific clock, timeout, terminal-state, and continuation logic.
- Updated the Streamlit UI with separate overtime-format and overtime-possession controls.

### Field goals

- Extended field-goal decision support through 70 yards.
- Added sparse-support messaging for 61–67 yard attempts.
- Added outside-observed-support messaging for 68–70 yard attempts.
- Attempts beyond 70 yards remain outside the model range.
- Development-period bootstrap analysis did not support adding a production FG calibration adjustment.

### Punts

- Reworked ordinary punt field-position simulation and validation.
- Preserved touchbacks as a discrete receiving-team own-20 state rather than smearing them through the continuous residual distribution.
- Final 2023–2024 validation matched the overall inside-the-20 rate closely: 40.15% observed vs. 40.17% simulated.
- Added direct regression tests for touchback and non-touchback punt sampling.

### Validation and testing

- Added calibration diagnostics and simulation-stability audits.
- Added comparison against a 4,000-simulation reference budget for adaptive Monte Carlo behavior.
- Added policy-level smoke tests for representative punt, field-goal, GO, goal-line, and overtime situations.
- Added GitHub Actions CI on pushes and pull requests to main.
- v1.1.0 finishes with 26 automated tests passing.

### Held-out 2025 snapshot

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
| Historical-action agreement — fixed 1,000-play policy sample | 53.20% |
| Mean model-estimated edge over historical action | +1.412 pp |

The policy edge is model-estimated. It is not an observed causal improvement in win probability.

## Earlier releases

- v1.0.2
- v1.0.1
- v1.0.0
