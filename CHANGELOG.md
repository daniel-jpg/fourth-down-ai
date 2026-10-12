# Changelog

Notable changes to Fourth Down AI are documented here.

## Unreleased — Research-demo candidate (post-v1.1.0)

**Status:** A qualified, tested research demo on `main`, NOT a new stable
`v1.2.0` or production-calibrated coaching policy. Stable `v1.1.0`
and the frozen official model remain unchanged.

### Interface and interpretation

- Streamlit focuses on the **official** GO (pass/run), FIELD GOAL and
  PUNT recommendations. Removed the public 2025 kickoff comparison
  panel, toggle, and separate experimental-strategy results following
  user feedback.
- Retained the 538-transition 2025 kickoff study and other optional
  historical/native-input evaluations in the research code and reports,
  not as interactive public app features.
- Added immediate modeled conversion, field-goal and punt outcome
  explanations alongside win probability or overtime game value.
- Preserved the **automatic Q4 post-FG model-reliability warning**, and
  now mark a material model-sensitive FIELD GOAL leader as
  **INCONCLUSIVE**, keeping the frozen WP numbers visible. The warning
  no longer points to a deleted experimental panel. No alternate model
  promotion or extra checkbox was added.
- Excluded unsupported fake punt and fake field-goal probabilities from
  public rankings and removed intrusive deployment revision UI.
- Clarified overtime tie valuation, regulation clock validation and
  Monte Carlo uncertainty limitations.

### Verification

- [GitHub Actions Tests #233](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193413685)
  ran against commit `85f08747ce737525e178cd0afa5d19ed87e32cf2`
  after the public UI removal and test updates.
- [Live browser smoke #9](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193289281)
  **passed** against the public Streamlit site on app commit `053e4a5`,
  confirming removal of the old panel and a working official decision.
- User accepted responsive presentation. [Release-QA Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5)
  is closed. Exact hosted Git SHA remains **not independently attested**.
- Retrospective 2023–2025 research does not provide fresh untouched
  prospective evidence. Once-only 2026 Step 66/83 checks are still
  protected and never authorize automatic model promotion.

### Release readiness

See [research-demo readiness](reports/88_research_demo_release_readiness.md)
and [release checklist](reports/90_release_candidate_checklist.md).
A draft research-demo prerelease was prepared before the kickoff-panel
removal and must be brought up to date before publication. Neither
those draft materials nor any passing test makes this a stable release.

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
