# Changelog

Notable changes to Fourth Down AI are documented here.

## Unreleased — Research-demo candidate (post-v1.1.0)

**Status:** A qualified, tested research demo on `main`, NOT a new stable
`v1.2.0` release or validated coaching policy. The published `v1.1.0`
model weights remain frozen; the live Q4 research demo uses a separate
native kickoff-input correction without model retraining.

### Model-validity follow-up (October 11 Pacific / October 12 UTC)

- Rejected the fixed-spec continuous-yardline GO conversion challenger after a 2014–2022 fit and 2023–2024 development evaluation: normal GO Brier was 0.217991 frozen versus 0.218233 challenger. Zero comparable own-5 GO training examples remain, and challenger predictions were more aggressive there. No model weights or conversion odds were promoted.
- For regulation fourth-and-2/3 inside the offense's own 10 **when GO is the model leader**, the primary recommendation now says **INSUFFICIENT DATA**. Original modeled win probabilities and ranked actions remain visible with explicit sparse-support caveats; no PUNT rule was invented.
- The 2023–2024 Q4 site-calibration audit found only 7 final-five-minute own-10/leading-4 play states across 3 games (zero HOME-offense examples). It reproduced an 11.11 pp mirrored frozen-WP site difference in an illustrative failed-GO continuation. The late Q4 goal-line demo now gives a narrow site-support caption, without any home-advantage adjustment.
- Both auditable development reports and their test workflows are linked in [Issue #9 follow-up](reports/issue9_field_position_go_q4_site_followup.md). The original 2026 once-only gates, preregistered 2027 research, and published stable v1.1.0 remain untouched.

### Interface and interpretation

- Streamlit focuses on the **official** GO (pass/run), FIELD GOAL and
  PUNT recommendations. Removed the public 2025 kickoff comparison
  panel, toggle, and separate experimental-strategy results following
  user feedback.
- Retained the 538-transition 2025 kickoff donor study and onside
  sensitivity analyses in repository research files only. The native
  kickoff-input correction runs automatically in Q4 on the demo;
  no historical-scenario toggle or secondary results panel is shown.
- Added immediate modeled conversion, field-goal and punt outcome
  explanations alongside win probability or overtime game value.
- Updated the research-demo's regulation-Q4 **action probabilities and
  recommendation** with native standard-35 kickoff input encoding, in a
  private model namespace with the original frozen classifier weights.
  This fixes the previous change that merely relabeled results without
  recalculating the probabilities.
- Replaced the Q4 **INCONCLUSIVE** label and verbose reliability
  warning with a normal recommendation and concise **Close call** note
  for input-sensitive states, with no extra checkbox and no stable
  model/forward-validation promotion.
- Excluded unsupported fake punt and fake field-goal probabilities from
  public rankings and removed intrusive deployment revision UI.
- Clarified overtime tie valuation, regulation clock validation and
  Monte Carlo uncertainty limitations.
- Added a narrowly scoped **limited historical GO support** notice
  inside a team's own 10 on fourth-and-2/3 when GO is recommended.
  [Issue #9's frozen-data audit](reports/issue9_deep_territory_site_audit.md)
  found one comparable training GO in the own 1–10 and none in 1–5.
  Verified the Q4 site effect originates in learned continuation WP,
  and added a neutral HOME/AWAY mirror invariant. No frozen model
  weights, action probabilities, eligibility, or precommitted future
  validation criteria are changed.
- Persisted completed fourth-down results across harmless Streamlit
  component reruns; the result is invalidated when the game inputs change.
  Isolated the Q4 feature correction without reloading a second copy
  of the heavy model and transition data. The associated Q4 kickoff
  sensitivity diagnostic is now cached with that completed result:
  reruns no longer drop the Close call notice by referencing an
  expired per-click engine namespace.

### Verification

- **[Tests #249](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422315)
  passed on `31176d50af25db9dd42dae07e9b17cb5c2b930b4`**, including
  frozen-artifact preflight, regression tests, Streamlit AppTest and
  same-seed Q4 legacy-vs-corrected action probability checks.
- **[Live Chromium smoke #15](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422175)
  passed on the same commit**: a corrected Q4 decision, normal
  recommendation, Close call notice and result persistence were visible
  without the removed experimental control.
- User accepted responsive presentation; [Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5)
  is closed. Exact hosted Git SHA remains **not independently attested**.
- Historical 2023–2025 checks show better pooled post-score kickoff
  Brier loss but slightly worse away-kicker performance. Those years
  are not a fresh untouched benchmark. The once-only 2026 Step 66/83
  gates and their model/source hashes remain protected.

### Release readiness

See [research-demo readiness](reports/88_research_demo_release_readiness.md)
and [release checklist](reports/90_release_candidate_checklist.md).
The unpublished prerelease draft still describes the deleted kickoff
panel; [Issue #7](https://github.com/daniel-jpg/fourth-down-ai/issues/7)
tracks correction or deletion before publication. Release editing
was denied by the GitHub connection (403); the working app is unaffected.
[Issue #6](https://github.com/daniel-jpg/fourth-down-ai/issues/6)
tracks review of the automatic 2026 post-gate research results.
Neither existing historical evidence nor a passing smoke test makes
this a prospectively validated coaching policy or a new stable release.

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
