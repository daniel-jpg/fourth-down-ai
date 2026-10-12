# Step 88 — Research-demo release readiness (2026-10-11)

**Updated October 12, 2026:** The live Streamlit research demo uses
one GO/FG/PUNT decision screen; the historical kickoff comparison and
checkbox are removed. Regulation-Q4 action estimates now **actually
recompute** with the native standard-35 kickoff-input correction while
retaining the published frozen `v1.1.0` model weights. The short
**Close call** note flags representation-sensitive model edges.

[Full CI #249](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422315)
and [live Chromium #15](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422175)
passed on app commit `31176d50af25db9dd42dae07e9b17cb5c2b930b4`.
Live result persistence, the absence of the historical toggle, and the
Q4 recommendation/warning were checked. The user's mobile-layout
signoff is in closed [Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5).
Exact Streamlit Cloud SHA is not independently attested.

The 2026 prospective evaluation remains *future and gated*; no 2026
outcomes have been scored for the frozen challenger, and no stable
release or causal coaching-policy claim has been made.

## Evidence and what it does not establish

- **Most recent evidence:** [Tests #249](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422315) and [live browser #15](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422175) passed on `31176d5`. The 4:59, fourth-and-4 away-team same-seed numerical audit compared legacy FG **41.91%** versus corrected **37.00%**, and legacy PASS **34.42%** versus corrected **34.34%**. The on-screen result is the corrected-model FG lead of **2.66 pp**, not an observed coaching benefit.
- [GitHub Actions run 38182568146](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38182568146) passed **339 automated tests**: 293 regression and 46 Streamlit AppTest integration cases. These verify program behavior, not statistical calibration or counterfactual decision value. See [Step 85 test review](85_streamlit_ui_release_readiness.md).
- Subsequent **user-provided live-app output** verified the newly deployed immediate-action-odds section and proper overtime **game value** labeling, with win=1, tie=0.5, loss=0. A live screenshot verified Q4 **15:01 rejected**; a later live output verified Q4 **15:00 accepted**. The previous observable deployment mismatch was recorded and closed in [Issue #4](https://github.com/daniel-jpg/fourth-down-ai/issues/4). **The exact deployed commit hash was not independently observed.**
- Custom field dragging was previously browser smoke-tested. The user accepted the narrow/mobile layout. [Playwright live-browser run #15](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422175) passed against the public hosted app with the corrected Q4 engine, persistent result and no historical kickoff controls. This does **not** constitute independent device-farm coverage or a cryptographic deployment-SHA attestation.

## How to interpret a recommendation

The decision tool compares normal GO (pass or run), FIELD GOAL and PUNT actions using frozen model weights and simulated transitions. For regulation Q4, the research demo now computes these values using the native standard-35 post-score kickoff input correction, without updating the published v1.1.0 model artifacts. Its displayed win probabilities are **model estimates**, and the advantage in percentage points is the difference between those estimates—not a measured increase in actual wins. Immediate conversion/FG/punt outcome odds describe short-run events and must not be substituted for overall win probability. Monte Carlo standard errors measure **sampling noise of the simulations only**, excluding model misspecification, dataset shift, and uncertainty about football decisions. In overtime the target is **game value**, not simply probability of winning, because a tie at expiration contributes 0.5.

The public UI offers **no historical kickoff scenario panel**. Observed-2025 return donors and onside studies remain available only in the research repository. The narrow **native Q4 feature-encoding correction** now powers the public research-demo Q4 estimate (without a toggle), while the published stable model and precommitted future-data gates stay frozen. A short **Close call** note flags materially representation-sensitive cases; this is not a calibrated coaching edge. The 538 matched ordinary/penalty-affected 2025 Q4 kickoff transitions describe a selected historical sample; they do not constitute a 2026+ calibrated forecast. Unmatched outcomes, onside policy, return TDs, and re-kicks limit extrapolation.

## Known calibration and causal limitations

The frozen original Q4 win-probability model has a documented post-score kickoff feature-encoding sensitivity: under some near-tie game states, site or encoding changes can alter strategy rankings. Research on 2023–2025 kickoff observations found improvement in pooled Brier loss with a native kickoff feature representation, but the away-kicking subgroup slightly worsened. These years have **already been examined** for exploratory research and cannot be represented as untouched prospective evidence for selecting the challenger. A better retrospective kickoff forecast also does **not** identify the counterfactual outcome of an unchosen fourth-down play. See [Q4 model audit](2025_q4_kickoff_site_model_defect.md), [broader calibration study](2023_2025_broad_q4_native_kickoff_calibration.md) and the [frozen Step 83 protocol](2026_native_kickoff_challenger_protocol.md).

The precommitted 2026 Weeks 6–18 **Step 66** one-time gate requires at least 100 distinct eligible completed games; the **Step 83** native challenger is triggered only after the original gate succeeds and must meet its own fixed reliability criteria. **Do not** fetch prospective outcomes ahead of the gates, tune the frozen model, run these workflows manually, or automatically promote a challenger. Even a successful prospective kickoff reliability check would not establish a causal fourth-down coaching advantage.

## Demo checklist (what can be shown responsibly)

1. Enter a valid fourth-down game state and display the GO/FG/PUNT comparison and run/pass uncertainty. For Q4, disclose the provisional native kickoff-input correction and avoid presenting its edge as prospectively validated.
2. Expand **Why this recommendation? Immediate action odds** and explain short-run conversion or kick probabilities versus estimated full-game WP.
3. In overtime, explain **game value** and its explicit tie treatment.
4. Verify no historical kickoff comparison controls appear in the public app; direct research questions to the versioned audit reports.
5. State the known model/causal limits and the future one-time validation plan without treating research results as guarantees.

## Release gate and scope

- **Ready now:** technical research demo of the tested `main` app with explicit uncertainty and limitations. [Latest demonstrated CI/live evidence](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422315) is pinned to `31176d5`; subsequent documentation-only changes are not a new model evaluation.
- **Not claimed:** production-grade coaching advice, independently calibrated 2026+ field-goal/onside policy, independently certified responsive behavior on all devices, or a new stable tagged version.
- **Functional release QA:** completed in [closed Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5). User accepted responsive presentation; hosted end-to-end behavior passed. **Exact hosted revision identity:** not independently attested; keep it as a documented limitation rather than reintroducing a build-ID widget. New stable-version publication and claims require separate signoff. [Issue #6](https://github.com/daniel-jpg/fourth-down-ai/issues/6) tracks post-gate validation review. [Issue #7](https://github.com/daniel-jpg/fourth-down-ai/issues/7) tracks an obsolete unpublished release draft that cannot be safely published as-is.
- **Frozen components:** No model coefficients, eligible-action rules or Step 66/83 precommitted protocols are modified. The Q4 research-demo display now uses the independently evaluated native input encoding and recomputes strategy probabilities; it is not an official stable-model release.
