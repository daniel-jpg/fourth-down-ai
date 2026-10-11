# Step 88 — Research-demo release readiness (2026-10-11)

**Status:** Current `main` is suitable for a **qualified research demonstration**, subject to the open visual/deployment signoff in [Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5). This is **not** a new tagged release or a validated real-world coaching policy. Stable tagged release remains **v1.1.0**.

## Evidence and what it does not establish

- [GitHub Actions run 38182568146](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38182568146) passed **339 automated tests**: 293 regression and 46 Streamlit AppTest integration cases. These verify program behavior, not statistical calibration or counterfactual decision value. See [Step 85 test review](85_streamlit_ui_release_readiness.md).
- Subsequent **user-provided live-app output** verified the newly deployed immediate-action-odds section and proper overtime **game value** labeling, with win=1, tie=0.5, loss=0. A live screenshot verified Q4 **15:01 rejected**; a later live output verified Q4 **15:00 accepted**. The previous observable deployment mismatch was recorded and closed in [Issue #4](https://github.com/daniel-jpg/fourth-down-ai/issues/4). **The exact deployed commit hash was not independently observed.**
- Custom field dragging was previously browser smoke-tested. Narrow/mobile layout, current desktop metric spacing and exact deployed SHA remain **unverified**; tracked in [Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5). Do not describe this as full mobile or browser end-to-end certification.

## How to interpret a recommendation

The official system compares normal GO (pass or run), FIELD GOAL and PUNT actions using frozen models and simulated transitions. Its displayed win probabilities are **model estimates**, and the advantage in percentage points is the difference between those estimates—not a measured increase in actual wins. Immediate conversion/FG/punt outcome odds describe short-run events and must not be substituted for overall win probability. Monte Carlo standard errors measure **sampling noise of the simulations only**, excluding model misspecification, dataset shift, and uncertainty about football decisions. In overtime the target is **game value**, not simply probability of winning, because a tie at expiration contributes 0.5.

Optional 2025 observed-kickoff, onside, and native-input analyses are **experimental** and remain separate from the official recommendation. The 538 matched ordinary/penalty-affected 2025 Q4 kickoff transitions describe a selected historical sample; they do not constitute a 2026+ calibrated forecast. Unmatched outcomes, onside policy, return TDs, and re-kicks limit extrapolation.

## Known calibration and causal limitations

The frozen original Q4 win-probability model has a documented post-score kickoff feature-encoding sensitivity: under some near-tie game states, site or encoding changes can alter strategy rankings. Research on 2023–2025 kickoff observations found improvement in pooled Brier loss with a native kickoff feature representation, but the away-kicking subgroup slightly worsened. These years have **already been examined** for exploratory research and cannot be represented as untouched prospective evidence for selecting the challenger. A better retrospective kickoff forecast also does **not** identify the counterfactual outcome of an unchosen fourth-down play. See [Q4 model audit](2025_q4_kickoff_site_model_defect.md), [broader calibration study](2023_2025_broad_q4_native_kickoff_calibration.md) and the [frozen Step 83 protocol](2026_native_kickoff_challenger_protocol.md).

The precommitted 2026 Weeks 6–18 **Step 66** one-time gate requires at least 100 distinct eligible completed games; the **Step 83** native challenger is triggered only after the original gate succeeds and must meet its own fixed reliability criteria. **Do not** fetch prospective outcomes ahead of the gates, tune the frozen model, run these workflows manually, or automatically promote a challenger. Even a successful prospective kickoff reliability check would not establish a causal fourth-down coaching advantage.

## Demo checklist (what can be shown responsibly)

1. Enter a valid fourth-down game state and display official GO/FG/PUNT comparison and run/pass uncertainty.
2. Expand **Why this recommendation? Immediate action odds** and explain short-run conversion or kick probabilities versus estimated full-game WP.
3. In overtime, explain **game value** and its explicit tie treatment.
4. Toggle the optional historical kickoff preview only with its experimental caveats, showing that it never replaces the official output.
5. State the known model/causal limits and the future one-time validation plan without treating research results as guarantees.

## Release gate and scope

- **Ready now:** technical research demo of the tested `main` app with explicit uncertainty and limitations.
- **Not claimed:** production-grade coaching advice, independently calibrated 2026+ field-goal/onside policy, complete mobile usability, or a new stable tagged version.
- **Before formal public release signoff:** resolve or explicitly waive Issue #5's responsive-layout review and deployed-commit verification. Any actual layout defect should be addressed and tested separately.
- **Frozen components:** No changes to model weights, official decision probabilities, action rules, experimental algorithms, or Step 66/83 preregistrations were made by this document.
