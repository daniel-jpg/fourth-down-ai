# Step 90 — Research-demo candidate packaging checklist

Prepared 2026-10-11. **This document is a release candidate checklist, not a declaration that a new stable version has been published.** The last tagged stable release remains `v1.1.0`. The working branch is `main`, with additional experimental research and presentation changes.

## Candidate scope

- Streamlit interactive app for regulation and both overtime formats.
- Official comparison of eligible normal fourth-down GO (pass/run), FG and PUNT using frozen production artifacts.
- Presentation-only immediate modeled action odds and uncertainty statements.
- Optional, isolated retrospective kickoff scenario tools, with explicit model limitations.
- No changes to production coefficients, no automatic candidate promotion, and no manual future-data validation execution.

## Reproducibility instructions

Use Python **3.12** in a clean clone at a **recorded commit SHA** (do not write `main` in a report as if it were immutable):

```bash
git clone https://github.com/daniel-jpg/fourth-down-ai.git
cd fourth-down-ai
git rev-parse HEAD
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

All runtime dependencies and pytest are pinned in `requirements.txt` and `requirements-dev.txt`. Historical and model artifacts are checked into the project and used by the deployed app without downloading live NFL data. An internet connection is necessary to install dependencies; future outcome datasets are **not** prerequisites for the application or its tests.

For the specific Step 85 implementation, [GitHub Actions run 38182568146](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38182568146) passed 339 tests (293 regression + 46 UI AppTest). That result is evidence for the tested revision, **not** proof that every later commit has independently passed CI.

## Verified release-QA baseline (2026-10-12)

**Evidence-pinned application revision:** `8239a8f7666b2cf00bf792c5977119c67127693a`.
This SHA was the head for *both* successful GitHub Actions runs:

- [Tests #229](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38191887196) — passed Python syntax, frozen-artifact release preflight,
  regression tests and Streamlit AppTest.
- [Hosted browser smoke #8](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38191887243) — passed in Chromium against the public
  Streamlit URL. Expanded the Q4 kickoff panel, enabled its only comparison
  toggle, clicked **Analyze Fourth Down**, and observed both the official
  decision and the completed historical scenario result.
- [Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5) — closed
  after the functional live QA and user acceptance of responsive layout.

## Pre-release gates

| Gate | Status (2026-10-12) | Evidence or qualification |
| --- | --- | --- |
| Frozen official artifacts and research protocols | Passed | Preflight in Tests #229 verified locked Git blobs; no model refit or early Step 66/83 execution |
| Regression and UI AppTest | Passed | [Tests #229](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38191887196) on `8239a8f` |
| Hosted kickoff comparison | Passed | [Live browser #8](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38191887243) on `8239a8f`, including official and experimental output |
| Immediate outcome explainer and OT game-value wording | Previously observed working | User-supplied hosted evidence; AppTest also exercises these flows |
| Regulation clock boundary | Previously observed working | Q4 15:01 rejected; 15:00 accepted |
| Live mobile/narrow layout | User accepted | Not claimed to be an independent multi-device visual test |
| Exact Streamlit Cloud deployed Git SHA | **Not verified** | Hosted functional behavior passed, but platform deployment SHA was not independently attested; no user-visible build-info panel |
| Independent 2026 kickoff reliability | Future prospective gate pending | Keep frozen Step 66/83 protocols and thresholds unchanged; never run early or auto-promote |
| New stable tag / GitHub Release | **Not created** | Separate version-number and research-only claim review required |

**Interpretation:** Passed end-to-end *functionality* supports a qualified public
research demo, but does not prove byte-for-byte deployed revision identity,
real-world calibration or causal coaching benefit. Later changes to the app
or model would require a new evidence-pinned QA run. Documentation-only edits
do not turn the historical browser result into a new experiment.

## Model limitations that must remain visible

These are model-estimated probabilities and Monte Carlo sampling errors, not causal coaching impact or total predictive uncertainty. Historic 2025 model performance originally provided a held-out v1.1.0 benchmark, but subsequent exploratory 2025 research means those years are not untouched future holdouts. The original frozen Q4 post-score kickoff encoding exhibits known site-sensitive rankings, and proposed corrections remain research-only. Observed 2025 kickoff donor coverage excludes unsupported onside/re-kick/return-TD states. Even passing a prospective kickoff Brier validation would not establish optimal fourth-down coaching decisions.

## Publishing instructions — intentionally NOT executed

1. Close or explicitly document acceptance of remaining release signoff limitations from Issue #5.
2. Verify the latest candidate HEAD passes all required GitHub Actions jobs and capture its exact SHA; keep evidence tied to that SHA.
3. Recheck README/CHANGELOG links and a representative live regulation and OT walkthrough.
4. Review the version number and release claims. Only after these gates, create an annotated version tag and GitHub Release with a clear research-only disclaimer.
5. Do **not** move or rewrite the stable `v1.1.0` tag; do not retrain model weights or use future data to accelerate tagging.
