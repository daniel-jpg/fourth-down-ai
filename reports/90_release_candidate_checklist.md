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

## Pre-release gates

| Gate | Status as of 2026-10-11 | Evidence or remaining action |
| --- | --- | --- |
| Frozen decision artifacts | Protected | No model, decision-engine, or simulation algorithm changes made for Steps 85–90; Step 85 report lists model/spec hashes |
| Regression and UI tests | Passed on Step 85 implementation | 339-test CI run linked above; recheck latest candidate head before tagging |
| Live immediate-odds UI | Observed working | User-supplied current hosted screenshot |
| Live overtime game-value wording | Observed working | User-supplied current hosted OT output |
| Clock max | Observed working | Q4 15:01 rejected and 15:00 accepted |
| Live mobile/narrow layout | Not verified | [Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5) |
| Hosted deployed commit identity | Not verified | [Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5) |
| Independent 2026 kickoff reliability | Pending future gate | Do not run Step 66 or Step 83 early; do not promote any challenger |
| New stable tag/GitHub Release | **Not created** | Must be separately reviewed after gates |

## Model limitations that must remain visible

These are model-estimated probabilities and Monte Carlo sampling errors, not causal coaching impact or total predictive uncertainty. Historic 2025 model performance originally provided a held-out v1.1.0 benchmark, but subsequent exploratory 2025 research means those years are not untouched future holdouts. The original frozen Q4 post-score kickoff encoding exhibits known site-sensitive rankings, and proposed corrections remain research-only. Observed 2025 kickoff donor coverage excludes unsupported onside/re-kick/return-TD states. Even passing a prospective kickoff Brier validation would not establish optimal fourth-down coaching decisions.

## Publishing instructions — intentionally NOT executed

1. Close or explicitly document acceptance of remaining release signoff limitations from Issue #5.
2. Verify the latest candidate HEAD passes all required GitHub Actions jobs and capture its exact SHA; keep evidence tied to that SHA.
3. Recheck README/CHANGELOG links and a representative live regulation and OT walkthrough.
4. Review the version number and release claims. Only after these gates, create an annotated version tag and GitHub Release with a clear research-only disclaimer.
5. Do **not** move or rewrite the stable `v1.1.0` tag; do not retrain model weights or use future data to accelerate tagging.
