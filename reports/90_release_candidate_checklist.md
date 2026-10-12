# Step 90 — Research-demo candidate packaging checklist

Prepared 2026-10-11. **This document is a release candidate checklist, not a declaration that a new stable version has been published.** The last tagged stable release remains `v1.1.0`. The working branch is `main`, with additional experimental research and presentation changes.

## Candidate scope

- Streamlit interactive app for regulation and both overtime formats.
- Single-screen comparison of eligible normal fourth-down GO (pass/run), FG and PUNT. Stable prediction weights remain frozen; Q4 research-demo results use native standard-35 kickoff input encoding instead of the previously misleading missing-field representation.
- Presentation-only immediate modeled action odds and uncertainty statements.
- Historical 2025 kickoff donor/onside simulations remain research-only, without a public checkbox. The native Q4 feature-encoding correction now runs automatically for public research-demo Q4 action estimates, with a close-call limitation note.
- No changes to published stable-model coefficients, no stable release promotion, and no manual execution of future-data validation. The research-demo UI uses a provisional Q4 input correction requested by the owner.

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

**Core public-app revision:** `053e4a5b7c16c88ef129fd39e6b780b8d965dda2`
removes the comparison, toggle and experiment result branch without
modifying the official engine, frozen data or future-data protocols.
[Live browser smoke #9](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193289281)
**passed** on that revision, confirming the hosted panel is gone and
the ordinary Analyze Fourth Down recommendation renders.

**Regression revision:** `85f08747ce737525e178cd0afa5d19ed87e32cf2`
updates two older guard/native-input source tests to reflect the
intentional removal, while retaining the automatic Q4 reliability
warning and the independently tested research module. See
[Tests #233](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193413685)
for regression, Streamlit AppTest and frozen-artifact preflight
evidence. There are no application changes between those commits.

## Pre-release gates

| Gate | Status (2026-10-12) | Evidence or qualification |
| --- | --- | --- |
| Frozen production model and prospective protocols | Protected | Preflight locks files and source hashes; Step 66/83 not run early |
| Regression and UI AppTest | See Tests #233 | [Full suite](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193413685) on `85f0874` after tests were updated |
| Hosted official-only app (kickoff controls removed) | **Passed** | [Live smoke #9](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38193289281): no panel and official recommendation rendered |
| Immediate action odds and OT game value | Previously verified | User-supplied hosted evidence and Streamlit AppTest cases |
| Regulation clock validation | Previously verified | 15:01 rejected in Q4 and 15:00 accepted |
| Live mobile/narrow layout | User accepted | Not independent multi-device visual certification |
| Exact Streamlit Cloud deployed SHA | **Not verified** | Hosted UI behavior verified; never reintroduce a large build-info panel |
| Independent future 2026 validation | Pending once-only gate | Step 66/83 rules unchanged, no automatic model promotion |
| New stable tag/GitHub release | **Not published** | Keep `v1.1.0` stable, refresh old draft notes before public release |

**Scope:** The preceding live-browser #8 test verified the former
public option; it is *not* evidence that the removed panel remains
available. The new #9 test checks the final user-facing scope.
Passing functionality does not establish causal coaching advantage,
total model uncertainty or a byte-for-byte hosted revision attestation.

## Model limitations that must remain visible

The Q4 demo now recomputes action probabilities with native kickoff features in a separate model module. These are model-estimated probabilities and Monte Carlo sampling errors, not causal coaching impact or total predictive uncertainty. Historic 2025 model performance originally provided a held-out v1.1.0 benchmark, but subsequent exploratory 2025 research means those years are not untouched future holdouts. The original frozen Q4 post-score kickoff encoding exhibits known site-sensitive rankings, and the provisional correction is being used in the research demo without promotion to the published v1.1.0 stable model. Observed 2025 kickoff donor coverage excludes unsupported onside/re-kick/return-TD states. Even passing a prospective kickoff Brier validation would not establish optimal fourth-down coaching decisions.

## Publishing instructions — intentionally NOT executed

1. Close or explicitly document acceptance of remaining release signoff limitations from Issue #5.
2. Verify the latest candidate HEAD passes all required GitHub Actions jobs and capture its exact SHA; keep evidence tied to that SHA.
3. Recheck README/CHANGELOG links and a representative live regulation and OT walkthrough.
4. Review the version number and release claims. Only after these gates, create an annotated version tag and GitHub Release with a clear research-only disclaimer.
5. Do **not** move or rewrite the stable `v1.1.0` tag; do not retrain model weights or use future data to accelerate tagging.
