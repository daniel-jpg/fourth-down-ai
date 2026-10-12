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

## Verified release-QA baseline (October 12, 2026)

**Pinned app/tested commit:**
`31176d50af25db9dd42dae07e9b17cb5c2b930b4`.

- [GitHub Actions Tests #249](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422315) —
  **passed**. Covers regression tests, Streamlit AppTest, frozen
  artifacts and same-seed Q4 legacy-versus-corrected action WP.
- [Hosted browser smoke #15](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422175) —
  **passed**. Corrected Q4 decision, normal recommendation, Close call,
  stable result after rerun and no second comparison/checkbox.
- The exact 4:59 / fourth-and-4 / opponent-6 / down-four AWAY
  fixed-seed audit changed FIELD GOAL from **41.91% to 37.00%**,
  PASS from **34.42% to 34.34%**, and the model FG edge
  from **7.49 pp to 2.66 pp**. This is input representation
  sensitivity, not a measured increase in actual wins.
- User accepted mobile presentation in closed
  [Issue #5](https://github.com/daniel-jpg/fourth-down-ai/issues/5).

## Pre-release gates

| Gate | Status | Qualification |
| --- | --- | --- |
| Frozen stable `v1.1.0` artifacts | **Passed** | Frozen hashes and unchanged stable model weights verified by CI #249 |
| Regression, UI AppTest and fixed-state numeric audit | **Passed** | [Tests #249](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422315), pinned to `31176d5` |
| Live hosted Q4 result and missing historical panel | **Passed** | [Browser #15](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38197422175), same SHA |
| OT game-value wording and regulation clock | Previously verified | Streamlit AppTest and earlier hosted evidence |
| Responsive/narrow layout | User accepted | Does not establish all-device visual coverage |
| Exact Streamlit-hosted deployment commit SHA | **Not verified** | Browser verified behavior, not cryptographic deployment provenance; no intrusive on-page build-ID control |
| Independent 2026 kickoff reliability | **Pending automatic once-only validation** | Step 66: at least 100 distinct eligible completed 2026 Weeks 6–18 games. Step 83 runs only afterward and checks overall Brier CI and home/away groups. See [Issue #6](https://github.com/daniel-jpg/fourth-down-ai/issues/6) |
| New stable GitHub release | **Not created** | `v1.1.0` stays stable. The obsolete unpublished draft must be corrected or discarded before publication: [Issue #7](https://github.com/daniel-jpg/fourth-down-ai/issues/7) |

**Scope:** The Q4 public demo uses provisional native kickoff-input
features with *unchanged classifier weights*; the separately published
stable version remains frozen. Retrospective reliability improvement
does not establish optimal fourth-down policy or the effect of an
unchosen FG/GO decision. The 2026 future-gate workflow must **not**
be manually accelerated. Changing only these documents does not
retest the deployment.

## Model limitations that must remain visible

The Q4 demo now recomputes action probabilities with native kickoff features in a separate model module. These are model-estimated probabilities and Monte Carlo sampling errors, not causal coaching impact or total predictive uncertainty. Historic 2025 model performance originally provided a held-out v1.1.0 benchmark, but subsequent exploratory 2025 research means those years are not untouched future holdouts. The original frozen Q4 post-score kickoff encoding exhibits known site-sensitive rankings, and the provisional correction is being used in the research demo without promotion to the published v1.1.0 stable model. Observed 2025 kickoff donor coverage excludes unsupported onside/re-kick/return-TD states. Even passing a prospective kickoff Brier validation would not establish optimal fourth-down coaching decisions.

## Publishing instructions — intentionally NOT executed

1. Note that Issue #5 is already closed; preserve its accepted mobile scope and documented deployed-SHA limitation.
2. Verify the latest candidate HEAD passes all required GitHub Actions jobs and capture its exact SHA; keep evidence tied to that SHA.
3. Recheck README/CHANGELOG links, the current public UI and a representative live regulation and OT walkthrough. Resolve [Issue #7](https://github.com/daniel-jpg/fourth-down-ai/issues/7) first; the existing draft still advertises a deleted checkbox.
4. Review the version number and release claims. Only after these gates, create an annotated version tag and GitHub Release with a clear research-only disclaimer.
5. Do **not** move or rewrite the stable `v1.1.0` tag; do not retrain model weights, manually run the 2026 Step 66/83 evaluations or use future data to accelerate tagging. See [future validation review](https://github.com/daniel-jpg/fourth-down-ai/issues/6).
