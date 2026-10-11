# Step 85: Streamlit UI integration and release-readiness review

Reviewed against `main` at `f3b5f7b96e183fb8ab6dc324789e80aa75803529` on
2026-10-11. This review concerns application behavior, not a new validation
of model accuracy or counterfactual fourth-down decisions.

## Changes

- Added 46 AppTest cases that run the actual `app.py`, manipulate widgets,
  click Analyze and inspect metrics, dataframes, warnings and exceptions.
- Most cases use fixed responses at the engine/simulation boundaries. Three
  smoke cases load the real frozen engine in regulation, Q4 and overtime.
- Each UI case forbids Python socket connections. All runtime models and
  frozen historical preview summaries are available from the checkout.
- Reproduced and fixed a clock-validation gap: regulation accepted `15:01`
  through `15:59`. The UI now rejects clocks above `15:00` in every regulation
  quarter, while accepting the `15:00` boundary.
- Corrected the immediate-odds explainer's OT label and captions to say
  **game value**, including the 0.5 tie value, instead of game win probability.
- Added a named GitHub Actions UI step, a 15-minute job timeout and a retained
  JUnit artifact. Regression and UI cases run once each in CI.

## Checks

| Area | Evidence | Assessment |
| --- | --- | --- |
| Inputs and result display | Regulation widgets pass the correct state; metrics and immediate odds match engine output | Covered |
| Eligibility | Unavailable normal actions have no displayed probability; fakes never enter official metrics or immediate odds | Covered |
| Optional failures | Explanation, reliability probe and experimental failures retain the official result with visible fallback text | Covered |
| Q4 reliability | Automatic material warning, scope boundaries and an actual frozen-engine home-site warning | Covered |
| Overtime | Format, timeout limits, clock limits, opening/response/sudden-death validity and value labels | Covered |
| Experimental separation | Default off; explicit selection required; a disagreeing experimental FG leader cannot replace the official GO result | Covered |
| Runtime artifacts | Actual frozen-engine UI smoke cases run without network connections | Covered |
| Existing regressions | 293 pre-existing tests pass locally with pinned project dependencies | Passed |
| Streamlit integration | 46 UI cases pass locally and in GitHub Actions; all check for unexpected Streamlit exceptions | Passed |
| Combined suite | 339 tests pass locally and in GitHub Actions (293 regression + 46 UI) | Passed |
| Custom field | Python-side initial field state reaches the engine | Covered only in Python |
| Live drag gesture | Browser drag from opponent 30 to opponent 20 updates both controls, the estimated FG distance to 38 yd and the decision's field position | Passed browser smoke |
| Hosted release | Freshly reloaded hosted app still omits the immediate-odds panel present in tested `main` | Deployment verification blocked |

The implementation commit is `b81de6de284731a25700caa8100b7e8e8a291170`.
Its [GitHub Actions run](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38182568146)
completed successfully, including the named UI step and uploaded JUnit report.

## Frozen-model and prospective-validation safeguards

No model weights, decision-engine code, simulation budgets, seeds, eligibility
rules or experimental algorithms changed. Valid input states retain the same
official calculations. Only an impossible regulation clock is newly rejected.

The Step 66 and Step 83 prospective workflows and source files are unchanged.
Step 66 was not manually executed, and no future outcomes were downloaded or
inspected. Existing future-data tests use synthetic fixtures.

The frozen artifacts still match their registered Git blob hashes:

| Artifact | Git blob SHA |
| --- | --- |
| `models/win_probability_model.joblib` | `4cdafd1687594ff624560ad140725d9c32d72a17` |
| `data/win_probability_spec.json` | `857ef77f7ca0a041502ed28811fdcbf48a5d6e95` |
| `src/83_forward_2026_native_challenger_once.py` | `90ea87140de3aa3bd442bc5546cb549bdea6f0cf` |

## Remaining release limits

Passing UI tests establishes tested application wiring, not real-world model
calibration. The known Q4 post-FG kickoff representation sensitivity still
requires its displayed reliability warning; research comparisons remain
opt-in. The frozen 2026 prospective gate remains the next independent model
evaluation, with no automatic production promotion.

The [live app](https://fourth-down-ai.streamlit.app/) passed a browser drag and
analysis smoke check. However, after a fresh reload and Analyze, its action
section still went directly to **Show detailed model output** without the
**Why this recommendation? Immediate action odds** panel in current `main`.
It also accepted `15:01` without the new regulation-clock error. The exact
hosted source revision was not exposed, so this is evidence of a deployment
mismatch rather than a claim about its precise running commit.

See the [browser evidence](../assets/browser-screenshot-1791759852220.jpg).
At the observed 1363-by-936 browser viewport, the RUN and FIELD GOAL
percentages also crowd each other. Recheck this layout on the newest hosted
source before visual signoff; the running deployment revision is unknown.
Update/verify the Streamlit deployment's repository, branch and entrypoint
against `daniel-jpg/fourth-down-ai`, `main`, `app.py`, then repeat the hosted
feature checks. GitHub repository write access does not provide Streamlit
Cloud deployment administration. No owner deployment controls were available
in this browser session.

AppTest does not execute browser JavaScript or validate layout/expander
animation. The live smoke check is a separate observed interaction, not
automated browser CI coverage or a certification of every hosted input.
