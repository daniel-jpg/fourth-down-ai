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
| Streamlit integration | 46 UI cases pass locally; all check for unexpected Streamlit exceptions | Passed |
| Combined suite | 339 tests pass locally | Passed |
| Custom field | Python-side initial field state reaches the engine | Covered only in Python |
| Drag gestures, browser layout, hosting | AppTest does not run the component's JavaScript or connect to the hosted app | Separate browser check required |

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

A browser test of the custom field drag gesture and the hosted deployment is
still needed before describing this as complete browser end-to-end coverage.
AppTest's collapsed expander contents are inspected as rendered Python output;
it does not verify the browser animation for opening the expander.
