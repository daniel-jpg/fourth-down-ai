# 2027 Fourth-Down AI: nflverse-shaped input to frozen policy (FABRICATED ONLY)

**Prepared:** October 12, 2026. **Status:** offline end-to-end engineering rehearsal; no NFL 2027 data are available or accessed. The frozen decision engine and separate 2026 once-only validation remain intact.

**Chosen source:** existing nflreadpy/nflverse play-by-play format; **no Sportradar subscription, vendor inquiry, live timestamp authority or paid service is required** for this retrospective out-of-time evaluation path.

## Verified CI evidence (October 12, 2026)

The [synthetic-only nflverse-to-frozen-engine workflow #38284784223](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38284784223) completed **successfully** using commit [515d9173](https://github.com/daniel-jpg/fourth-down-ai/commit/515d9173ecab17470b78a15a98130d530387df9f). It passed **19 dedicated regression tests**, then exercised a single **invented Q4 fourth down at the opponent 6** through the actual verified frozen model using full 1,200-trial native-kickoff logic. Its fictional sample selected NORMAL_GO_PASS; that is an **illustration only**, not an observed coaching advantage or a real NFL 2027 recommendation. The logged machine-readable audit confirmed frozen engine provenance, one fake forecast, zero missing decisions, no label access and all future/prospective certification flags false. The dedicated run archived only a fabricated-source audit artifact for 14 days.

This result tests **software integration**, not forecasting accuracy, actual before-snap timing or 2027 historical extrapolation.

## Implementation

- [Nflverse field projector](../src/nflverse_2027_reconstruction.py): accepts supplied pandas/Polars frames, selects source-reported fourth downs on the **down field alone**, projects only approved pre-play-looking state columns and records exclusions. It separately exposes observed-action and final-score *claims*, but this forecast integration **never invokes that label projection**.
- [New frozen engine replay](../src/nflverse_2027_frozen_policy_replay.py): accepts only fabricated_2027_only mode and fictional game IDs beginning with synthetic_; it converts the fixed projection into [the pinned policy bridge](../src/frozen_2027_policy_bridge.py) input. Q2 rows with missing second-half receiver, missing roof context, implausible goal-line/clock/distance, duplicate IDs and other invalid game states are either skipped with explicit reasons or fail closed. Every source candidate is forecast or accounted for.
- [Existing pinned engine and nuisances](../src/frozen_2027_policy_bridge.py): the default verifies original policy/model hashes and historical nuisance model provenance. The **300/1,200 adaptive Q1–Q3**, **Q4 native standard-35 1,200/seed43**, original eligibility rules and 2014–2025 nuisance probability/reward estimators are reused without fitting new models.
- [Synthetic regression suite](../tests/test_nflverse_2027_frozen_policy_replay.py): covers separation of labels from inputs, pandas/Polars parity, context gaps, duplicates, source exclusions and full registered-engine Q4 precision on **invented data**.
- [Focused CI](../.github/workflows/2027_nflverse_frozen_engine_bridge.yml): runs the tests, executes one fabricated Q4 policy smoke test with full registered precision, checks false prospective claims, and archives an aggregate for 14 days. No cron, real future games, protected Step 66/83 scoring, or public-app changes.

The engine's accepted feature_asof=pre_snap contract is used **only inside synthetic fixtures**. The adapter retains source status POSTGAME_RECONSTRUCTED_NOT_WITNESSED, a real_2027_holdout_gate of NOT_AUTHORIZED_OR_COMPLETED, actual_pre_snap_forecasts_observed=false, calibration_evaluated=false and future_season_scoring_authorized=false. Do **not** describe synthetic policy outputs as actual live forecasts, future empirical scores or causal coaching benefit.

## Running now on invented data only

Run these from the repository root after installing requirements-dev.txt:

    PYTHONPATH=. python -m pytest -q tests/test_nflverse_2027_frozen_policy_replay.py
    PYTHONPATH=. python -m src.nflverse_2027_frozen_policy_replay --fabricated-demo-only --audit-json-out fake_policy_preview.json

The CLI invents one 2027-shaped Q4 fourth down and loads the verified frozen hashes/models. Console output can include simulator diagnostic lines; the optional --audit-json-out argument writes a new, clean JSON audit file using exclusive creation (it refuses to overwrite an existing file). It has no URL, real game-ID input or bypass of the synthetic gate. A fake engine can be injected only for fabricated testing and is labeled separately.

## Remaining work before a genuine nflverse-based 2027 evaluation

The current bridge **always denies** real 2027 intake, even with a caller-supplied completed-season flag. There is no authorized real-data scoring entry point.

Before an *entirely separate* post-season scoring path can be reviewed and opened:

1. Wait until **all 2027 NFL REG Weeks 1–18 games are final** and the complete NFLverse source version is published. Independently verify season completeness, cancellations/ties, game/play keys, acquisition version, and finished-game scores. A JSON "completed" flag alone is not proof.
2. Independently review any future NFLverse schema drift **without tuning based on the reserved 2027 outcomes**. Lock input whitelists, exclusion rules, original policy commit, outcome-provenance procedures and source hashes.
3. Investigate second-half receiving-team direction, relocated stadium/roof context and unsupported kicker/FG states. These must be recorded as abstentions in the complete opportunity denominator rather than inferred from future results.
4. Run the frozen model with original 300/1,200 and native Q4 parameters, and seal prediction hashes **before** separately keyed actual observed coach action and final-winning outcomes are joined.
5. Run the [registered factual action and observational comparison](2027_fourth_down_action_policy_protocol.md) once with game clustering, complete exclusions, registered subgroups and insufficient-overlap suppression. Label the result a *future-season retrospective reconstruction*, **not a live timestamped before-snap study**.
6. Leave the original protected **2026 Step 66/83** validation workflows untouched; never use their holdouts for feature design.

This stage proves **software plumbing and safeguards only**. Paid live data and independently attested snap times are optional for a separate, stronger realtime research question, not necessary for the ordinary nflverse retrospective path.
