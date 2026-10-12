# Isolated historical intake for a future fourth-down model (shadow development v1)

**Created:** October 12, 2026. **Status:** development-only source/staging pipeline. No newly admitted seasons, trained model, live feed, or model deployment.

The next version of the fourth-down AI can use nflverse again **after** its new data have been formally admitted to an independent development version. For now, prospective 2026/2027 outcomes are not eligible for a newly trained model, and the frozen 2026 validation Steps 66/83 and registered 2027 policy must not be compromised.

## Provenance and scope

The original data builder, [01_build_fourth_down_dataset.py](../src/01_build_fourth_down_dataset.py), loaded historical fourth downs with nflreadpy.load_pbp for 2014–2025. This was a **retrospective** source, updated by nflverse, not a live before-snap capture. The subsequent [historical labeler](../src/03_label_fourth_down_actions.py) and [isolated pre-snap cohort extractor](../src/historical_fourth_down_cohort.py) already separated decision-time predictors, subsequently observed normal action, and final-game winner.

The new [shadow intake](../src/shadow_historical_data_refresh.py) reuses that historical extractor, rather than writing an alternative action parser. It **requires** Git blob SHA-1 d7230df63cad1c24579ee40cabaacd0b330e4784 for the source data/fourth_downs_labeled.parquet before parsing even one play. Any substituted or updated (including 2026, 2027 or newer) PBP file fails **before opening Parquet**. It then reproduces the separate [2014–2025 external ESPN final-score archive comparison](../src/independent_espn_margin_archive.py), 3,151/3,151 games with exact final-score margins.

### Split contract

| Historical seasons | Role in shadow engineering |
| --- | --- |
| 2014–2022 | Historical model-training candidate |
| 2023–2024 | Historical validation candidate |
| 2025 | **Previously examined development only** — cannot be called a new independent test or used to select models for the registered 2027 policy |
| 2026 | Prohibited from this intake, including protected kickoff holdout |
| 2027 | Prohibited from this intake, including preregistered 2027 prospective holdout |
| 2028 or later | Not admitted by this version; requires a new reviewed source policy, separate data vintage and explicit study boundary |

No training, calibration retuning, model selection, scoring of 2026/2027 heldout data, or deployment occurs. New development candidates will receive their own artifacts, evaluation metrics, and approval process rather than changing original frozen assets.

## Verified complete historical staging result

The [passing full staging run](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38277133358) confirms that the source contains **50,095** fourth-down candidates in total: **47,942** regular-season and **2,153** postseason/other candidates. The independent ESPN final-score audit's 47,942 denominator applies to **regular-season games only**, not the entire original labeled file.

The historical normal-action extractor includes **44,507** usable decisions and records **5,588** exclusions (including 2,153 non-REG rows). Its explicit development memberships are **32,961** decisions from 2014–2022, **7,794** from 2023–2024, and **3,752** previously inspected 2025 decisions. The focused source/feature/split tests all passed, the five staged tables' SHA-256 digests matched their manifest, and no tracked frozen model or training data were changed.

The test has **not** produced a new learned model, processed 2026/2027 NFL plays, or assigned prospective meaning to backfilled historical features.

## What the pipeline produces

Running the module on an already checked-out historical repository produces one isolated untracked directory:

    PYTHONPATH=. python -m src.shadow_historical_data_refresh

The directory artifacts/shadow_historical_2014_2025 contains:

- presnap_features.parquet — game and play IDs plus **only** the defined pre-snap covariates, never an observed coach choice, result, conversion, end state or winner.
- observed_actions.parquet — the separately observed, post-play normal action labels (GO run/pass, FG, PUNT).
- postgame_outcomes.parquet — terminal offense win/loss labels (not input to the pre-snap forecast).
- excluded_candidates.parquet — **every** rejected historical fourth down, keyed and assigned an explicit reason.
- chronological_roles.parquet — per-game/play training, validation or previously-seen-development role, never based on observed outcomes.
- manifest.json — source Git blob and SHA-256, independently confirmed final-game source hashes, row counts, split counts, exclusion breakdown, feature and label schemas, canonical table row digests and staged Parquet SHA-256 digests. The manifest content is itself hashed.

Outputs are placed under artifacts/ (ignored from Git). The writer refuses to overwrite a prior snapshot or write under tracked data/, models/, reports/ or src/. The [CI workflow](../.github/workflows/shadow_historical_data_refresh.yml) preserves **only the manifest and log** as a temporary job artifact; it does not publish outcome data snapshots and never modifies the public app or registered model weights. Executing it a second time in the same local directory must use a new isolated workspace, not overwrite history.

## Verification and safeguards

The [shadow intake tests](../tests/test_shadow_historical_data_refresh.py) require: exact pinned source identity before Parquet read; a test that turns the source hash invalid and fails if Parquet is still opened; disjoint features and labels; stable game/play IDs; chronological splits; future-year rejection (2013, 2026, 2027, 2028); 2025 not falsely treated as an untouched test; a complete accounting of fake and overtime exclusions; distinct output table hashes; refusal to overwrite and refusal to write beneath tracked production data/model/report paths.

The paired older [cohort tests](../tests/test_historical_fourth_down_cohort.py) already demonstrate that changing post-play winner, observed action, yards, or EPA **cannot** change the pre-snap feature matrix. Every new source vintage would require a deliberate reviewed admission rule and an entirely separate experiment; simply changing the years passed to nflreadpy is prohibited.

### What is still needed?

1. Wait for the **independent 2026 Step 66/83 tests** to finish under their existing once-only automatic workflow. Do not run them, adjust their gates, or inspect their holdout outcomes early.
2. Before 2027 Week 1, acquire independently timestamped real *before-snap* state, reliable physical roof/stadium and halftime receiver, a separately administered witness, and a signed pre-season freeze if the user wants to make **prospective coaching calibration** claims. nflverse is retrospective and cannot replace these.
3. For a later **next-generation model** only, propose and freeze a new season-intake policy **after** protected studies and their public holdouts have been respected. Any new version must choose a new development period and truly future evaluation split before training or tuning. The old 2025 benchmark has already been examined and cannot be advertised as a fresh independent test.
4. Model improvement candidates for under-supported own-territory GO, site sensitivity and other weak spots remain *exploratory* until independently evaluated. No positive simulated recommendation edge proves a causal FG-versus-GO win benefit.

**Outcome:** a deterministic, contamination-resistant historical **input staging pipeline**, not a completed new model, a live automated recommendation or evidence that the original policy is prospectively validated.
