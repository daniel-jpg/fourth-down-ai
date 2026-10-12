"""Build a DEVELOPMENT-ONLY 2027 policy-study nuisance snapshot.

Loads ONLY the checked-in labeled 2014-25 historic PBP parquet. Never uses
network, nflreadpy, future 2026+ outcomes, protected kickoff Steps 66/83,
2027 prospective scoring, or model/policy promotion.

1. Extract separate audit frames pre_snap / logged coach action / final label.
2. Fit e(a|X), mu(a,X) on 2014-22 for 2023-24 exploration.
3. Freeze separate final e/mu fits on 2014-25 in artifacts/, not models/.
4. Record SHA-256 of source, features, algorithm and saved artifacts.

The outcome is repeated from the historical PBP 'result' column; independent
external game-score verification is NOT complete. Do not call this an
independently verified 2027 production nuisance model yet.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from src.fourth_down_policy_study import POLICY_COMMIT
from src.historical_coach_nuisances import (
    ACTION_ORDER, DEV_TRAIN_END, DEV_VALIDATION_YEARS, FINAL_FIT_END,
    INTERACTIONS, LOGISTIC_C, MAX_ITER, RANDOM_STATE,
    development_metrics, fit_nuisances,
)
from src.historical_fourth_down_cohort import (
    FEATURE_COLUMNS, SOURCE_COLUMNS,
    extract_historical_cohort, fingerprint_frame, join_training_labels,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/fourth_downs_labeled.parquet"
OUTDIR = ROOT / "artifacts"
FROZEN_INPUT_GIT_BLOB = "d7230df63cad1c24579ee40cabaacd0b330e4784"
EXTRACTOR = ROOT / "src/historical_fourth_down_cohort.py"
TRAINER = ROOT / "src/historical_coach_nuisances.py"
PROP_PATH = OUTDIR / "coach_propensity_2014_2025_development.joblib"
REWARD_PATH = OUTDIR / "action_win_reward_2014_2025_development.joblib"
REPORT_PATH = OUTDIR / "historical_nuisance_development_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def main() -> None:
    if not INPUT.is_file() or git_blob_sha(INPUT) != FROZEN_INPUT_GIT_BLOB:
        raise RuntimeError("Historical source parquet differs from pinned Git blob")
    if DEV_TRAIN_END != 2022 or DEV_VALIDATION_YEARS != (2023, 2024) or FINAL_FIT_END != 2025:
        raise RuntimeError("Nuisance chrono split differs from approved 2014-25 historical plan")
    # Column projection prevents importing play descriptions, yards_gained,
    # conversion flags or other post-snap outcomes into pre-snap frame.
    raw = pd.read_parquet(INPUT, columns=SOURCE_COLUMNS)
    cohort = extract_historical_cohort(raw)
    labeled = join_training_labels(cohort)
    train = labeled.loc[labeled["season"].between(2014, 2022)].copy()
    eval_dev = labeled.loc[labeled["season"].isin(DEV_VALIDATION_YEARS)].copy()
    if train.empty or eval_dev.empty:
        raise RuntimeError("Missing separate train 2014-22 / development 2023-24 seasons")
    if set(train["game_id"]) & set(eval_dev["game_id"]):
        raise RuntimeError("Game leakage between training and development seasons")
    print("Historical candidate census:", cohort["audit"], flush=True)
    print(f"Fitting 2014-22 propensity/reward on {len(train)} decisions; "
          f"checking 2023-24 on {len(eval_dev)} decisions.", flush=True)
    developmental = fit_nuisances(train, end_year=DEV_TRAIN_END)
    benchmark = development_metrics(developmental, eval_dev)
    if benchmark["dev_distinct_games"] < 100:
        raise RuntimeError("Development evaluation needs >=100 distinct games")
    print("OUT-OF-TIME DEVELOPMENT METRICS", benchmark, flush=True)

    # FINAL nuisance fit is separate from the above development fit. It uses
    # 2025 outcomes solely for its historical training (per Issue #8), not
    # for tuning or claiming a pristine validation.
    print("Fitting final unpromoted historical nuisance model 2014-2025...",
          flush=True)
    final = fit_nuisances(labeled, end_year=FINAL_FIT_END)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final.propensity, PROP_PATH)
    joblib.dump(final.reward, REWARD_PATH)

    source_sha = sha256(INPUT)
    labels = cohort["final_labels"]
    actions = cohort["observed_actions"]
    audit = {
        "research_status": "HISTORICAL DEV ONLY; NOT APPROVED FOR PROSPECTIVE 2027 OPE",
        "frozen_policy_commit": POLICY_COMMIT,
        "historical_input_git_blob": FROZEN_INPUT_GIT_BLOB,
        "historical_input_sha256": source_sha,
        "pre_snap_sha256": fingerprint_frame(cohort["pre_snap"]),
        "historical_observed_actions_sha256": fingerprint_frame(actions),
        "historical_final_labels_sha256": fingerprint_frame(labels),
        "historical_exclusions_sha256": fingerprint_frame(cohort["excluded"]),
        "extractor_code_sha256": sha256(EXTRACTOR),
        "trainer_code_sha256": sha256(TRAINER),
        "fit_train_seasons": "2014-2022",
        "exploratory_development_seasons": "2023-2024",
        "historical_final_fit_seasons": "2014-2025",
        "seasons_2026_2027_inspected": False,
        "feature_policy": "explicit_pre_snap_whitelist_no_observed_action_or_final_outcome",
        "feature_names": list(FEATURE_COLUMNS),
        "action_names": list(ACTION_ORDER),
        "reward_action_interactions": list(INTERACTIONS),
        "logistic_regularization_C": LOGISTIC_C,
        "max_iter": MAX_ITER,
        "random_state": RANDOM_STATE,
        "historical_source_census": cohort["audit"],
        "fit_action_counts": dict(sorted(Counter(labeled["observed_action"]).items())),
        "training_decisions": int(len(train)),
        "development_decisions": int(len(eval_dev)),
        "final_fit_decisions": int(len(labeled)),
        "final_fit_distinct_games": int(labeled["game_id"].nunique()),
        "out_of_time_development_metrics": benchmark,
        "artifacts": {
            "propensity": {
                "filename": PROP_PATH.name, "sha256": sha256(PROP_PATH),
            },
            "conditional_win_reward": {
                "filename": REWARD_PATH.name, "sha256": sha256(REWARD_PATH),
            },
        },
        "limitations": [
            "No independent external final-score/game-type cross-source verification yet",
            "2023-24 diagnostics are historical development, not untouched validation",
            "2025 used in final historical nuisance fit only; already-explored year",
            "Observational treatment choice may hide wind, injuries, kicker quality, personnel",
            "Low propensity and rare GO short-yardage actions can violate positivity",
            "Model fit may predict implausibly for actions outside historical support",
            "Nuisance accuracy does not prove a causal action-policy improvement",
            "No candidate or stable model weights deployed; no prospective run permitted",
        ],
    }
    REPORT_PATH.write_text(json.dumps(
        audit, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n")
    # Persist as separate artifacts for independent provenance review,
    # never silently add them into the public app or frozen 2026 gates.
    cohort["pre_snap"].to_parquet(OUTDIR / "historical_2014_2025_pre_snap.parquet", index=False)
    cohort["observed_actions"].to_parquet(OUTDIR / "historical_2014_2025_actions.parquet", index=False)
    cohort["final_labels"].to_parquet(OUTDIR / "historical_2014_2025_final_labels.parquet", index=False)
    cohort["excluded"].to_parquet(OUTDIR / "historical_2014_2025_exclusions.parquet", index=False)
    print("Saved historical-source audit and UNPROMOTED model artifacts:", REPORT_PATH, flush=True)
    print("No 2026/2027 outcome access; no 2026 Step 66/83 workflow activity.", flush=True)


if __name__ == "__main__":
    main()
