"""Execute the frozen-vs-field-position GO conversion audit (development only).

Run from repository root: PYTHONPATH=. python scripts/audit_go_field_position_challenger.py
Reads ONLY 2014-2024 rows of the checked-in historical data. Never changes
production weights, public recommendations, or 2026/2027 validation gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.field_position_go_research import (
    CHALLENGER_FEATURES, DATA_FEATURES, FROZEN_FEATURES, NORMAL,
    adjusted_run_probability, benchmark, extra_go_features,
    split_dev, support_counts, train_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data/fourth_down_modeling_split.parquet"
FROZEN = ROOT / "models/fake_conversion_model.joblib"
OUTDIR = ROOT / "artifacts"
EXPECTED_GIT_BLOBS = {
    PARQUET: "fa1d64f187810c08edc9a113051c8b633e2e9168",
    FROZEN: "d4f2dd5037209d263c20cf28bca50d70138245d4",
}
# Fixed before reviewing development-validation outputs.
SUBGROUPS = (
    "ALL_GO", "NORMAL_GO", "PASS", "RUN", "FOURTH_AND_2_OR_3",
    "OWN_1_10_SHORT", "OWN_1_20_SHORT", "OWN_1_35_SHORT",
    "Q4_NORMAL_GO", "VALIDATION_2023", "VALIDATION_2024",
)


def git_blob_hash(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def assert_frozen_blobs() -> None:
    for path, expected in EXPECTED_GIT_BLOBS.items():
        if not path.is_file() or git_blob_hash(path) != expected:
            raise RuntimeError(f"Frozen data/model blob mismatch: {path.name}")


def cohort_masks(valid: pd.DataFrame) -> dict[str, np.ndarray]:
    normal = valid["action"].isin(NORMAL).to_numpy(bool)
    pas = valid["action"].eq("NORMAL_GO_PASS").to_numpy(bool)
    run = valid["action"].eq("NORMAL_GO_RUN").to_numpy(bool)
    two_three = valid["ydstogo"].between(1.5, 3.5).to_numpy(bool)
    yard = valid["yardline_100"].to_numpy(float)
    season = valid["season"].to_numpy(int)
    qtr = valid["qtr"].to_numpy(int)
    return {
        "ALL_GO": np.ones(len(valid), dtype=bool),
        "NORMAL_GO": normal,
        "PASS": pas,
        "RUN": run,
        "FOURTH_AND_2_OR_3": normal & two_three,
        "OWN_1_10_SHORT": normal & two_three & (yard >= 90),
        "OWN_1_20_SHORT": normal & two_three & (yard >= 80),
        "OWN_1_35_SHORT": normal & two_three & (yard >= 65),
        "Q4_NORMAL_GO": normal & (qtr == 4),
        "VALIDATION_2023": season == 2023,
        "VALIDATION_2024": season == 2024,
    }


def main() -> None:
    assert_frozen_blobs()
    # Parquet predicate pushdown: do not load, inspect or score 2025 rows.
    frame = pd.read_parquet(
        PARQUET,
        engine="pyarrow",
        filters=[("season", ">=", 2014), ("season", "<=", 2024)],
        columns=sorted(DATA_FEATURES),
    )
    train, valid = split_dev(frame)
    frozen = joblib.load(FROZEN)
    challenger = train_candidate(train)
    old = frozen.predict_proba(
        valid[list(FROZEN_FEATURES)].to_numpy(float)
    )[:, 1]
    new = challenger.predict_proba(
        valid[list(CHALLENGER_FEATURES)].to_numpy(float)
    )[:, 1]

    mask = cohort_masks(valid)
    assert set(mask) == set(SUBGROUPS)
    metrics = {}
    for key in SUBGROUPS:
        selected = mask[key]
        if int(selected.sum()) < 2:
            metrics[key] = {
                "decisions": int(selected.sum()), "status": "insufficient_support"
            }
            continue
        subset = valid.loc[selected].copy()
        scores = benchmark(
            subset, old[selected], new[selected], draws=1000, seed=2024,
        )
        scores["status"] = "descriptive_development_only"
        if scores["distinct_games"] < 20:
            scores["interpretation"] = "Few independent games; no stable subgroup claim."
        metrics[key] = scores

    # This state has no comparable 2014-22 training GO attempts at own 1-5.
    scenario = {}
    for action in ("NORMAL_GO_RUN", "NORMAL_GO_PASS"):
        features = extra_go_features({}, is_pass=action.endswith("_PASS"))
        b = float(frozen.predict_proba(
            np.asarray([[features[f] for f in FROZEN_FEATURES]], dtype=float)
        )[0, 1])
        c = float(challenger.predict_proba(
            np.asarray([[features[f] for f in CHALLENGER_FEATURES]], dtype=float)
        )[0, 1])
        scenario[action] = {
            "frozen_raw_conversion": b,
            "challenger_raw_conversion": c,
            "frozen_with_existing_run_adjustment": adjusted_run_probability(b, action, 3.),
            "challenger_with_same_run_adjustment": adjusted_run_probability(c, action, 3.),
            "status": "UNSUPPORTED_EXTRAPOLATION_NOT_VALIDATED",
        }

    support = support_counts(train, valid)
    if not any(x["territory"] == "OWN_1_TO_5"
               and x["period"] == "train_2014_2022"
               and x["fourth_and_2_or_3_count"] == 0 for x in support):
        raise RuntimeError("Own-5 support changed; reassess validation assumptions.")
    report = {
        "status": "RESEARCH ONLY; DO NOT PROMOTE",
        "source_seasons": "2014-2024 only",
        "training_seasons": "2014-2022",
        "exploratory_validation_seasons": "2023-2024",
        "2025_outcomes": "NOT READ",
        "2026_2027_outcomes": "NOT READ",
        "frozen_model_blob_sha": EXPECTED_GIT_BLOBS[FROZEN],
        "source_data_blob_sha": EXPECTED_GIT_BLOBS[PARQUET],
        "normal_run_existing_logit_shift": .2925,
        "candidate": "frozen logistic specification plus continuous yardline_100",
        "candidate_features": list(CHALLENGER_FEATURES),
        "n_train": len(train),
        "n_validate": len(valid),
        "historical_support": support,
        "model_metrics": metrics,
        "own_5_scenario": scenario,
        "promotable_for_own_5": False,
        "interpretation": (
            "Observed historical conversion prediction, not causal policy evaluation. "
            "Zero same-territory training examples at own 5 means an average "
            "Brier improvement cannot validate the own-5 fourth-down choice. "
            "Both models remain extrapolations. The app/frozen artifacts unchanged."
        ),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    target = OUTDIR / "go_field_position_challenger_development.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    artifact = OUTDIR / "go_field_position_challenger.joblib"
    joblib.dump(challenger, artifact)
    print("=== GO: frozen vs continuous field-position challenger ===", flush=True)
    print(f"Training rows: {len(train)}; validation rows: {len(valid)}", flush=True)
    for row in support:
        print("SUPPORT", row, flush=True)
    for name in SUBGROUPS:
        rec = metrics[name]
        print(f"{name}: {rec}", flush=True)
    for act, rates in scenario.items():
        print(f"OWN-5 {act}: {rates}", flush=True)
    print("Written development-only JSON and candidate artifact:", target, flush=True)
    print("NO production model update. NO protected future outcomes read.", flush=True)


if __name__ == "__main__":
    main()
