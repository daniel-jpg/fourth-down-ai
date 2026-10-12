"""Offline 2014-2024 weather-ablation study for GO and ordinary PUNT.

Targets:
* GO (normal RUN or PASS): actual logged conversion, Brier and log loss
* PUNT (ordinary, opponent possession, verified next state): receiving
  offense's next-start yardline_100, MAE and RMSE

The baseline and every candidate train and validate on the EXACT same
outdoor/open-roof cases with valid game-level wind/temp; no 2025+
outcomes, no nflreadpy network load, no production model changes.
TRAIN 2014-2022; VALIDATION 2023-2024. The simpler baselines are
research comparators, NOT claimed to equal the frozen production model.

A complete-game weather text may mention rain or snow without proof
that precipitation occurred at the snap. Head/tail/crosswinds cannot
be trained without trustworthy wind direction relative to play.
Weather features are NEVER used to issue official recommendations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .weather_pbp_signal import (
    audit_direction_availability,
    classify_weather_description,
    text_features,
)

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / "data/fourth_downs_labeled.parquet"
PUNT_AUDIT = ROOT / "data/punt_outcome_audit.parquet"
YEARS_TRAIN = (2014, 2022)
YEARS_VALIDATE = (2023, 2024)
VALID_YEARS = frozenset(range(2014, 2025))
ROOFS = frozenset(("outdoors", "open"))
GO_ACTIONS = frozenset(("NORMAL_GO_PASS", "NORMAL_GO_RUN"))
RAW_COLUMNS = (
    "season", "game_id", "play_id", "action", "execution_status",
    "roof", "temp", "wind", "weather", "home_team", "posteam", "location",
    "ydstogo", "yardline_100", "goal_to_go", "qtr",
    "game_seconds_remaining", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "converted",
)
PUNT_COLUMNS = (
    "game_id", "play_id", "season", "punt_branch",
    "transition_class", "state_usable", "state_yardline_100",
)
BASE_GENERAL = (
    "ydstogo", "yardline_100", "goal_to_go", "qtr",
    "game_seconds_remaining", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "is_home_offense", "is_neutral_site", "roof_open",
)
WEATHER_NUMERIC = ("temperature_20f", "wind_10mph")
PRECIP_TEXT = (
    "precip_rain_word", "precip_snow_word",
    "reported_dry_word", "weather_text_unspecified",
)
TEXT_MIN_TRAIN = 25
TEXT_MIN_VAL = 10
MIN_TRAIN = {"GO": 500, "PUNT": 400}
MIN_VAL = {"GO": 90, "PUNT": 75}


def _schema_missing(path: Path, needed: tuple[str, ...]) -> list[str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or symlinked pinned development parquet")
    schema = pl.scan_parquet(path).collect_schema().names()
    return sorted(set(needed).difference(schema))


def _label_source() -> tuple[pd.DataFrame, dict]:
    missing = _schema_missing(LABELED, RAW_COLUMNS)
    if missing:
        raise ValueError(f"Missing pinned historical PBP fields: {missing}")
    actual = pl.scan_parquet(LABELED).collect_schema().names()
    directions = audit_direction_availability(actual)
    # Critical: this temporal filter runs BEFORE collecting rows in pandas.
    raw = (
        pl.scan_parquet(LABELED)
        .filter(pl.col("season").is_between(2014, 2024))
        .filter(pl.col("action").is_in(list(GO_ACTIONS | {"PUNT"})))
        .filter(pl.col("roof").is_in(list(ROOFS)))
        .select(list(RAW_COLUMNS))
        .collect()
        .to_pandas()
    )
    if raw.empty or not raw["season"].between(2014, 2024).all():
        raise ValueError("Outdoor GO/PUNT research cohort invalid")
    if raw.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate historical fourth-down game/play identifiers")
    return raw, directions


def _punt_targets() -> pd.DataFrame:
    missing = _schema_missing(PUNT_AUDIT, PUNT_COLUMNS)
    if missing:
        raise ValueError(f"Missing verified punt audit columns: {missing}")
    # Source target is the original script 39 next-opponent-offensive-state
    # reconstruction; no novel post-snap label reconstruction is attempted.
    punt = (
        pl.scan_parquet(PUNT_AUDIT)
        .filter(pl.col("season").is_between(2014, 2024))
        .filter(pl.col("punt_branch") == "REGULAR")
        .filter(pl.col("transition_class") == "opponent_ball_no_score")
        .filter(pl.col("state_usable") == True)
        .select(list(PUNT_COLUMNS))
        .collect()
        .to_pandas()
    )
    if punt.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate original verified punt outcome")
    return punt


def _prepare(data: pd.DataFrame, *, action: str) -> tuple[pd.DataFrame, dict]:
    if action not in ("GO", "PUNT"):
        raise ValueError("Unknown weather research action")
    data = data.copy()
    if data.empty:
        raise ValueError(f"{action} cohort empty before weather quality checks")
    if not set(data["season"]).issubset(VALID_YEARS):
        raise PermissionError("Protected or future season in weather study")
    data["is_pass"] = data["action"].eq("NORMAL_GO_PASS").astype(int)
    data["roof_open"] = data["roof"].eq("open").astype(int)
    data["is_home_offense"] = data["posteam"].eq(data["home_team"]).astype(int)
    data["is_neutral_site"] = data["location"].eq("Neutral").astype(int)
    for col in (*BASE_GENERAL, "temp", "wind"):
        if col in ("is_home_offense", "is_neutral_site", "roof_open"):
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["temperature_20f"] = (data["temp"] - 65.0) / 20.0
    data["wind_10mph"] = (data["wind"] - 5.0) / 10.0
    data["weather_text_label"] = data["weather"].map(
        classify_weather_description
    )
    for col in PRECIP_TEXT:
        data[col] = data["weather_text_label"].map(
            lambda label, k=col: text_features(label)[k]
        ).astype(int)
    if action == "GO":
        target = "converted"
        data[target] = pd.to_numeric(data[target], errors="coerce")
        valid_target = data[target].isin([0, 1])
        baseline = (*BASE_GENERAL, "is_pass")
    else:
        target = "state_yardline_100"
        data[target] = pd.to_numeric(data[target], errors="coerce")
        valid_target = data[target].between(0, 100)
        baseline = BASE_GENERAL
    mask = (
        valid_target
        & data["temp"].between(-25, 115)
        & data["wind"].between(0, 40)
        & data[list(baseline)].notna().all(axis=1)
    )
    kept = data.loc[mask].copy()
    if (kept.empty or kept.duplicated(["game_id", "play_id"]).any()
            or kept[list(baseline + WEATHER_NUMERIC + PRECIP_TEXT)].isna().any().any()):
        raise ValueError(f"Invalid or empty unique {action} weather cohort")
    train = kept["season"].between(2014, 2022)
    validation = kept["season"].between(2023, 2024)
    if (int(train.sum()) < MIN_TRAIN[action]
            or int(validation.sum()) < MIN_VAL[action]
            or not (train | validation).all()):
        raise ValueError(f"Insufficient {action} weather-covered rows")
    if action == "GO":
        if (kept.loc[train, "converted"].nunique() < 2
                or kept.loc[validation, "converted"].nunique() < 2):
            raise ValueError("GO conversion target lacks both classes")
    coverage = {
        "action": action,
        "eligible_outdoor_rows": int(len(data)),
        "weather_covered_complete_rows": int(len(kept)),
        "excluded_weather_or_missing_pregame_features": int(len(data) - len(kept)),
        "weather_complete_fraction": float(len(kept) / len(data)),
        "training_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "training_games": int(kept.loc[train, "game_id"].nunique()),
        "validation_games": int(kept.loc[validation, "game_id"].nunique()),
        "train_precip_text_counts": dict(Counter(
            kept.loc[train, "weather_text_label"]
        )),
        "validation_precip_text_counts": dict(Counter(
            kept.loc[validation, "weather_text_label"]
        )),
        "text_labels_are_observed_at_snap": False,
        "unknown_precip_misclassified_as_dry": False,
    }
    return kept, coverage


def _estimator(action: str):
    if action == "GO":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        )
    return HistGradientBoostingRegressor(
        learning_rate=.05, max_iter=240,
        max_leaf_nodes=25, min_samples_leaf=25,
        l2_regularization=2.0, random_state=42,
    )


def _metrics(action: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if not np.isfinite(pred).all():
        raise ValueError("Research prediction is non-finite")
    if action == "GO":
        if (pred < 0).any() or (pred > 1).any():
            raise ValueError("GO predictions must be probabilities")
        return {
            "brier": float(brier_score_loss(y, pred)),
            "log_loss": float(log_loss(y, pred, labels=[0, 1])),
            "actual_conversion_rate": float(np.mean(y)),
            "mean_prediction": float(np.mean(pred)),
        }
    return {
        "mae_yards": float(mean_absolute_error(y, pred)),
        "rmse_yards": float(np.sqrt(mean_squared_error(y, pred))),
        "actual_mean_receiving_offense_yardline_100": float(np.mean(y)),
        "predicted_mean_receiving_offense_yardline_100": float(np.mean(pred)),
    }


def _gain(action: str, y: np.ndarray, reference: np.ndarray,
          candidate: np.ndarray) -> float:
    if action == "GO":
        return float(
            brier_score_loss(y, reference)
            - brier_score_loss(y, candidate)
        )
    return float(
        mean_absolute_error(y, reference)
        - mean_absolute_error(y, candidate)
    )


def _cluster_ci(action: str, y: np.ndarray, first: np.ndarray,
                second: np.ndarray, games: np.ndarray) -> list[float]:
    unique, inverse = np.unique(games.astype(str), return_inverse=True)
    if len(unique) < 20:
        raise ValueError("Insufficient validation games for clustered interval")
    by_game = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(42)
    improvements = []
    for _ in range(350):
        selected = rng.integers(len(unique), size=len(unique))
        positions = np.concatenate([by_game[i] for i in selected])
        improvements.append(_gain(
            action, y[positions], first[positions], second[positions]
        ))
    return [float(x) for x in np.quantile(improvements, [.025, .975])]


def _precip_support(coverage: dict) -> dict:
    def count_mentions(hist: dict) -> int:
        return int(
            hist.get("RAIN_MENTION", 0)
            + hist.get("SNOW_ICE_MENTION", 0)
            + hist.get("MIXED_MENTION", 0)
        )
    train_mentions = count_mentions(coverage["train_precip_text_counts"])
    val_mentions = count_mentions(coverage["validation_precip_text_counts"])
    enough = (
        train_mentions >= TEXT_MIN_TRAIN
        and val_mentions >= TEXT_MIN_VAL
    )
    return {
        "training_rain_or_snow_word_plays": train_mentions,
        "validation_rain_or_snow_word_plays": val_mentions,
        "minimum_training_mentions": TEXT_MIN_TRAIN,
        "minimum_validation_mentions": TEXT_MIN_VAL,
        "text_fit_eligible_for_research": bool(enough),
        "precipitation_at_exact_snap_confirmed": False,
    }


def _compare(action: str, frame: pd.DataFrame, coverage: dict) -> dict:
    train = frame.loc[frame["season"].between(2014, 2022)].copy()
    val = frame.loc[frame["season"].between(2023, 2024)].copy()
    target = "converted" if action == "GO" else "state_yardline_100"
    features = (*BASE_GENERAL, "is_pass") if action == "GO" else BASE_GENERAL
    y_train = train[target].to_numpy(dtype=float)
    y_val = val[target].to_numpy(dtype=float)
    predictions = {}
    models = [
        ("preplay_baseline", features),
        ("weather_temperature_wind", features + WEATHER_NUMERIC),
    ]
    precip_support = _precip_support(coverage)
    if precip_support["text_fit_eligible_for_research"]:
        models.append((
            "weather_plus_precipitation_text",
            features + WEATHER_NUMERIC + PRECIP_TEXT,
        ))
    measurements = {}
    for name, cols in models:
        estimator = _estimator(action)
        estimator.fit(train[list(cols)].to_numpy(dtype=float), y_train)
        if action == "GO":
            pred = estimator.predict_proba(
                val[list(cols)].to_numpy(dtype=float)
            )[:, 1]
        else:
            pred = np.clip(
                estimator.predict(val[list(cols)].to_numpy(dtype=float)),
                0, 100,
            )
        predictions[name] = np.asarray(pred, dtype=float)
        measurements[name] = _metrics(action, y_val, predictions[name])
    baseline = predictions["preplay_baseline"]
    deltas = {}
    for name in predictions:
        if name == "preplay_baseline":
            continue
        delta = _gain(action, y_val, baseline, predictions[name])
        lower, upper = _cluster_ci(
            action, y_val, baseline, predictions[name],
            val["game_id"].to_numpy(),
        )
        deltas[name] = {
            "improvement_baseline_minus_candidate": delta,
            "game_cluster_95pct_ci": [lower, upper],
            "improvement_sign": (
                "BETTER" if delta > 0 else
                "WORSE" if delta < 0 else "EQUAL"
            ),
            "validation_ci_strictly_above_zero": bool(lower > 0),
        }
    return {
        "action": action,
        "target": (
            "FOURTH_DOWN_ACTUAL_CONVERSION"
            if action == "GO" else
            "NEXT_OPPONENT_OFFENSE_YARDLINE_100_ON_REGULAR_PUNT"
        ),
        "preplay_baseline_is_original_frozen_production_model": False,
        "all_candidates_use_identical_validation_rows": True,
        "coverage": coverage,
        "precipitation_text_support": precip_support,
        "model_validation": measurements,
        "gains_against_preplay_baseline": deltas,
        "weather_effect_on_go_or_punt_calibrated_for_official_policy": False,
        "weather_text_is_at_snap_provenance": False,
        "reported_precipitation_is_a_text_mention_not_a_sensor": True,
        "no_causal_coach_action_conclusion": True,
    }


def run_historical_weather_study() -> dict:
    raw, direction = _label_source()
    go = raw.loc[raw["action"].isin(GO_ACTIONS)].copy()
    go = go.loc[go["execution_status"].eq("NORMAL")].copy()
    go_frame, go_coverage = _prepare(go, action="GO")

    targets = _punt_targets()
    punt_raw = raw.loc[raw["action"].eq("PUNT")].copy()
    punt = punt_raw.merge(
        targets[list(PUNT_COLUMNS)],
        on=["game_id", "play_id", "season"],
        how="inner", validate="one_to_one",
    )
    if punt.empty:
        raise ValueError("No exact verified regular punt/terminal state joins")
    punt_frame, punt_coverage = _prepare(punt, action="PUNT")
    result = {
        "status": "2014_2024_HISTORICAL_WEATHER_GO_PUNT_DEVELOPMENT_ONLY",
        "train_seasons": "2014-2022",
        "validation_seasons": "2023-2024",
        "year_filter_applied_before_loading_full_outcomes": True,
        "study_inputs": "NFLVERSE_GAME_LEVEL_TEMPERATURE_WIND_WEATHER_TEXT",
        "source_files": [
            "data/fourth_downs_labeled.parquet",
            "data/punt_outcome_audit.parquet",
        ],
        "wind_direction_provenance": direction,
        "go": _compare("GO", go_frame, go_coverage),
        "punt": _compare("PUNT", punt_frame, punt_coverage),
        "rain_snow_intensity_fitted": False,
        "weather_at_exact_decision_time_verified": False,
        "2025_holdout_rows_examined": False,
        "protected_2026_rows_examined": False,
        "2027_outcomes_examined": False,
        "frozen_go_and_punt_models_modified": False,
        "official_recommendation_changed": False,
        "production_promotion_authorized": False,
        "interpretation": (
            "Matched-row, out-of-time predictive ablations on logged "
            "decisions only. GO run/pass selection is confounded; punt "
            "target excludes scoring, broken and unsupported branches. "
            "Weather descriptions are game-level and directional wind "
            "relative to the play is unavailable. No official decision "
            "policy or coaching causal effect can be inferred."
        ),
    }
    # Prevent "NaN" from being serialized as valid statistical evidence.
    json.dumps(result, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.development_only:
        parser.error("Only --development-only is supported; real holdouts denied")
    result = run_historical_weather_study()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fd:
        json.dump(result, fd, sort_keys=True, indent=2, allow_nan=False)
        fd.write("\n")
    print(json.dumps({
        "status": result["status"],
        "go": result["go"]["gains_against_preplay_baseline"],
        "go_coverage": result["go"]["coverage"],
        "punt": result["punt"]["gains_against_preplay_baseline"],
        "punt_coverage": result["punt"]["coverage"],
        "wind_direction": result["wind_direction_provenance"],
        "production_promotion_authorized": False,
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
