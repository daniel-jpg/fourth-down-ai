"""Fit an experimental OUTDOOR field-goal temperature/wind adjustment.

Strict 2014-2022 training and 2023-2024 validation ONLY. No 2025
held-out outcomes, no 2026/2027 sources, no change to the frozen FG bundle.

The conditional clean-kick log-odds coefficient is used as a research
sensitivity on top of the frozen distance/roof/kicker model. This is
not a calibrated, policy-changing production model. nflverse temp/wind
are stadium game-level weather estimates, not exact fourth-down-time
measurements. A categorical precipitation condition is NOT fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/fourth_downs_labeled.parquet"
TRAIN_LAST = 2022
VALIDATION_YEARS = (2023, 2024)
MODEL_COLUMNS = (
    "fg_distance", "fg_distance_sq", "roof_open",
    "temperature_20f", "wind_10mph",
)
BASE_COLUMNS = MODEL_COLUMNS[:3]
MIN_TRAIN = 900
MIN_VALIDATION = 200


def load_development_fg() -> pd.DataFrame:
    """Lazy pushdown rejects 2025 records BEFORE materialization."""
    fields = (
        "season", "game_id", "play_id", "yardline_100", "roof",
        "temp", "wind", "action", "execution_status", "field_goal_result",
    )
    lazy = (
        pl.scan_parquet(ARCHIVE)
        .filter(pl.col("season").is_between(2014, 2024))
        .filter(pl.col("action") == "FIELD_GOAL")
        .filter(pl.col("execution_status") == "NORMAL")
        .filter(pl.col("roof").is_in(["outdoors", "open"]))
        .select(list(fields))
    )
    frame = lazy.collect().to_pandas()
    if frame.empty or (frame["season"] >= 2025).any():
        raise ValueError("Weather development source empty or includes held-out season")
    frame["outcome"] = frame["field_goal_result"].fillna("").str.lower()
    frame = frame.loc[frame["outcome"].isin(["made", "missed"])].copy()
    frame["temp"] = pd.to_numeric(frame["temp"], errors="coerce")
    frame["wind"] = pd.to_numeric(frame["wind"], errors="coerce")
    frame["yardline_100"] = pd.to_numeric(frame["yardline_100"], errors="coerce")
    # Do not extrapolate a model over missing/outlier game-level weather.
    valid = (
        frame["temp"].between(-25, 115)
        & frame["wind"].between(0, 40)
        & frame["yardline_100"].between(1, 62)
    )
    frame = frame.loc[valid].copy()
    frame["made"] = frame["outcome"].eq("made").astype(int)
    frame["fg_distance"] = frame["yardline_100"] + 18.
    frame["fg_distance_sq"] = frame["fg_distance"] ** 2
    frame["roof_open"] = frame["roof"].eq("open").astype(int)
    frame["temperature_20f"] = (frame["temp"] - 65.0) / 20.0
    frame["wind_10mph"] = (frame["wind"] - 5.0) / 10.0
    if frame.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate NFL FG record")
    if (frame["season"].min() < 2014 or frame["season"].max() > 2024
            or not np.isfinite(frame[list(MODEL_COLUMNS)].to_numpy(dtype=float)).all()):
        raise ValueError("Malformed weather development evidence")
    return frame.sort_values(["season", "game_id", "play_id"]).reset_index(drop=True)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def train_candidate() -> dict:
    df = load_development_fg()
    train = df.loc[df["season"].le(TRAIN_LAST)]
    validation = df.loc[df["season"].isin(VALIDATION_YEARS)]
    if (len(train) < MIN_TRAIN or len(validation) < MIN_VALIDATION
            or train["made"].nunique() != 2):
        raise ValueError("Too little weather-covered historical FG data")
    fitted = {}
    predictions = {}
    for label, cols in (
        ("baseline_distance_roof", BASE_COLUMNS),
        ("candidate_weather", MODEL_COLUMNS),
    ):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        )
        clf.fit(train[list(cols)], train["made"])
        fitted[label] = clf
        predictions[label] = clf.predict_proba(
            validation[list(cols)]
        )[:, 1]
    y = validation["made"].to_numpy(dtype=int)
    baseline = _metrics(y, predictions["baseline_distance_roof"])
    candidate = _metrics(y, predictions["candidate_weather"])
    candidate_model = fitted["candidate_weather"]
    scaler, logistic = candidate_model.steps[0][1], candidate_model.steps[1][1]
    effects = dict(zip(
        MODEL_COLUMNS,
        [float(coef/scale) for coef, scale in
         zip(logistic.coef_[0], scaler.scale_)],
    ))
    # This is a weather log-odds contrast holding distance, roof, kicker
    # and every other frozen-engine variable fixed.
    coefs = {
        "per_20f_log_odds": effects["temperature_20f"],
        "per_10mph_log_odds": effects["wind_10mph"],
    }
    games = validation["game_id"].to_numpy()
    groups = [np.flatnonzero(games == game) for game in np.unique(games)]
    rng = np.random.default_rng(42)
    improvements = []
    for _ in range(300):
        draw = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in draw])
        improvements.append(float(
            brier_score_loss(y[rows], predictions["baseline_distance_roof"][rows])
            - brier_score_loss(y[rows], predictions["candidate_weather"][rows])
        ))
    low, high = np.quantile(improvements, [.025, .975])
    brier_gain = baseline["brier"] - candidate["brier"]
    log_loss_gain = baseline["log_loss"] - candidate["log_loss"]
    # Keep a diagnostic even if the weather model isn't superior:
    # it must NEVER be automatically promoted to the official model.
    report = {
        "status": "HISTORICAL_DEVELOPMENT_WEATHER_RESEARCH_ONLY",
        "source_seasons": {"train": "2014-2022", "validation": "2023-2024"},
        "forecast_scope": "OUTDOOR_OR_OPEN_ROOF_CLEAN_FIELD_GOALS_ONLY",
        "source_weather": "nflverse stadium game-level temperature F / wind mph",
        "number_training": len(train),
        "number_validation": len(validation),
        "training_games": train["game_id"].nunique(),
        "validation_games": validation["game_id"].nunique(),
        "train_median_temperature_f": float(train["temp"].median()),
        "train_median_wind_mph": float(train["wind"].median()),
        "baseline_validation": baseline,
        "weather_candidate_validation": candidate,
        "brier_improvement_baseline_minus_weather": brier_gain,
        "log_loss_improvement_baseline_minus_weather": log_loss_gain,
        "brier_gain_95pct_game_cluster_ci": [float(low), float(high)],
        "effect_coefficients": coefs,
        "weather_coefficients_shrinkage_capped": True,
        "condition_rain_snow_fitted": False,
        "calibrated_official_decision_effect": False,
        "may_change_frozen_production": False,
        "2025_heldout_data_used": False,
        "2026_or_2027_data_used": False,
        "independent_live_forecast_issued": False,
        "interpretation": (
            "Game-level outdoor temperature/wind are exploratory inputs for "
            "an optional clean-field-goal sensitivity only. Do not infer "
            "causality or a validated effect on GO, PUNT, drive value or "
            "precipitation. Frozen official recommendations remain unchanged."
        ),
    }
    if not np.isfinite(list(coefs.values())).all():
        raise ValueError("Nonfinite temperature/wind coefficients")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.development_only:
        parser.error("Only --development-only, never future data")
    report = train_candidate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as fd:
        json.dump(report, fd, indent=2, sort_keys=True)
        fd.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
