"""Time-split GO/PUNT weather ablation without touching production models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import weather_go_punt_development as study


def invented_weather_plays(action: str, *, with_weather_text: bool) -> pd.DataFrame:
    records = []
    for i in range(750):
        season = 2018 if i < 600 else 2023
        wind = float(2 + (i * 7) % 31)
        temp = float(22 + (i * 11) % 81)
        description = (
            "Snow" if i % 11 == 0 else
            "Rain" if i % 5 == 0 else "Clear"
        ) if with_weather_text else "Cloudy"
        record = {
            "season": season, "game_id": f"synthetic_{season}_{i // 4}",
            "play_id": str(i),
            "action": (
                "NORMAL_GO_PASS" if i % 2 else "NORMAL_GO_RUN"
            ) if action == "GO" else "PUNT",
            "execution_status": "NORMAL",
            "roof": "outdoors",
            "temp": temp, "wind": wind, "weather": description,
            "home_team": "ABC", "posteam": "ABC",
            "location": "Home",
            "ydstogo": 1.0 + i % 10,
            "yardline_100": 12.0 + i % 70,
            "goal_to_go": 0.0, "qtr": i % 4 + 1,
            "game_seconds_remaining": 100.0 + (i * 43) % 3200,
            "score_differential": float(i % 25 - 12),
            "posteam_timeouts_remaining": float(i % 4),
            "defteam_timeouts_remaining": float((i+2) % 4),
            "converted": int((i * 17) % 100 <= 51 - (i % 8)),
            "state_yardline_100": float(
                min(99, max(1, 59.0 - (i % 70) * .3 + wind * .13))
            ),
        }
        records.append(record)
    return pd.DataFrame(records)


def test_temporal_train_val_and_same_row_baseline_weather_comparison():
    go, audit = study._prepare(
        invented_weather_plays("GO", with_weather_text=True),
        action="GO",
    )
    assert audit["training_rows"] == 600
    assert audit["validation_rows"] == 150
    assert audit["eligible_outdoor_rows"] == 750
    assert audit["weather_complete_fraction"] == 1
    result = study._compare("GO", go, audit)
    assert result["target"] == "FOURTH_DOWN_ACTUAL_CONVERSION"
    assert result["preplay_baseline_is_original_frozen_production_model"] is False
    assert result["all_candidates_use_identical_validation_rows"] is True
    assert set(result["model_validation"]) == {
        "preplay_baseline", "weather_temperature_wind",
        "weather_plus_precipitation_text",
    }
    assert result["precipitation_text_support"]["text_fit_eligible_for_research"]
    assert not result["precipitation_text_support"][
        "precipitation_at_exact_snap_confirmed"
    ]
    assert len(result["gains_against_preplay_baseline"][
        "weather_temperature_wind"
    ]["game_cluster_95pct_ci"]) == 2
    assert result["no_causal_coach_action_conclusion"]


def test_punt_target_has_distinct_error_metrics_and_no_policy_promotion():
    punt, audit = study._prepare(
        invented_weather_plays("PUNT", with_weather_text=True),
        action="PUNT",
    )
    result = study._compare("PUNT", punt, audit)
    assert result["target"] == (
        "NEXT_OPPONENT_OFFENSE_YARDLINE_100_ON_REGULAR_PUNT"
    )
    assert "mae_yards" in result["model_validation"]["preplay_baseline"]
    assert "rmse_yards" in result["model_validation"]["weather_temperature_wind"]
    assert "brier" not in result["model_validation"]["weather_temperature_wind"]
    assert result["weather_effect_on_go_or_punt_calibrated_for_official_policy"] is False
    assert result["reported_precipitation_is_a_text_mention_not_a_sensor"]


def test_insufficient_observed_precipitation_labels_skip_text_model():
    go, coverage = study._prepare(
        invented_weather_plays("GO", with_weather_text=False),
        action="GO",
    )
    result = study._compare("GO", go, coverage)
    assert not result["precipitation_text_support"]["text_fit_eligible_for_research"]
    assert set(result["model_validation"]) == {
        "preplay_baseline", "weather_temperature_wind",
    }


@pytest.mark.parametrize("year", [2025, 2026, 2027, 2030])
def test_protected_future_year_rejected_without_model_fit(year):
    data = invented_weather_plays("GO", with_weather_text=True)
    data.loc[0, "season"] = year
    with pytest.raises(PermissionError, match="Protected or future"):
        study._prepare(data, action="GO")


def test_missing_weather_counts_abstention_never_imputes_dry():
    data = invented_weather_plays("GO", with_weather_text=True)
    data.loc[0, "temp"] = np.nan
    data.loc[1, "wind"] = np.nan
    data.loc[2, "weather"] = None
    kept, audit = study._prepare(data, action="GO")
    assert audit["eligible_outdoor_rows"] == 750
    assert audit["weather_covered_complete_rows"] == 748
    assert audit["excluded_weather_or_missing_pregame_features"] == 2
    missing = kept.loc[kept["weather"].isna()]
    assert len(missing) == 1
    assert missing.iloc[0]["weather_text_label"] == "UNSPECIFIED"
    assert missing.iloc[0]["reported_dry_word"] == 0


def test_no_production_or_future_season_downloads_in_candidate_source():
    source = (Path(__file__).resolve().parents[1]
              / "src/weather_go_punt_development.py").read_text()
    assert "nfl.load_pbp(" not in source
    assert "nflreadpy.load_pbp(" not in source
    assert "joblib.dump(" not in source
    assert "load_pbp([2025" not in source
    assert 'filter(pl.col("season").is_between(2014, 2024))' in source
    assert "production_promotion_authorized" in source
    assert "weather_at_exact_decision_time_verified" in source


def test_precipitation_research_threshold_is_conservative():
    insufficient = {
        "train_precip_text_counts": {
            "RAIN_MENTION": 20, "SNOW_ICE_MENTION": 3,
        },
        "validation_precip_text_counts": {
            "RAIN_MENTION": 9,
        },
    }
    result = study._precip_support(insufficient)
    assert not result["text_fit_eligible_for_research"]
    assert result["precipitation_at_exact_snap_confirmed"] is False
