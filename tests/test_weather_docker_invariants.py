"""Additional real-frozen-engine weather research invariants.

These are separate from the browser test and intentionally don't let
unvalidated weather alter the official policy.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from pathlib import Path

import pytest

from src import weather_fg_sensitivity as wx

ENGINE_PATH = (
    Path(__file__).resolve().parents[1] / "src/42_build_decision_engine.py"
)


@pytest.fixture(scope="module")
def frozen():
    spec = importlib.util.spec_from_file_location("weather_qa_frozen", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def state(*, roof="outdoors"):
    return {
        "qtr": 4, "game_seconds_remaining": 458.0,
        "yardline_100": 30.0, "ydstogo": 3.0,
        "score_differential": -3.0,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "site": "HOME", "roof": roof,
    }


def observation(condition, direction):
    return {
        "roof": "outdoors", "temperature_f": 32,
        "wind_mph": 25, "conditions": condition,
        "wind_relative_direction": direction,
    }


def test_real_q4_precipitation_and_direction_are_descriptive_not_policy(frozen):
    original_callback = frozen.field_goal_probabilities
    original_recommend = frozen.recommend
    snow = wx.compare_weather_scenario(
        state(), weather=observation("Snow", "Headwind"),
        live_engine=frozen,
    )
    rain = wx.compare_weather_scenario(
        state(), weather=observation("Rain", "Tailwind"),
        live_engine=frozen,
    )
    assert snow["applicable"] is rain["applicable"] is True
    assert snow["applied_log_odds_change"] == rain["applied_log_odds_change"]
    for key in (
        "original_made_probability", "weather_made_probability",
        "weather_fg_win_probability", "weather_modeled_top_action",
    ):
        assert snow[key] == rain[key], (key, snow[key], rain[key])
    assert snow["weather_made_probability"] < snow["original_made_probability"]
    assert frozen.field_goal_probabilities is original_callback
    assert frozen.recommend is original_recommend
    assert original_recommend.__globals__["field_goal_probabilities"] is original_callback
    for item in (snow, rain):
        assert item["official_model_unchanged"] is True
        assert item["validation_shows_brier_improvement"] is False
        assert item["adjusted_rain_snow_numeric_effect"] is False
        assert item["wind_direction_measured"] is False
        assert item["go_and_punt_weather_adjusted"] is False
        assert item["real_2027_data_accessed"] is False


@pytest.mark.parametrize("temp", [-25, 0, 32, 65, 90, 115])
def test_extreme_valid_temperatures_cannot_inject_invalid_probability(temp):
    report = wx.load_weather_research()
    for wind in (0, 5, 25, 40):
        ctx = wx.validate_weather_input(
            roof="outdoors", temperature_f=temp,
            wind_mph=wind, conditions="Unknown",
        )
        shift = wx.weather_log_odds_contrast(ctx, report)
        assert math.isfinite(shift) and abs(shift) <= wx.MAX_LOG_ODDS_CHANGE
        out = wx.adjust_fg_probability_distribution(
            {"made": .70, "missed": .25, "blocked": .04, "broken": .01},
            log_odds_shift=shift,
        )
        assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in out.values())
        assert out["broken"] == .01


@pytest.mark.parametrize("roof", ["dome", "closed"])
def test_indoor_never_runs_engine_for_weather(frozen, roof):
    result = wx.compare_weather_scenario(
        state(roof=roof), weather={
            "roof": roof, "temperature_f": 0,
            "wind_mph": 40, "conditions": "Snow",
        }, live_engine=frozen,
    )
    assert result["applicable"] is False
    assert "Covered stadium" in result["reason"]
    assert result["official_model_unchanged"] is True
