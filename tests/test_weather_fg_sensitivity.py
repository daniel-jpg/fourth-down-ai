"""Weather input and frozen-engine-isolated FG what-if regression tests."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
from pathlib import Path

import pytest

from src import weather_fg_sensitivity as weather


def current_weather(*, temperature=65, wind=20, condition="Rain", roof="outdoors"):
    return weather.validate_weather_input(
        roof=roof, temperature_f=temperature,
        wind_mph=wind, conditions=condition,
    )


def test_training_report_explicitly_not_validated_for_production():
    report = weather.load_weather_research()
    assert report["number_training"] == 5705
    assert report["number_validation"] == 1186
    assert report["source_seasons"] == {
        "train": "2014-2022", "validation": "2023-2024"
    }
    assert report["effect_coefficients"]["per_10mph_log_odds"] < 0
    assert report["brier_improvement_baseline_minus_weather"] < 0
    assert report["validation_supports_official_weather_use"] is False
    assert report["2025_heldout_data_used"] is False
    assert report["2026_or_2027_data_used"] is False
    assert report["condition_rain_snow_fitted"] is False


def test_temperature_and_wind_and_conditions_are_independently_recorded():
    context = current_weather(temperature=22, wind=17, condition="Snow")
    assert context == {
        "roof": "outdoors", "temperature_f": 22.0,
        "wind_mph": 17.0, "conditions": "Snow",
        "scenario_applicable": True, "reason": None,
    }
    coeff = weather.load_weather_research()
    assert weather.weather_log_odds_contrast(
        current_weather(wind=5, condition="Snow"), coeff
    ) == 0
    with pytest.raises(ValueError, match="range"):
        current_weather(temperature=-60, wind=5)
    with pytest.raises(ValueError, match="range"):
        current_weather(temperature=65, wind=55)
    with pytest.raises(ValueError, match="range"):
        current_weather(temperature=float("nan"), wind=5)
    with pytest.raises(ValueError, match="condition"):
        current_weather(condition="Hurricane category 6")


def test_more_wind_reduces_what_if_make_but_no_precipitation_bonus():
    model = weather.load_weather_research()
    calm = weather.weather_log_odds_contrast(
        current_weather(wind=5, condition="Rain"), model,
    )
    strong = weather.weather_log_odds_contrast(
        current_weather(wind=25, condition="Clear / dry"), model,
    )
    assert calm == 0
    assert strong < 0
    rain = weather.weather_log_odds_contrast(
        current_weather(wind=25, condition="Rain"), model,
    )
    snow = weather.weather_log_odds_contrast(
        current_weather(wind=25, condition="Snow"), model,
    )
    assert rain == snow == strong


def test_dome_and_retractable_closed_roofs_do_not_get_outdoor_weather_effect():
    model = weather.load_weather_research()
    for roof in ("dome", "closed"):
        context = current_weather(roof=roof, wind=30, temperature=0)
        assert context["scenario_applicable"] is False
        assert weather.weather_log_odds_contrast(context, model) == 0


def test_probability_redistribution_preserves_broken_and_block_ratios():
    initial = {"made": .72, "missed": .24, "blocked": .03, "broken": .01}
    more_wind = weather.adjust_fg_probability_distribution(
        initial, log_odds_shift=-.30,
    )
    assert more_wind["made"] < initial["made"]
    assert more_wind["broken"] == initial["broken"]
    assert more_wind["blocked"] > initial["blocked"]
    assert math.isclose(sum(more_wind.values()), 1.0, abs_tol=1e-10)
    assert math.isclose(
        more_wind["blocked"] / (more_wind["missed"]+more_wind["blocked"]),
        .03 / (.24+.03), abs_tol=1e-10,
    )
    assert weather.adjust_fg_probability_distribution(
        initial, log_odds_shift=0.0,
    ) == initial
    with pytest.raises(ValueError, match="sum"):
        weather.adjust_fg_probability_distribution(
            {**initial, "made": 1.01}, log_odds_shift=-.2,
        )


def test_wrong_or_mutated_research_report_fails_closed(tmp_path):
    from json import loads, dumps
    payload = loads(weather.RESEARCH_REPORT.read_text())
    payload["validation_supports_official_weather_use"] = True
    target = tmp_path / "wrong-report.json"
    target.write_text(dumps(payload))
    with pytest.raises(ValueError, match="not approved"):
        weather.load_weather_research(target)
    payload["validation_supports_official_weather_use"] = False
    payload["effect_coefficients"]["per_10mph_log_odds"] = float("nan")
    target.write_text(dumps(payload))
    with pytest.raises(ValueError, match="not approved|plausible"):
        weather.load_weather_research(target)


@pytest.fixture(scope="module")
def real_engine():
    path = Path(__file__).resolve().parents[1] / "src/42_build_decision_engine.py"
    spec = importlib.util.spec_from_file_location(
        "weather_integration_real_frozen_engine", str(path),
    )
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def test_full_precision_q4_weather_sensitivity_isolated_from_frozen_engine(real_engine):
    state = {
        "qtr": 4, "game_seconds_remaining": 299.0,
        "yardline_100": 6.0, "ydstogo": 4.0,
        "score_differential": -4.0,
        "posteam_timeouts_remaining": 2,
        "defteam_timeouts_remaining": 3,
        "site": "AWAY", "roof": "outdoors",
    }
    original = real_engine.field_goal_probabilities
    recommend = real_engine.recommend
    out = weather.compare_weather_scenario(
        state, weather=current_weather(wind=25),
        live_engine=real_engine,
    )
    assert out["applicable"] is True
    assert out["data_source"] == "NFLVERSE_2014_2024_DEVELOPMENT_ONLY"
    assert out["original_made_probability"] > out["weather_made_probability"]
    assert 0 <= out["weather_fg_win_probability"] <= 1
    assert out["official_model_unchanged"] is True
    assert out["model_retrained"] is False
    assert out["real_2027_data_accessed"] is False
    assert out["validation_shows_brier_improvement"] is False
    assert out["adjusted_rain_snow_numeric_effect"] is False
    # Functional global isolation: the cached source engine is unchanged.
    assert real_engine.field_goal_probabilities is original
    assert real_engine.recommend is recommend
    assert recommend.__globals__["field_goal_probabilities"] is original


def test_weather_scenario_does_not_admit_ot_or_short_simulation_budget(real_engine):
    state = {
        "qtr": 5, "game_seconds_remaining": 50,
        "yardline_100": 6, "ydstogo": 4,
        "score_differential": 0, "site": "AWAY", "roof": "outdoors",
    }
    with pytest.raises(ValueError, match="regulation quarters only"):
        weather.compare_weather_scenario(
            state, weather=current_weather(), live_engine=real_engine,
        )
    state["qtr"] = 4
    with pytest.raises(ValueError, match="1200"):
        weather.compare_weather_scenario(
            state, weather=current_weather(), live_engine=real_engine,
            simulations=200,
        )


def test_closed_stadium_what_if_never_calls_frozen_engine():
    class Trap:
        def __getattribute__(self, name):
            raise AssertionError("Indoor scenario should not load an engine")
    out = weather.compare_weather_scenario(
        {"qtr": 4, "roof": "closed"},
        weather=current_weather(roof="closed"), live_engine=Trap(),
    )
    assert out["applicable"] is False
    assert out["official_model_unchanged"] is True
