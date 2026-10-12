"""Optional OUTDOOR weather what-if on top of the FROZEN FG engine.

Manual current temperature F and wind mph are compared to 65 F / 5 mph
using a pinned *2014-2022 trained, 2023-24 validated* exploratory logistic
FG clean-make contrast. The WEATHER model failed the held-out Brier
improvement check, so it MUST NOT change official production forecasts.
Rain/snow/fog are recorded but not quantitatively fitted.

The entire scenario runs in a copied per-user Python function namespace.
All existing production model artifacts, Q4 native kickoff patches and
future-season holdout gates remain untouched; no external weather API.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .streamlit_kickoff_preview import (
    isolated_engine_from_loaded_engine,
    preview_native_kickoff_feature_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_REPORT = ROOT / "data/fg_weather_2014_2024_research_only.json"
CONDITIONS = (
    "Unknown",
    "Clear / dry",
    "Cloudy",
    "Rain",
    "Snow",
    "Mixed precipitation",
    "Fog",
)
OUTDOOR_ROOFS = frozenset(("outdoors", "open"))
COEFFICIENT_KEYS = frozenset(("per_20f_log_odds", "per_10mph_log_odds"))
TEMP_REF_F = 65.0
WIND_REF_MPH = 5.0
MAX_LOG_ODDS_CHANGE = 0.55


def validate_weather_input(
    *, roof: Any, temperature_f: Any,
    wind_mph: Any, conditions: Any,
) -> dict[str, Any]:
    if roof not in ("outdoors", "open", "dome", "closed"):
        raise ValueError("Unsupported or unavailable stadium roof state")
    if not isinstance(conditions, str) or conditions not in CONDITIONS:
        raise ValueError("Select a supported descriptive weather condition")
    if (isinstance(temperature_f, bool) or
            isinstance(wind_mph, bool)):
        raise ValueError("Temperature and wind must be numeric, not booleans")
    try:
        temp = float(temperature_f)
        wind = float(wind_mph)
    except (TypeError, ValueError) as exc:
        raise ValueError("Both outdoor temperature and wind speed are required") from exc
    if (not math.isfinite(temp) or not -25 <= temp <= 115
            or not math.isfinite(wind) or not 0 <= wind <= 40):
        raise ValueError(
            "Weather research range is -25 to 115 F and 0 to 40 mph; "
            "refuse extrapolation"
        )
    if roof not in OUTDOOR_ROOFS:
        return {
            "roof": roof, "temperature_f": temp, "wind_mph": wind,
            "conditions": conditions, "scenario_applicable": False,
            "reason": "Covered stadium: no outdoor wind/temperature adjustment",
        }
    return {
        "roof": roof, "temperature_f": temp, "wind_mph": wind,
        "conditions": conditions, "scenario_applicable": True,
        "reason": None,
    }


def load_weather_research(path: Path = RESEARCH_REPORT) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Pinned historical weather research report unavailable")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") !=
            "HISTORICAL_DEVELOPMENT_WEATHER_RESEARCH_ONLY"
            or report.get("source_seasons") != {
                "train": "2014-2022", "validation": "2023-2024"
            }
            or report.get("number_training", 0) < 900
            or report.get("number_validation", 0) < 200
            or report.get("validation_supports_official_weather_use") is not False
            or report.get("research_sensitivity_not_production") is not True
            or report.get("may_change_frozen_production") is not False
            or report.get("2025_heldout_data_used") is not False
            or report.get("2026_or_2027_data_used") is not False
            or report.get("condition_rain_snow_fitted") is not False):
        raise ValueError("Research report is not approved for weather scenarios")
    effects = report.get("effect_coefficients")
    if not isinstance(effects, dict) or set(effects) != COEFFICIENT_KEYS:
        raise ValueError("Weather research coefficients are missing")
    if any(
        type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1
        for v in effects.values()
    ):
        raise ValueError("Historical weather coefficient outside plausible bounds")
    if not isinstance(report.get("brier_improvement_baseline_minus_weather"), float):
        raise ValueError("Historical Brier validation result missing")
    return report


def weather_log_odds_contrast(
    weather: Mapping[str, Any], research: Mapping[str, Any],
) -> float:
    if not weather.get("scenario_applicable", False):
        return 0.0
    beta = research["effect_coefficients"]
    change = (
        beta["per_20f_log_odds"] *
        (weather["temperature_f"] - TEMP_REF_F) / 20.0
        + beta["per_10mph_log_odds"] *
        (weather["wind_mph"] - WIND_REF_MPH) / 10.0
    )
    if not math.isfinite(change):
        raise ValueError("Nonfinite outdoor weather contrast")
    return max(-MAX_LOG_ODDS_CHANGE, min(MAX_LOG_ODDS_CHANGE, change))


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def adjust_fg_probability_distribution(
    original: Mapping[str, float], *, log_odds_shift: float,
) -> dict[str, float]:
    """Only shift clean kick makes; keep breakdown and normalization valid."""
    if type(log_odds_shift) not in (float, int) or not math.isfinite(log_odds_shift):
        raise ValueError("Invalid research FG log-odds shift")
    required = {"made", "missed", "blocked", "broken"}
    if set(original) != required:
        raise ValueError("Unexpected frozen FG distribution keys")
    probs = {k: float(original[k]) for k in required}
    if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values())
            or abs(sum(probs.values()) - 1.0) > 1e-6):
        raise ValueError("Invalid frozen FG distribution")
    if abs(log_odds_shift) < 1e-12:
        return probs
    clean = 1.0 - probs["broken"]
    if clean <= 1e-9:
        raise ValueError("No clean FG outcomes available for weather scenario")
    original_clean_make = probs["made"] / clean
    eps = 1e-9
    original_clean_make = min(1-eps, max(eps, original_clean_make))
    logit = math.log(original_clean_make/(1-original_clean_make))
    clean_make = _logistic(logit+log_odds_shift)
    failures = probs["missed"]+probs["blocked"]
    conditional_block = probs["blocked"]/failures if failures > eps else 0.0
    total_failure = clean*(1-clean_make)
    answer = {
        "made": clean*clean_make,
        "blocked": total_failure*conditional_block,
        "missed": total_failure*(1-conditional_block),
        "broken": probs["broken"],
    }
    if any(not 0 <= x <= 1 for x in answer.values()):
        raise ValueError("Weather-adjusted FG outcome probability is invalid")
    if abs(sum(answer.values())-1.0) > 1e-8:
        raise ValueError("Weather-adjusted probabilities do not sum to one")
    return answer


def compare_weather_scenario(
    state: Mapping[str, Any], *,
    weather: Mapping[str, Any], live_engine: Any,
    research_path: Path = RESEARCH_REPORT,
    simulations: int = 1200, seed: int = 43,
) -> dict[str, Any]:
    """Run frozen action simulator in PRIVATE namespace, research-only.

    Same original frozen weights, plus empirically estimated FG clean-make
    log-odds contrast. Punts, GO conversion and continuation weather effects
    are NOT estimated; temperature wind/roof alone do not identify headwind.
    """
    qtr = state.get("qtr")
    if type(qtr) is not int or qtr not in (1, 2, 3, 4):
        raise ValueError("Weather scenario supports regulation quarters only")
    if type(simulations) is not int or simulations != 1200 or seed != 43:
        raise ValueError("Weather scenario requires original full 1200/seed43")
    if state.get("roof") != weather.get("roof"):
        raise ValueError("Weather roof differs from frozen decision state")
    context = validate_weather_input(
        roof=weather.get("roof"),
        temperature_f=weather.get("temperature_f"),
        wind_mph=weather.get("wind_mph"),
        conditions=weather.get("conditions"),
    )
    if not context["scenario_applicable"]:
        return {
            "applicable": False,
            "reason": context["reason"],
            "official_model_unchanged": True,
        }
    research = load_weather_research(research_path)
    shift = weather_log_odds_contrast(context, research)
    private = isolated_engine_from_loaded_engine(live_engine)
    baseline = private.get("field_goal_probabilities")
    normalizer = private.get("normalize_state")
    if not callable(baseline) or not callable(normalizer):
        raise ValueError("Loaded engine lacks original field-goal interface")
    normalized = normalizer(dict(state))
    original_fg = baseline(normalized)
    changed_fg = adjust_fg_probability_distribution(
        original_fg, log_odds_shift=shift,
    )

    # ALL frozen engine callbacks use this per-request dictionary (never
    # mutate global module attributes or the shared cached FG artifact).
    def local_weather_fg(base: dict) -> dict:
        unchanged = baseline(base)
        return adjust_fg_probability_distribution(
            unchanged, log_odds_shift=shift,
        )
    private["field_goal_probabilities"] = local_weather_fg

    if qtr == 4:
        result = preview_native_kickoff_feature_candidate(
            dict(state), simulations=simulations, action_seed=seed,
            isolated_engine=private,
        )
    else:
        result = private["recommend"](
            dict(state), n_simulations=simulations, seed=seed,
        )
    rows = result.get("results")
    if rows is None or not {"action", "eligible", "expected_win_probability"}.issubset(rows):
        raise ValueError("Weather-research engine response missing actions")
    fg_row = rows.loc[rows["action"].eq("FIELD_GOAL") & rows["eligible"].eq(True)]
    if len(fg_row) != 1:
        return {
            "applicable": False,
            "reason": "Frozen model does not support FIELD GOAL here",
            "official_model_unchanged": True,
        }
    eligible = rows.loc[rows["eligible"].eq(True)]
    if eligible.empty or not eligible["expected_win_probability"].notna().all():
        raise ValueError("Weather what-if estimates invalid")
    winner = eligible.sort_values(
        "expected_win_probability", ascending=False
    ).iloc[0]
    return {
        "applicable": True,
        "weather": dict(context),
        "data_source": "NFLVERSE_2014_2024_DEVELOPMENT_ONLY",
        "scenario": "EXPERIMENTAL_WEATHER_FG_CLEAN_MAKE_ONLY",
        "original_made_probability": float(original_fg["made"]),
        "weather_made_probability": float(changed_fg["made"]),
        "weather_fg_win_probability": float(
            fg_row.iloc[0]["expected_win_probability"]
        ),
        "weather_modeled_top_action": str(winner["action"]),
        "applied_log_odds_change": shift,
        "research_validation_brier_gain": research[
            "brier_improvement_baseline_minus_weather"
        ],
        "validation_shows_brier_improvement": False,
        "adjusted_rain_snow_numeric_effect": False,
        "wind_direction_measured": False,
        "go_and_punt_weather_adjusted": False,
        "model_retrained": False,
        "official_model_unchanged": True,
        "real_2027_data_accessed": False,
        "note": (
            "Exploratory weather effect on clean FG outcomes only; "
            "weather-added validation Brier did not improve, so NO "
            "official action recommendation is changed."
        ),
    }
