"""Conservative parsing of historical NFLverse GAME-LEVEL weather descriptions.

These text tags are evidence that a precipitation word was reported, NOT
measurement of rain/snow at the fourth-down snap. Missing or cloudy text
is never interpreted as confirmed dry. Wind direction is not inferred from
stadium-level wind speed or a generic historical weather description.

No HTTP calls, future-season access, pretrained weights or side effects.
"""
from __future__ import annotations

import re
from typing import Any

RAIN_WORDS = re.compile(
    r"\b(rain|raining|rainfall|drizzle|drizzling|shower|showers|"
    r"thunderstorm|thunderstorms)\b", re.I,
)
SNOW_WORDS = re.compile(
    r"\b(snow|snowing|snowfall|sleet|sleeted|hail|hailing|"
    r"flurry|flurries|blizzard)\b", re.I,
)
DRY_WORDS = re.compile(r"\b(clear|sunny|sunshine|fair|dry)\b", re.I)
UNCERTAIN = re.compile(
    r"\b(chance of|possible|possibility of|forecast|expected|likely|"
    r"without|no)\s+(?:light\s+|heavy\s+|some\s+)?"
    r"(?:rain|snow|sleet|showers?|precipitation)(?:\s+showers?)?\b", re.I,
)
MISSING = frozenset({
    "", "none", "null", "nan", "n/a", "na", "unknown",
    "not available", "not recorded",
})


def classify_weather_description(raw: Any) -> str:
    """Conservative *text* label, never a validated per-play precipitation."""
    if raw is None or not isinstance(raw, str):
        return "UNSPECIFIED"
    value = raw.strip().lower()
    if value in MISSING:
        return "UNSPECIFIED"
    # Probabilistic forecasts and negations are not confirmed rain/snow.
    value = UNCERTAIN.sub(" ", value)
    value = re.sub(
        r"\b(?:rain|snow|showers?)\s+"
        r"(?:unlikely|likely|possible|expected)\b", " ", value,
    )
    # A "snow shower" is snow, not simultaneous rain and snow.
    value = re.sub(r"\bsnow\s+showers?\b", "snow", value)
    rain = bool(RAIN_WORDS.search(value))
    snow = bool(SNOW_WORDS.search(value))
    if rain and snow:
        return "MIXED_MENTION"
    if rain:
        return "RAIN_MENTION"
    if snow:
        return "SNOW_ICE_MENTION"
    if DRY_WORDS.search(value):
        return "EXPLICIT_DRY_WORD"
    return "UNSPECIFIED"


def text_features(label: str) -> dict[str, int]:
    """Mutually exclusive descriptive indicators; all else is unknown."""
    supported = {
        "RAIN_MENTION", "SNOW_ICE_MENTION", "MIXED_MENTION",
        "EXPLICIT_DRY_WORD", "UNSPECIFIED",
    }
    if label not in supported:
        raise ValueError("Unsupported historical weather text classification")
    return {
        "precip_rain_word": int(label in ("RAIN_MENTION", "MIXED_MENTION")),
        "precip_snow_word": int(label in ("SNOW_ICE_MENTION", "MIXED_MENTION")),
        "reported_dry_word": int(label == "EXPLICIT_DRY_WORD"),
        "weather_text_unspecified": int(label == "UNSPECIFIED"),
    }


def audit_direction_availability(columns: list[str]) -> dict[str, object]:
    """Do not equate an unverified column with per-play directional data."""
    candidates = sorted({
        c for c in columns
        if isinstance(c, str) and (
            "wind" in c.lower() or "direction" in c.lower()
        )
    })
    return {
        "source_columns_matching_wind_or_direction": candidates,
        "trusted_snap_level_wind_direction_available": False,
        "kick_relative_wind_direction_reconstructed": False,
        "directional_effect_estimated": False,
        "why_not": (
            "Historical game-level wind speed and text do not establish "
            "the wind vector or its direction relative to the line of "
            "scrimmage at a specific fourth-down decision."
        ),
    }
