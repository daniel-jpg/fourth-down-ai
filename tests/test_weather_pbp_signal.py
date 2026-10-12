"""Weather text is NOT a sensor and wind speed is NOT wind direction."""
import pytest

from src.weather_pbp_signal import (
    audit_direction_availability,
    classify_weather_description,
    text_features,
)


@pytest.mark.parametrize("description,expected", [
    (None, "UNSPECIFIED"),
    ("", "UNSPECIFIED"),
    ("Unknown", "UNSPECIFIED"),
    ("Cloudy", "UNSPECIFIED"),
    ("Overcast at kick-off", "UNSPECIFIED"),
    ("Clear and dry", "EXPLICIT_DRY_WORD"),
    ("SUNNY", "EXPLICIT_DRY_WORD"),
    ("Light rain, windy", "RAIN_MENTION"),
    ("Thunderstorms at the stadium", "RAIN_MENTION"),
    ("Snow showers", "SNOW_ICE_MENTION"),
    ("Snow flurries", "SNOW_ICE_MENTION"),
    ("Sleet, freezing", "SNOW_ICE_MENTION"),
    ("Heavy rain and snow", "MIXED_MENTION"),
    ("Chance of rain", "UNSPECIFIED"),
    ("Possible snow showers", "UNSPECIFIED"),
    ("No rain", "UNSPECIFIED"),
    ("Without snow", "UNSPECIFIED"),
    ("RAIN LIKELY", "UNSPECIFIED"),
    ("Teams were training indoors", "UNSPECIFIED"),
])
def test_weather_description_cannot_impute_dry_or_future_weather(
    description, expected,
):
    assert classify_weather_description(description) == expected


def test_precipitation_text_features_explicit_and_mutually_audited():
    rain = text_features("RAIN_MENTION")
    unknown = text_features("UNSPECIFIED")
    mixed = text_features("MIXED_MENTION")
    assert rain["precip_rain_word"] == 1
    assert rain["precip_snow_word"] == 0
    assert rain["reported_dry_word"] == 0
    assert unknown["reported_dry_word"] == 0
    assert unknown["weather_text_unspecified"] == 1
    assert mixed["precip_rain_word"] == mixed["precip_snow_word"] == 1
    with pytest.raises(ValueError, match="Unsupported"):
        text_features("ACTUAL_RAIN_AT_SNAP")


def test_no_kick_relative_wind_direction_reconstructed():
    audit = audit_direction_availability([
        "temp", "wind", "weather", "season", "wind_direction",
        "wind_gust", "kick_direction", "non_weather",
    ])
    assert audit["source_columns_matching_wind_or_direction"] == [
        "kick_direction", "wind", "wind_direction", "wind_gust",
    ]
    assert audit["trusted_snap_level_wind_direction_available"] is False
    assert audit["kick_relative_wind_direction_reconstructed"] is False
    assert audit["directional_effect_estimated"] is False
