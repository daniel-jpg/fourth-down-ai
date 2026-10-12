"""Realistic synthetic METAR fixtures; NO NOAA network or football labels."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from src.metar_decision_weather import (
    NO_REAL_POLICY, KNOT_TO_MPH,
    build_metar_decision_evidence, distance_km, parse_raw_metar, utc_time,
)
from scripts import capture_awc_decision_weather as cli


OBSERVED = "2026-10-12T18:00:00Z"
RECEIVED = "2026-10-12T18:04:00Z"
DECISION = "2026-10-12T18:35:00Z"
LATER_CAPTURE = "2026-10-12T18:45:00Z"
FIELD = {"field_lat": 47.595, "field_lon": -122.332}


def sample(
    raw: str = "METAR KSEA 121800Z 09012G18KT 10SM -RA BKN015 12/08 A2996 RMK AO2",
) -> dict:
    return {
        "icaoId": "KSEA",
        "obsTime": int(datetime(2026, 10, 12, 18, 0, tzinfo=timezone.utc).timestamp()),
        "receiptTime": RECEIVED,
        "lat": 47.4489,
        "lon": -122.3094,
        "rawOb": raw,
        "wdir": 90,
        "wspd": 12,
        "wxString": "-RA",
    }


def packet(observations=None, *, heading=90, station="KSEA",
           decision=DECISION, captured=LATER_CAPTURE, roof="outdoors",
           source=None, verified=False, **kwargs):
    reports = [sample()] if observations is None else observations
    source_bytes = json.dumps(reports, sort_keys=True).encode() if source is None else source
    return build_metar_decision_evidence(
        observations=reports, source_bytes=source_bytes,
        station_id=station, decision_at_utc=decision, captured_at_utc=captured,
        offense_attack_heading_true_deg=heading, roof=roof,
        fetched_from_awc_https=verified, **(FIELD | kwargs),
    )


def test_airport_metar_produces_headwind_but_no_field_certification():
    d = packet()
    assert d["status"] == NO_REAL_POLICY
    assert d["station_id"] == "KSEA"
    assert d["raw_metar"].startswith("METAR KSEA")
    assert d["wind_from_true_degrees"] == 90
    assert d["wind_speed_knots"] == 12
    assert d["headwind_mph_signed"] == pytest.approx(
        12*KNOT_TO_MPH, abs=.002,
    )
    assert abs(d["crosswind_from_right_mph_signed"]) < .002
    assert d["tailwind_mph"] == 0
    assert d["airport_present_weather_code"] == "AIRPORT_RAIN_REPORTED"
    assert d["observed_at_utc"] == OBSERVED
    assert d["noaa_receipt_at_utc"] == RECEIVED
    assert d["decision_at_utc"] == DECISION
    assert d["observation_age_minutes"] == 35
    assert 10 < d["station_distance_km"] < 30
    assert d["observed_and_received_before_decision"] is True
    assert d["source_downloaded_after_decision"] is True
    assert d["provider_origin_https_verified_for_this_capture"] is False
    assert d["user_supplied_field_direction_independently_verified"] is False
    assert d["source_received_by_user_before_decision_independently_verified"] is False
    assert d["precipitation_mm_per_hour_at_field"] is None
    assert d["airport_precipitation_is_not_field_intensity"] is True
    assert d["snap_specific_weather_verified"] is False
    assert d["may_change_official_go_fg_or_punt_model"] is False
    assert d["real_2027_outcomes_accessed"] is False
    assert len(d["source_response_sha256"]) == 64
    assert len(d["raw_metar_sha256"]) == 64


def test_head_tail_crosswind_components_follow_meteorological_from_not_to():
    head = packet(heading=90)
    tail = packet(heading=270)
    from_right = packet(heading=0)
    from_left = packet(heading=180)
    assert head["headwind_mph_signed"] == pytest.approx(
        -tail["headwind_mph_signed"], abs=.002,
    )
    assert tail["tailwind_mph"] == pytest.approx(
        12*KNOT_TO_MPH, abs=.002,
    )
    assert from_right["crosswind_from_right_mph_signed"] > 0
    assert from_left["crosswind_from_right_mph_signed"] < 0
    assert abs(from_left["headwind_mph_signed"]) < .002
    assert abs(from_right["headwind_mph_signed"]) < .002


def test_calm_and_variable_wind_are_not_false_directional_forecasts():
    calm = sample("METAR KSEA 121800Z 00000KT 10SM CLR 12/08 A2996")
    calm["wdir"] = 0
    calm["wspd"] = 0
    a = packet([calm])
    assert a["orientation_status"] == "CALM_WIND"
    assert a["headwind_mph_signed"] == a["crosswind_from_right_mph_signed"] == 0
    assert a["airport_present_weather_code"] == (
        "AIRPORT_PRECIP_NOT_REPORTED_UNKNOWN"
    )

    variable = sample("METAR KSEA 121800Z VRB12KT 10SM BKN020 12/08 A2996")
    variable["wdir"] = "VRB"
    a = packet([variable])
    assert a["orientation_status"] == "VARIABLE_WIND_DIRECTION_UNRESOLVED"
    assert a["headwind_mph_signed"] is None
    assert a["crosswind_from_right_mph_signed"] is None

    sector = sample("METAR KSEA 121800Z 09012KT 010V180 10SM 12/08 A2996")
    a = packet([sector])
    assert a["wind_direction_variable"] is True
    assert a["headwind_mph_signed"] is None


@pytest.mark.parametrize("raw,expected", [
    ("KSEA 121800Z 09012KT 10SM -RA SCT030 12/08 A3000",
     "AIRPORT_RAIN_REPORTED"),
    ("KSEA 121800Z 09012KT 10SM +TSRA SCT030 12/08 A3000",
     "AIRPORT_RAIN_REPORTED"),
    ("KSEA 121800Z 09012KT 10SM -SN SCT030 12/08 A3000",
     "AIRPORT_SNOW_ICE_REPORTED"),
    ("KSEA 121800Z 09012KT 10SM -RASN SCT030 12/08 A3000",
     "AIRPORT_MIXED_PRECIP_REPORTED"),
    ("KSEA 121800Z 09012KT 10SM CLR 12/08 A3000 RMK RA",
     "AIRPORT_PRECIP_NOT_REPORTED_UNKNOWN"),
    ("KSEA 121800Z 09012KT 10SM SCT030 12/08 A3000",
     "AIRPORT_PRECIP_NOT_REPORTED_UNKNOWN"),
])
def test_present_weather_only_from_station_report_tokens(raw, expected):
    assert parse_raw_metar(raw)["present_weather_class"] == expected


@pytest.mark.parametrize("raw", [
    "KSEA 121800Z 09012MPS 10SM BKN020 12/08 A2996",
    "KSEA 121800Z 00012KT 10SM BKN020 12/08 A2996",
    "KSEA 121800Z 09012G05KT 10SM BKN020 12/08 A2996",
    "KSEA 121800Z VRB199KT 10SM BKN020 12/08 A2996",
    "KSEA 121800Z UNKNOWN 10SM BKN020 12/08 A2996",
    "KSEA 329900Z 09012KT 10SM BKN020 12/08 A2996",
    "FGBAD 121800Z 09012KT 10SM BKN020 12/08 A2996",
])
def test_malformed_or_unapproved_metar_wind_refused(raw):
    with pytest.raises(ValueError):
        parse_raw_metar(raw)


def test_stale_future_or_late_published_sample_refused():
    with pytest.raises(ValueError, match="within 90 minutes"):
        packet(decision="2026-10-12T19:40:00Z",
               captured="2026-10-12T19:45:00Z")
    later = sample()
    later["obsTime"] = int(datetime(
        2026, 10, 12, 18, 50, tzinfo=timezone.utc,
    ).timestamp())
    later["rawOb"] = later["rawOb"].replace("121800Z", "121850Z")
    later["receiptTime"] = "2026-10-12T18:52:00Z"
    with pytest.raises(ValueError, match="before the football decision"):
        packet([later])
    same_obs_late_receipt = sample()
    same_obs_late_receipt["receiptTime"] = "2026-10-12T18:36:00Z"
    with pytest.raises(ValueError, match="received by NOAA before"):
        packet([same_obs_late_receipt])
    # In a mixed 6-hour API result, the future report must be skipped.
    chosen = packet([later, sample()])
    assert chosen["observed_at_utc"] == OBSERVED


def test_station_distance_exact_id_timestamp_and_raw_json_binding():
    with pytest.raises(ValueError, match="too far"):
        packet(field_lat=40.0, field_lon=-73.0)
    with pytest.raises(ValueError, match="station mismatched"):
        packet(station="KDEN")
    altered = sample()
    altered["wspd"] = 4
    with pytest.raises(ValueError, match="wind speed conflicts"):
        packet([altered])
    altered = sample()
    altered["obsTime"] += 60
    with pytest.raises(ValueError, match="observation time"):
        packet([altered])
    with pytest.raises(ValueError, match="hashed source bytes"):
        packet(source=b'[{"rawOb":"forged"}]')
    assert distance_km((0, 0), (0, 0)) == 0


def test_unavailable_sources_invalid_venue_and_bad_bearings():
    with pytest.raises(ValueError, match="Only open-roof"):
        packet(roof="closed")
    with pytest.raises(ValueError, match="offense attacking"):
        packet(heading=361)
    with pytest.raises(ValueError, match="timezone"):
        packet(decision="2026-10-12T18:35:00")
    with pytest.raises(ValueError, match="retrieval precedes NOAA"):
        packet(captured="2026-10-12T18:03:00Z")
    with pytest.raises(ValueError, match="Required|ICAO|station"):
        packet(station="KSEA;rm -rf .")


def test_duplicate_conflicting_reports_at_same_time_fail_closed():
    one, two = sample(), sample()
    two["rawOb"] = two["rawOb"].replace("10SM", "9SM")
    with pytest.raises(ValueError, match="Ambiguous duplicated"):
        packet([one, two])


def test_explicit_noaa_download_is_single_host_and_a_one_shot_not_ui(monkeypatch):
    assert cli.awc_url("KSEA") == (
        "https://aviationweather.gov/api/data/metar?ids=KSEA&format=json&hours=6"
    )
    with pytest.raises(ValueError, match="ICAO"):
        cli.awc_url("KSEA&ids=KKKK")
    requested = []
    payload = json.dumps([sample()]).encode()
    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}
        def __init__(self, url):
            self.url = url
        def geturl(self):
            return self.url
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self, n):
            return payload
    def fake_urlopen(req, timeout):
        requested.append((req.full_url, timeout))
        return FakeResponse(req.full_url)
    monkeypatch.setattr(cli, "urlopen", fake_urlopen)
    data, captured = cli.fetch_recent_awc_metar("KSEA")
    assert data == payload
    assert captured.tzinfo == timezone.utc
    assert requested == [(cli.awc_url("KSEA"), 20)]


def test_offline_fixture_and_exclusive_audit_output(tmp_path):
    blob = json.dumps([sample()], sort_keys=True).encode()
    evidence = cli.capture_weather(
        source_bytes=blob, station="KSEA",
        decision_utc=DECISION, captured_utc=LATER_CAPTURE,
        field_lat=FIELD["field_lat"], field_lon=FIELD["field_lon"],
        offense_heading_true_deg=90, roof="outdoors",
        fetched_from_awc_https=False,
    )
    assert evidence["source_endpoint"] == (
        "OFFLINE_JSON_FIXTURE_NOT_A_VERIFIED_NOAA_FETCH"
    )
    assert evidence["evidence_can_be_used_for_production_wps"] is False
    target = tmp_path / "raw_metar_receipt.json"
    cli.write_weather_receipt(target, evidence)
    assert json.loads(target.read_text()) == evidence
    original = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="already exists"):
        cli.write_weather_receipt(target, evidence)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == original


def test_utc_accepts_epoch_and_offsets_without_guessing_timezone():
    assert utc_time(1791828000).tzinfo == timezone.utc
    assert utc_time("2026-10-12T11:00:00-07:00") == utc_time(OBSERVED)
    with pytest.raises(ValueError):
        utc_time(float("nan"))
    with pytest.raises(ValueError):
        utc_time("Monday at noon")



def test_research_receipt_display_is_arithmetically_checked_but_unsigned():
    from src.metar_decision_weather import validate_metar_receipt_for_display
    data = packet()
    result = validate_metar_receipt_for_display(data)
    assert result["airport"] == "KSEA"
    assert result["observed_at_utc"] == OBSERVED
    assert result["age_minutes"] == 35
    assert result["captured_after_decision"] is True
    assert result["headwind_mph_signed"] > 13
    assert result["airport_precipitation"] == "AIRPORT_RAIN_REPORTED"
    assert result["unsigned_manual_import_not_source_authenticated"] is True
    assert result["changes_official_recommendation"] is False
    for field, bad in (
        ("headwind_mph_signed", -100),
        ("airport_present_weather_code", "CLEAR_SKIES_ON_FIELD"),
        ("snap_specific_weather_verified", True),
        ("source_downloaded_after_decision", False),
        ("raw_metar_sha256", "0"*64),
    ):
        altered = deepcopy(data)
        altered[field] = bad
        with pytest.raises(ValueError):
            validate_metar_receipt_for_display(altered)



def test_noaa_published_json_example_shape_parses_without_false_rain():
    """NOAA-compatible published 2023 sample, not a live HTTP response."""
    sample = [{
        "icaoId": "KORD",
        "receiptTime": "2023-11-03 21:54:03.000Z",
        "obsTime": 1699048260,
        "reportTime": "2023-11-03 22:00:00.000Z",
        "temp": 14.4,
        "dewp": 2.8,
        "wdir": 230,
        "wspd": 6,
        "wgst": 12,
        "wxString": "-RA",  # Not actually in the coded present-weather group.
        "rawOb": (
            "KORD 032151Z 23006KT 10SM BKN110 OVC250 "
            "14/03 A3000 RMK AO2 SLP162 VIRGA OHD T01440028"
        ),
        "lat": 41.9602,
        "lon": -87.9316,
    }]
    original = json.dumps(sample).encode("utf-8")
    report = build_metar_decision_evidence(
        observations=sample, source_bytes=original, station_id="KORD",
        decision_at_utc="2023-11-03T22:10:00Z",
        captured_at_utc="2023-11-03T22:12:00Z",
        field_lat=41.9602, field_lon=-87.9316,
        offense_attack_heading_true_deg=230,
        roof="outdoors", fetched_from_awc_https=False,
    )
    assert report["observed_at_utc"] == "2023-11-03T21:51:00Z"
    assert report["noaa_receipt_at_utc"] == "2023-11-03T21:54:03Z"
    assert report["wind_speed_knots"] == 6
    assert report["headwind_mph_signed"] > 6.8
    assert report["airport_present_weather_code"] == (
        "AIRPORT_PRECIP_NOT_REPORTED_UNKNOWN"
    )
    assert report["may_change_official_go_fg_or_punt_model"] is False
    assert report["provider_origin_https_verified_for_this_capture"] is False


def test_noaa_one_transient_timeout_is_retried_at_most_once(monkeypatch):
    from unittest.mock import Mock
    payload = json.dumps([sample()]).encode()
    calls = []
    delays = []
    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}
        def __init__(self, url):
            self.url = url
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def geturl(self):
            return self.url
        def read(self, n):
            return payload

    def flaky(req, timeout):
        calls.append((req.full_url, timeout))
        if len(calls) == 1:
            raise TimeoutError("Transient NOAA socket read")
        return FakeResponse(req.full_url)

    monkeypatch.setattr(cli, "urlopen", flaky)
    monkeypatch.setattr(cli.time, "sleep", delays.append)
    raw, fetched = cli.fetch_recent_awc_metar("KSEA")
    assert raw == payload
    assert fetched.tzinfo == timezone.utc
    assert len(calls) == 2
    assert calls[0] == calls[1] == (cli.awc_url("KSEA"), 20)
    assert delays == [2]


def test_noaa_unavailable_is_not_misreported_as_verified_weather(monkeypatch):
    attempts = []
    def always_timeout(req, timeout):
        attempts.append(req.full_url)
        raise TimeoutError("NOAA not responding")
    monkeypatch.setattr(cli, "urlopen", always_timeout)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    with pytest.raises(
        ConnectionError, match="NOAA_AWC_READ_TIMEOUT_TWICE_LIVE_WEATHER_NOT_VERIFIED"
    ):
        cli.fetch_recent_awc_metar("KSEA")
    assert attempts == [cli.awc_url("KSEA")] * 2


def test_noaa_rate_limits_and_empty_observations_are_never_retried(monkeypatch):
    from urllib.error import HTTPError
    counts = []
    def rate_limited(req, timeout):
        counts.append(req.full_url)
        raise HTTPError(req.full_url, 429, "rate limited", None, None)
    monkeypatch.setattr(cli, "urlopen", rate_limited)
    with pytest.raises(ConnectionError, match="NOAA_AWC_NETWORK_UNAVAILABLE"):
        cli.fetch_recent_awc_metar("KSEA")
    assert counts == [cli.awc_url("KSEA")]

    class NoContentResponse:
        status = 204
        headers = {"Content-Type": "application/json"}
        def geturl(self):
            return cli.awc_url("KSEA")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(
        cli, "urlopen", lambda *args, **kw: NoContentResponse(),
    )
    with pytest.raises(ValueError, match="NOAA_AWC_NO_OBSERVATIONS_AVAILABLE"):
        cli.fetch_recent_awc_metar("KSEA")
