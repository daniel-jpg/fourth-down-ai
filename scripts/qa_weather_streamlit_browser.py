"""Browser QA of optional weather fields and official decision isolation.

Exercises the real frozen Q4 engine plus the separately unvalidated FG
weather scenario. Research venue and receipt controls must remain absent.
"""
from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get(
    "FOURTH_DOWN_SMOKE_BASE_URL", "http://127.0.0.1:8531"
).rstrip("/")
SCREENSHOT = ROOT / "docker-streamlit-weather-qa.png"


def await_health(seconds: int = 120) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                BASE_URL + "/_stcore/health", timeout=2
            ) as response:
                if response.status == 200:
                    print("WEATHER_SERVER_HEALTHY", flush=True)
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise AssertionError("Docker Streamlit health endpoint unavailable")


def find_app(page, timeout: int = 90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in page.frames:
            try:
                if (
                    frame.get_by_text("Game Situation", exact=True).count() == 1
                    and frame.get_by_role(
                        "button", name="Analyze Fourth Down"
                    ).count() == 1
                ):
                    return frame
            except Exception:
                pass
        page.wait_for_timeout(1000)
    raise AssertionError(
        "Docker Streamlit game form did not render: "
        + str([(f.name, f.url) for f in page.frames])
    )


def body_text(frame) -> str:
    return frame.locator("body").inner_text(timeout=8000)


def metric(frame, label: str) -> str | None:
    cards = frame.locator('[data-testid="stMetric"]').filter(
        has=frame.get_by_text(label, exact=True)
    )
    if cards.count() == 0:
        return None
    assert cards.count() == 1, f"Expected one metric {label}: {cards.count()}"
    return cards.locator('[data-testid="stMetricValue"]').inner_text(
        timeout=8000
    ).strip()


def require_metric(frame, label: str) -> str:
    value = metric(frame, label)
    assert value is not None, f"Missing metric: {label}"
    return value


def pct(value: str) -> float:
    raw = value.strip().replace("%", "").replace(",", "")
    number = float(raw)
    assert 0 <= number <= 100, f"Invalid modeled percentage: {value}"
    return number


def await_analysis(frame, page, *, weather: bool) -> dict:
    # Keep the local/Docker default; allow a longer browser wait on the
    # 0.15-CPU Render Free host without reducing the frozen 1200-draw model.
    timeout = int(os.environ.get("FOURTH_DOWN_WEATHER_TIMEOUT_SECONDS", "110"))
    assert 30 <= timeout <= 360, "Invalid weather browser timeout"
    deadline = time.monotonic() + timeout
    consecutive = 0
    last = ""
    while time.monotonic() < deadline:
        last = body_text(frame)
        if "Older percentages have NOT been silently substituted" in last:
            raise AssertionError("Corrected Q4 reliability failure: " + last[-500:])
        labels = ("Recommendation", "Win probability if chosen")
        ready = all(metric(frame, label) is not None for label in labels)
        ready = ready and "Q4 estimates use corrected standard-35 kickoff inputs" in last
        if weather:
            ready = (
                ready
                and metric(frame, "Frozen-model FG make estimate") is not None
                and metric(frame, "Weather what-if FG make") is not None
                and metric(frame, "Weather what-if FG win probability") is not None
                and "Not validated for coaching decisions" in last
            )
        consecutive = consecutive + 1 if ready else 0
        if consecutive >= 2:
            result = {label: require_metric(frame, label) for label in labels}
            if weather:
                for label in (
                    "Frozen-model FG make estimate",
                    "Weather what-if FG make",
                    "Weather what-if FG win probability",
                ):
                    result[label] = require_metric(frame, label)
            return result
        page.wait_for_timeout(1200)
    raise AssertionError(
        f"Q4 analysis weather={weather} did not finish: {last[-1000:]}"
    )


def click_text(frame, label: str, timeout: int = 20000) -> None:
    control = frame.get_by_text(label, exact=True).first
    control.click(timeout=timeout)


def set_number(frame, page, label: str, value: int) -> None:
    spin = frame.get_by_role("spinbutton", name=label)
    if spin.count() != 1:
        spin = frame.get_by_label(label, exact=True)
    assert spin.count() == 1, f"Number input inaccessible: {label}"
    spin.fill(str(value), timeout=15000)
    spin.press("Tab")
    page.wait_for_timeout(1200)


def select_choice(frame, page, label: str, value: str) -> None:
    combobox = frame.get_by_role("combobox", name=label)
    assert combobox.count() == 1, (
        f"{label} selectbox inaccessible; count={combobox.count()}"
    )
    option = frame.get_by_role("option", name=value, exact=True)
    for attempt in range(2):
        combobox.click(timeout=15000)
        try:
            # The Streamlit combobox menu mounts asynchronously after click.
            # An immediate .count() can report zero and send the test to a
            # locator that never appeared, even though the widget is correct.
            option.wait_for(state="visible", timeout=4500)
            option.click(timeout=12000)
            page.wait_for_timeout(1400)
            return
        except Exception as exc:
            options = frame.get_by_role("option").all_inner_texts()
            print(
                "WEATHER_SELECT_RETRY",
                {
                    "label": label, "value": value, "attempt": attempt + 1,
                    "expanded": combobox.get_attribute("aria-expanded"),
                    "available_options": options[:10],
                    "error": str(exc)[:160],
                },
                flush=True,
            )
            page.keyboard.press("Escape")
            page.wait_for_timeout(600)
    if label == "Stadium / roof" and value == "Fixed dome":
        # Keyboard fallback for the four-option roof widget, which starts
        # at outdoor/open-air. We validate the covered-roof panel below.
        combobox.click(timeout=12000)
        combobox.press("Home")
        combobox.press("ArrowDown")
        combobox.press("Enter")
        page.wait_for_timeout(1700)
        print("WEATHER_ROOF_KEYBOARD_FALLBACK_USED", flush=True)
        return
    raise AssertionError(f"Cannot select {value} from {label}")


def await_metric_cleared(frame, page, label: str, timeout: int = 40) -> None:
    # Remote Streamlit reruns are asynchronous after selecting weather.
    # Wait for stale metrics to disappear instead of racing the UI.
    deadline = time.monotonic() + timeout
    consecutive = 0
    stale_cards = frame.locator('[data-testid="stMetric"]').filter(
        has=frame.get_by_text(label, exact=True)
    )
    while time.monotonic() < deadline:
        # A Streamlit rerun can detach stMetricValue between count() and
        # inner_text(); only the metric's presence matters here.
        consecutive = consecutive + 1 if stale_cards.count() == 0 else 0
        if consecutive >= 2:
            return
        page.wait_for_timeout(700)
    raise AssertionError(
        f"Changed weather controls left stale {label} visible after "
        f"{timeout} seconds. Visible metrics: "
        + repr(frame.locator('[data-testid="stMetric"]').all_inner_texts())[:450]
    )


def analyze(frame, page, *, weather: bool) -> dict:
    button = frame.get_by_role("button", name="Analyze Fourth Down")
    button.wait_for(state="visible", timeout=30000)
    deadline = time.monotonic() + 30
    while not button.is_enabled() and time.monotonic() < deadline:
        page.wait_for_timeout(500)
    assert button.count() == 1 and button.is_enabled(), (
        "Analyze is missing/disabled after full form render: "
        + body_text(frame)[-600:]
    )
    start = time.monotonic()
    button.click(timeout=15000)
    result = await_analysis(frame, page, weather=weather)
    print(
        "WEATHER_ANALYSIS_COMPLETED",
        {"weather": weather, "seconds": round(time.monotonic() - start, 2)},
        flush=True,
    )
    return result


def check_baseline_and_weather(frame, page) -> dict:
    default = analyze(frame, page, weather=False)
    assert metric(frame, "Weather what-if FG make") is None
    print("WEATHER_BASELINE_OFFICIAL", default, flush=True)

    click_text(frame, "Weather observations (optional)")
    temperature = frame.get_by_role("spinbutton", name="Temperature (°F)")
    wind = frame.get_by_role("spinbutton", name="Wind speed (mph)")
    temperature.wait_for(state="visible", timeout=25000)
    wind.wait_for(state="visible", timeout=25000)
    assert temperature.input_value() == "", (
        "Temperature input must start blank, not assume 65 F"
    )
    assert wind.input_value() == "", (
        "Wind input must start blank, not assume 5 mph"
    )
    text_before = body_text(frame)
    for removed in (
        "Include temperature", "Include wind speed",
        "Advanced reference: stadiums",
        "Weather conditions", "Wind direction relative to offense",
        "METAR research receipt JSON",
    ):
        assert removed not in text_before, f"Non-predictive control remains: {removed}"
    print("WEATHER_BLANK_FIELDS_NO_EXTRA_CONTROLS_PASS", flush=True)

    set_number(frame, page, "Temperature (°F)", 32)
    frame.get_by_text(
        "Add the other weather value", exact=False
    ).wait_for(state="visible", timeout=35000)
    assert wind.input_value() == "", "Temperature entry filled wind unexpectedly"
    only_temperature = analyze(frame, page, weather=False)
    assert only_temperature == default
    assert metric(frame, "Weather what-if FG make") is None
    print("WEATHER_TEMPERATURE_ONLY_OFFICIAL_UNCHANGED", flush=True)

    set_number(frame, page, "Wind speed (mph)", 25)
    await_metric_cleared(frame, page, "Recommendation")
    strong_wind = analyze(frame, page, weather=True)
    assert strong_wind["Recommendation"] == default["Recommendation"]
    assert strong_wind["Win probability if chosen"] == default["Win probability if chosen"]
    assert (
        pct(strong_wind["Weather what-if FG make"]) <
        pct(strong_wind["Frozen-model FG make estimate"])
    ), "Cooler/windier weather should reduce the research FG make estimate"
    assert "not improve held-out Brier score" in body_text(frame)
    assert "rain/snow and wind direction are not modeled" in body_text(frame)
    print("WEATHER_BOTH_OPTIONAL_VALUES_PASS", strong_wind, flush=True)

    # Wind speed zero is present data, not a blank or missing measurement.
    set_number(frame, page, "Wind speed (mph)", 0)
    await_metric_cleared(frame, page, "Weather what-if FG make")
    calm_wind = analyze(frame, page, weather=True)
    assert calm_wind["Recommendation"] == default["Recommendation"]
    assert calm_wind["Win probability if chosen"] == default["Win probability if chosen"]
    assert (
        pct(calm_wind["Weather what-if FG make"]) >
        pct(strong_wind["Weather what-if FG make"])
    ), "The what-if must respond to a 25-to-0 mph wind change"
    print("WEATHER_ZERO_WIND_VALID_VALUE_PASS", calm_wind, flush=True)

    # Clearing either weather box disables only the experimental comparison.
    temperature.fill("", timeout=15000)
    temperature.press("Tab")
    frame.get_by_text(
        "Add the other weather value", exact=False
    ).wait_for(state="visible", timeout=35000)
    await_metric_cleared(frame, page, "Weather what-if FG make")
    assert temperature.input_value() == ""
    empty_temp = analyze(frame, page, weather=False)
    assert empty_temp == default
    assert metric(frame, "Weather what-if FG make") is None
    print("WEATHER_CLEARED_FIELD_OFFICIAL_UNCHANGED", flush=True)

    # A dome must hide the outdoor-only fields.
    select_choice(frame, page, "Stadium / roof", "Fixed dome")
    frame.get_by_text(
        "Covered stadium: outdoor temperature and wind do not apply",
        exact=False,
    ).wait_for(state="visible", timeout=50000)
    for label in ("Temperature (°F)", "Wind speed (mph)"):
        assert frame.get_by_role("spinbutton", name=label).count() == 0, (
            f"Covered roof still offered {label}"
        )
    await_metric_cleared(frame, page, "Recommendation")
    dome = analyze(frame, page, weather=False)
    assert dome["Recommendation"], "Dome official recommendation missing"
    assert metric(frame, "Weather what-if FG make") is None
    print("WEATHER_COVERED_ROOF_POLICY_ONLY_PASS", dome, flush=True)
    return {"baseline": default, "windy": strong_wind, "calm": calm_wind,
            "dome": dome}


def main() -> None:
    await_health()
    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: print(
            "WEATHER_BROWSER_PAGE_ERROR", str(error)[:300], flush=True
        ))
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
            frame = find_app(page)
            assert frame.get_by_text(
                "Weather observations (optional)", exact=True
            ).count() == 1
            check_baseline_and_weather(frame, page)
            print("WEATHER_BROWSER_QA_COMPLETE", flush=True)
        finally:
            try:
                page.screenshot(path=str(SCREENSHOT), full_page=True)
            except Exception as error:
                print("WEATHER_SCREENSHOT_UNAVAILABLE", repr(error), flush=True)
            browser.close()


if __name__ == "__main__":
    main()
