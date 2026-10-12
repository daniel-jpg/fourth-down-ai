"""Browser QA of the real research-only weather UI against Streamlit.

Runs the actual official Q4 engine and weather what-if in the independently
working Docker container. No remote NOAA requests, play outcomes, model
updates, or prospective data access. Source snapshots remain in CI logs.
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
    assert metric(frame, "Weather what-if FG make") is None, (
        "Weather analysis must be opt-in"
    )
    print("WEATHER_BASELINE_OFFICIAL", default, flush=True)

    click_text(frame, "Weather observations (optional)")
    click_text(frame, "Include temperature")
    assert frame.get_by_text("Wind speed (mph)", exact=True).count() == 0
    click_text(frame, "Include wind speed")
    frame.get_by_text("Temperature (°F)", exact=True).wait_for(
        state="visible", timeout=25000
    )
    page.wait_for_timeout(1200)
    assert metric(frame, "Recommendation") is None, (
        "Enabling weather must invalidate a stale completed result"
    )

    set_number(frame, page, "Temperature (°F)", 32)
    set_number(frame, page, "Wind speed (mph)", 25)
    select_choice(frame, page, "Weather conditions", "Snow")
    # Remote 0.15-CPU Streamlit may take several seconds to rerun after a
    # widget change. Complete the precipitation rerun before changing wind
    # direction; otherwise an overlapping update can show stale choices.
    precipitation_note = frame.get_by_text(
        "no unvalidated rain/snow/fog probability penalty", exact=False,
    )
    precipitation_note.wait_for(state="visible", timeout=45000)
    select_choice(
        frame, page, "Wind direction relative to offense", "Headwind"
    )
    precipitation_note.wait_for(state="visible", timeout=45000)
    b = body_text(frame)
    assert "no unvalidated rain/snow/fog probability penalty" in b, (
        "Precipitation selection did not persist after wind-direction rerun: "
        + b[-600:]
    )
    cold_snow = analyze(frame, page, weather=True)
    assert cold_snow["Recommendation"] == default["Recommendation"], (
        "Weather research changed the official recommended action"
    )
    assert cold_snow["Win probability if chosen"] == (
        default["Win probability if chosen"]
    ), "Weather research changed the official win probability"
    assert (
        pct(cold_snow["Weather what-if FG make"]) <
        pct(cold_snow["Frozen-model FG make estimate"])
    ), "Windier/cooler conditions should lower the exploratory clean FG make"
    assert "not improve held-out Brier score" in body_text(frame)
    assert "Wind direction and precipitation effects are NOT modeled" in (
        body_text(frame)
    )
    print("WEATHER_WIND_COLD_SNOW_PASS", cold_snow, flush=True)

    # Precipitation was NOT fitted: with the same temp/wind/seed,
    # changing Snow to Rain must not change the what-if numbers.
    select_choice(frame, page, "Weather conditions", "Rain")
    await_metric_cleared(frame, page, "Weather what-if FG make")
    rain = analyze(frame, page, weather=True)
    for label in (
        "Recommendation", "Win probability if chosen",
        "Weather what-if FG make", "Weather what-if FG win probability",
    ):
        assert rain[label] == cold_snow[label], (
            f"Unvalidated precipitation changed {label}"
        )
    print("WEATHER_RAIN_SNOW_DESCRIPTIVE_ONLY_PASS", flush=True)

    select_choice(
        frame, page, "Wind direction relative to offense", "Tailwind"
    )
    await_metric_cleared(frame, page, "Weather what-if FG make")
    tailwind = analyze(frame, page, weather=True)
    for label in (
        "Recommendation", "Win probability if chosen",
        "Weather what-if FG make", "Weather what-if FG win probability",
    ):
        assert tailwind[label] == rain[label], (
            f"Unvalidated wind direction changed {label}"
        )
    print("WEATHER_WIND_DIRECTION_DESCRIPTIVE_ONLY_PASS", flush=True)

    # A pasted unsupported JSON receipt must be rejected and never change
    # the official or experimental outcome. No NOAA network request.
    click_text(frame, "Advanced reference: stadiums and saved airport data (does not affect predictions)")
    click_text(frame, "Open saved airport report (JSON required)")
    textarea = frame.get_by_role(
        "textbox", name="METAR research receipt JSON"
    )
    textarea.fill('{"unsupported": "made-up airport weather"}')
    textarea.press("Tab")
    # On the constrained public host, the Streamlit warning may only
    # appear after a slow server rerun. Wait for the new warning rather
    # than reading the old page immediately after Tab.
    warning = frame.get_by_text(
        "Invalid, inconsistent or unsupported METAR receipt", exact=False
    )
    warning.wait_for(state="visible", timeout=60000)
    b = body_text(frame)
    assert "Invalid, inconsistent or unsupported METAR receipt" in b
    assert metric(frame, "Recommendation") == default["Recommendation"]
    assert metric(frame, "Weather what-if FG make") == (
        rain["Weather what-if FG make"]
    )
    print("WEATHER_INVALID_METAR_REJECTED_NO_POLICY_CHANGE", flush=True)

    click_text(frame, "Show stadium information (reference only)")
    frame.get_by_text("2026 venue context", exact=True).wait_for(
        state="visible", timeout=50000
    )
    click_text(frame, "Official 2026 international game")
    frame.get_by_text("2026 NFL international fixture", exact=True).wait_for(
        state="visible", timeout=50000
    )
    assert frame.get_by_text("2026 venue to inspect", exact=True).count() == 0
    assert metric(frame, "Recommendation") == default["Recommendation"]
    print("WEATHER_INTERNATIONAL_VENUE_DOES_NOT_LEAK_HOME_GEOMETRY", flush=True)

    # Covered-roof games must NOT show outdoor weather inputs or stale
    # what-if metrics. Roof is an official input, so don't require the
    # original outdoor prediction to remain the same after this change.
    select_choice(frame, page, "Stadium / roof", "Fixed dome")
    frame.get_by_text(
        "Covered stadium: outdoor temperature and wind do not apply",
        exact=False,
    ).wait_for(state="visible", timeout=50000)
    for label in ("Include temperature", "Include wind speed"):
        assert frame.get_by_text(label, exact=True).count() == 0, (
            "Dome offered outdoor weather adjustment"
        )
    await_metric_cleared(frame, page, "Weather what-if FG make")
    await_metric_cleared(frame, page, "Recommendation")
    dome = analyze(frame, page, weather=False)
    assert dome["Recommendation"], "Dome official result missing"
    assert metric(frame, "Weather what-if FG make") is None
    print("WEATHER_COVERED_ROOF_POLICY_ONLY_PASS", dome, flush=True)
    return {"baseline": default, "snow": cold_snow, "rain": rain, "dome": dome}


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
