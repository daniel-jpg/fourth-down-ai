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
                if frame.get_by_text("Game Situation", exact=True).count() == 1:
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
    deadline = time.monotonic() + 110
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
    combobox.click(timeout=15000)
    option = frame.get_by_role("option", name=value, exact=True)
    if option.count() != 1:
        option = frame.get_by_text(value, exact=True).last
    option.click(timeout=15000)
    page.wait_for_timeout(1400)


def analyze(frame, page, *, weather: bool) -> dict:
    button = frame.get_by_role("button", name="Analyze Fourth Down")
    assert button.count() == 1 and button.is_enabled()
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
    click_text(frame, "Enter observed temperature and wind")
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
    select_choice(
        frame, page, "Wind direction relative to offense", "Headwind"
    )
    b = body_text(frame)
    assert "no unvalidated rain/snow/fog probability penalty" in b
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
    assert metric(frame, "Weather what-if FG make") is None, (
        "Changed weather controls must invalidate the previous scenario"
    )
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
    assert metric(frame, "Weather what-if FG make") is None
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
    click_text(frame, "Inspect timestamped airport METAR evidence")
    textarea = frame.get_by_role(
        "textbox", name="METAR research receipt JSON"
    )
    textarea.fill('{"unsupported": "made-up airport weather"}')
    textarea.press("Tab")
    page.wait_for_timeout(1200)
    b = body_text(frame)
    assert "Invalid, inconsistent or unsupported METAR receipt" in b
    assert metric(frame, "Recommendation") == default["Recommendation"]
    assert metric(frame, "Weather what-if FG make") == (
        rain["Weather what-if FG make"]
    )
    print("WEATHER_INVALID_METAR_REJECTED_NO_POLICY_CHANGE", flush=True)

    click_text(frame, "Show 2026 stadium compass lookup (research only)")
    frame.get_by_text("2026 venue context", exact=True).wait_for(
        state="visible", timeout=20000
    )
    click_text(frame, "Official 2026 international game")
    frame.get_by_text("2026 NFL international fixture", exact=True).wait_for(
        state="visible", timeout=20000
    )
    assert frame.get_by_text("2026 venue to inspect", exact=True).count() == 0
    assert metric(frame, "Recommendation") == default["Recommendation"]
    print("WEATHER_INTERNATIONAL_VENUE_DOES_NOT_LEAK_HOME_GEOMETRY", flush=True)

    # Covered-roof games must NOT show outdoor weather inputs or stale
    # what-if metrics. Roof is an official input, so don't require the
    # original outdoor prediction to remain the same after this change.
    select_choice(frame, page, "Stadium / roof", "Fixed dome")
    frame.get_by_text(
        "The roof is closed or domed.", exact=False
    ).wait_for(state="visible", timeout=20000)
    assert frame.get_by_text(
        "Enter observed temperature and wind", exact=True
    ).count() == 0, "Dome offered outdoor weather adjustment"
    assert metric(frame, "Weather what-if FG make") is None
    assert metric(frame, "Recommendation") is None
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
