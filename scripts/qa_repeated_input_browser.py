"""Real browser regression: repeated form editing must not darken the page.

Runs the actual Docker-hosted Streamlit form, including a complete initial Q4
analysis followed by realistic edits to score, down/distance, wind,
temperature, roof, game site and the interactive field. It never downloads
NFL results, trains models or changes the official decision policy.
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

from scripts.qa_weather_streamlit_browser import (
    BASE_URL,
    await_analysis,
    await_health,
    find_app,
    metric,
    select_choice,
)

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = ROOT / "repeated-input-no-dimming.png"

VISUAL_STATE_JS = r"""(element) => {
  const chain = [];
  let opacity = 1;
  let current = element;
  while (current && current.nodeType === 1) {
    const style = getComputedStyle(current);
    const alpha = Number(style.opacity);
    if (Number.isFinite(alpha)) opacity *= alpha;
    if (alpha < 0.99 || current.hasAttribute("data-stale")) {
      chain.push({
        element: current.tagName,
        testid: current.getAttribute("data-testid"),
        stale: current.getAttribute("data-stale"),
        opacity: style.opacity,
      });
    }
    current = current.parentElement;
  }
  const rect = element.getBoundingClientRect();
  const top = document.elementFromPoint(rect.left + rect.width * 0.55,
                                        rect.top + rect.height * 0.5);
  const blocked = !!top && !(top === element ||
                              element.contains(top) ||
                              top.contains(element));
  return {opacity, blocked, chain};
}"""


def visible_title_state(frame) -> dict:
    return frame.locator("h1").first.evaluate(VISUAL_STATE_JS)


def wait_bright(frame, page, context: str) -> None:
    last = None
    # Streamlit can temporarily fade widgets DURING a rerun. What must
    # never happen is a persistent greyed-out, unusable screen afterwards.
    for _ in range(24):
        last = visible_title_state(frame)
        if last["opacity"] >= .95 and not last["blocked"]:
            return
        page.wait_for_timeout(450)
    raise AssertionError(f"Page remained dimmed after {context}: {last}")


def edit_number(frame, page, label: str, value: int) -> None:
    field = frame.get_by_role("spinbutton", name=label)
    field.fill(str(value))
    field.press("Tab")
    page.wait_for_timeout(250)
    wait_bright(frame, page, f"{label}={value}")


def edit_time(frame, page, value: str) -> None:
    field = frame.get_by_role("textbox", name="Time remaining (MM:SS)")
    field.fill(value)
    field.press("Tab")
    page.wait_for_timeout(250)
    wait_bright(frame, page, f"clock={value}")


def edit_site(frame, page, label: str) -> None:
    frame.get_by_text(label, exact=True).first.click()
    page.wait_for_timeout(400)
    wait_bright(frame, page, f"game site={label}")


def edit_yardline(page, frame, value: int) -> None:
    current_field = None
    for _ in range(25):
        for embedded in page.frames:
            try:
                if embedded.locator("#los-input").count() == 1:
                    current_field = embedded
                    break
            except Exception:
                continue
        if current_field is not None:
            break
        page.wait_for_timeout(400)
    assert current_field is not None, "No interactive field for line of scrimmage"
    los_input = current_field.locator("#los-input")
    los_input.fill(str(value))
    los_input.press("Tab")
    page.wait_for_timeout(500)
    wait_bright(frame, page, f"yardline={value}")


def main() -> None:
    await_health()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        failures = []
        page.on("pageerror", lambda error: failures.append(str(error)[:400]))
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
            frame = find_app(page)
            wait_bright(frame, page, "first load")
            print("DIMMING_INITIAL_SCREEN_BRIGHT_PASS", flush=True)

            # The main result should be real and unchanged by UI-only edits.
            frame.get_by_role("button", name="Analyze Fourth Down").click()
            first = await_analysis(frame, page, weather=False)
            assert first["Recommendation"], first
            wait_bright(frame, page, "first official Q4 calculation")
            print("DIMMING_AFTER_REAL_Q4_ANALYSIS_PASS", first, flush=True)

            # Representative late-game scenario (Q4 down 4, 4:59, 4th & 4)
            # followed by repeated user revisions; nothing is recomputed
            # unless Analyze is clicked again.
            edit_number(frame, page, "Yards to go", 4)
            edit_number(frame, page, "Offense score", 27)
            edit_number(frame, page, "Defense score", 31)
            edit_time(frame, page, "04:59")
            edit_yardline(page, frame, 6)
            edit_number(frame, page, "Temperature (°F)", 34)
            edit_number(frame, page, "Wind speed (mph)", 23)
            edit_number(frame, page, "Wind speed (mph)", 0)
            edit_number(frame, page, "Temperature (°F)", 45)
            edit_yardline(page, frame, 65)
            edit_yardline(page, frame, 30)
            edit_site(frame, page, "Home")
            edit_site(frame, page, "Away")
            edit_number(frame, page, "Yards to go", 3)
            edit_number(frame, page, "Offense score", 24)
            edit_number(frame, page, "Defense score", 27)
            edit_time(frame, page, "07:38")
            select_choice(frame, page, "Stadium / roof", "Fixed dome")
            wait_bright(frame, page, "fixed dome roof change")
            assert frame.get_by_role("spinbutton", name="Temperature (°F)").count() == 0
            select_choice(frame, page, "Stadium / roof", "Outdoor / open-air")
            wait_bright(frame, page, "outdoor roof change")
            assert frame.get_by_role("spinbutton", name="Temperature (°F)").count() == 1

            # Prior result must disappear after edits, while controls are
            # enabled. A full rerun also clears Streamlit's stale opacity.
            assert metric(frame, "Recommendation") is None, (
                "Previous game's probabilities linger after input changes"
            )
            assert frame.get_by_role(
                "button", name="Analyze Fourth Down"
            ).is_enabled(), "Input controls became stuck/disabled"
            assert not failures, "Browser errors after reruns: " + repr(failures)
            wait_bright(frame, page, "final edit")
            print(
                "REPEATED_20_GAME_AND_WEATHER_EDITS_NO_STALE_DIMMING_PASS",
                visible_title_state(frame), flush=True,
            )
        finally:
            try:
                page.screenshot(path=str(SCREENSHOT), full_page=True)
            except Exception as exc:
                print("DIMMING_SCREENSHOT_ERROR", repr(exc), flush=True)
            browser.close()


if __name__ == "__main__":
    main()
