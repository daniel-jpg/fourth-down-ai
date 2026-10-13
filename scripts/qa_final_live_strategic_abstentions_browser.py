"""Browser proof that real hosted Fourth Down AI abstains on two model hazards.

Uses full live 1,200-draw Q4 official calculations; NO input shortcuts,
model substitutions, web scraping of football outcomes, or paid resources.
Both original estimates must remain visible while the recommendation
heading reads INSUFFICIENT DATA. Includes the already-deployed brand title.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from scripts.qa_weather_streamlit_browser import (
    BASE_URL, await_health, find_app, await_analysis,
    body_text, metric, pct, set_number,
)
from scripts.qa_repeated_input_browser import (
    edit_time, edit_yardline, edit_site, wait_bright,
)

SCREENSHOT = Path("live-new-strategic-abstentions.png")


def choose_timeout(frame, page, label: str, choice: int) -> None:
    group = frame.locator('[data-testid="stRadio"]').filter(
        has=frame.get_by_text(label, exact=True)
    )
    assert group.count() == 1, f"Radio group {label}: {group.count()}"
    item = group.get_by_text(str(choice), exact=True)
    assert item.count() >= 1, f"Missing {label}={choice}"
    item.first.click(timeout=20000)
    page.wait_for_timeout(450)
    wait_bright(frame, page, f"{label}={choice}")


def run_scenario(frame, page, *,
                 name: str, expected_wp: dict[str, float],
                 warning: str) -> None:
    button = frame.get_by_role("button", name="Analyze Fourth Down")
    button.wait_for(state="visible", timeout=30000)
    assert button.is_enabled()
    button.click(timeout=20000)
    # Two back-to-back analyses may both display INSUFFICIENT DATA. Do NOT
    # mistake the preceding scenario's cached result for the new one.
    # Wait for the new numerical fingerprint AND its new situation-specific
    # warning, not just a generic Recommendation metric.
    deadline = time.monotonic() + 210
    start = time.monotonic()
    last_print = -20
    verified = 0
    last = {}
    while time.monotonic() < deadline:
        body = body_text(frame)
        recommendation = metric(frame, "Recommendation")
        values = {}
        for label in expected_wp:
            raw = metric(frame, label)
            values[label] = pct(raw) if raw is not None else None
        nums_match = all(
            values[label] is not None and abs(values[label] - expected) <= .08
            for label, expected in expected_wp.items()
        )
        state_ok = recommendation == "INSUFFICIENT DATA"
        warning_ok = warning in body
        edge_ok = "Modeled edge over " in body
        last = {
            "recommendation": recommendation, "values": values,
            "state_ok": state_ok, "warning_ok": warning_ok,
            "edge_ok": edge_ok, "body_tail": body[-1600:],
        }
        if all((nums_match, state_ok, warning_ok, edge_ok)):
            verified += 1
            if verified >= 2:
                wait_bright(frame, page, name)
                print("LIVE_NEW_SCENARIO_PASS", {
                    "name": name,
                    "recommendation": recommendation,
                    "verified_wp": values,
                    "warning": warning,
                    "seconds": round(time.monotonic() - start, 1),
                    "site": BASE_URL,
                }, flush=True)
                return
        else:
            verified = 0
        elapsed = time.monotonic() - start
        if elapsed - last_print >= 20:
            last_print = elapsed
            print("LIVE_NEW_SCENARIO_WAIT", {
                "name": name, "elapsed": round(elapsed, 1),
                "recommendation": recommendation, "values": values,
                "warning_ok": warning_ok, "edge_ok": edge_ok,
            }, flush=True)
        page.wait_for_timeout(1300)
    raise AssertionError(
        f"{name}: New scenario never reached matching warning/estimates: {last}"
    )


def main() -> None:
    assert BASE_URL.startswith("https://") and BASE_URL.endswith(".onrender.com")
    await_health()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)[:300]))
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
            frame = find_app(page, timeout=150)
            assert page.title() == "Fourth Down AI"
            wait_bright(frame, page, "initial public-branded load")

            # Newly exposed model issue: at 0:18 while down 4, no timeouts,
            # 4th-and-10 at own 40. Original model ranks PUNT by 1.21 pp.
            # The UI must *not* endorse that as a validated coaching action.
            edit_time(frame, page, "00:18")
            set_number(frame, page, "Yards to go", 10)
            set_number(frame, page, "Offense score", 27)
            set_number(frame, page, "Defense score", 31)
            choose_timeout(frame, page, "Offense timeouts", 0)
            choose_timeout(frame, page, "Defense timeouts", 0)
            edit_site(frame, page, "Away")
            edit_yardline(page, frame, 60)
            run_scenario(
                frame, page,
                name="Q4 0:18 trailing4 own40 4th&10 noTO",
                expected_wp={"PUNT": 2.93, "PASS / DROPBACK": 1.72},
                warning="Insufficient support for a final-minute trailing PUNT",
            )

            # Observed real 2025 ARI-SF Week 3 fourth down, own1 4th&11:
            # GO PASS 37.23% > PUNT 36.51%, but the gap is tiny.
            edit_time(frame, page, "14:11")
            set_number(frame, page, "Yards to go", 11)
            set_number(frame, page, "Offense score", 20)
            set_number(frame, page, "Defense score", 20)
            choose_timeout(frame, page, "Offense timeouts", 2)
            choose_timeout(frame, page, "Defense timeouts", 3)
            edit_site(frame, page, "Away")
            edit_yardline(page, frame, 99)
            run_scenario(
                frame, page,
                name="2025 ARI-SF own1 4th&11 tied Q4 14:11",
                expected_wp={"PASS / DROPBACK": 37.23, "PUNT": 36.51},
                warning="Insufficient support for a long GO deep in own territory",
            )
            assert page.title() == "Fourth Down AI"
            assert not errors, f"Browser exceptions: {errors[:5]}"
            print(
                "LIVE_NEW_ABSTENTIONS_AND_BRANDING_COMPLETE",
                {"passed_scenarios": 2, "page_title": page.title(),
                 "browser_errors": len(errors)}, flush=True,
            )
        finally:
            try:
                page.screenshot(path=str(SCREENSHOT), full_page=True)
            except Exception as exc:
                print("NEW_SAFETY_SCREENSHOT_UNAVAILABLE", repr(exc), flush=True)
            browser.close()


if __name__ == "__main__":
    main()
