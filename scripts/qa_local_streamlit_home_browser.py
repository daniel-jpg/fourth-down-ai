"""Real local Streamlit/Chromium smoke: Q4 HOME analysis through the browser.

Unlike Streamlit AppTest, this runs the actual server and tests the widget
websocket, HOME radio event, full frozen engine, and rendered recommendation.
It uses only shipped historical frozen artifacts; no future NFL data or
experimental model promotion. No application model code is modified.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8531
# CI may reuse the same browser assertions against an already-running
# Streamlit process inside a portable Docker container.
BASE_URL = os.environ.get(
    "FOURTH_DOWN_SMOKE_BASE_URL", f"http://127.0.0.1:{PORT}"
).rstrip("/")
HEALTH_URL = f"{BASE_URL}/_stcore/health"
SCREENSHOT = ROOT / "local-streamlit-q4-home.png"


def wait_for_server(process: subprocess.Popen | None, seconds: int = 90) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Streamlit server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
                if response.status == 200:
                    print("LOCAL_SERVER_HEALTHY", flush=True)
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise AssertionError("Local Streamlit health endpoint did not start")


def find_app(page, seconds: int = 90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for frame in page.frames:
            try:
                if frame.get_by_text("Game Situation", exact=True).count() == 1:
                    return frame
            except Exception:
                pass
        page.wait_for_timeout(1000)
    raise AssertionError(
        "Local Streamlit page did not show its game inputs; "
        + str([(f.name, f.url) for f in page.frames])
    )


def run_browser() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda error: print(
            "CLIENT_ERROR", str(error)[:350], flush=True
        ))
        page.on("websocket", lambda ws: print(
            "LOCAL_WEBSOCKET", ws.url.split("?")[0], flush=True
        ))
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=45000)
            frame = find_app(page)
            print("LOCAL_FORM_READY", flush=True)

            weather = frame.get_by_text("Weather observations (optional)", exact=True)
            assert weather.count() == 1, "Latest weather UI missing from local app"

            # The default Q4 state is 7:38, 4th-and-3 at opponent 30,
            # down 24-27. Change the site as real users do.
            home = frame.get_by_role("radio", name="Home")
            # The input may register one render tick after the page header.
            # Clicking the visible label auto-waits for a stable control;
            # don't assert a transient .count() before it has mounted.
            frame.get_by_text("Home", exact=True).first.click(timeout=20000)
            page.wait_for_timeout(3000)
            home.wait_for(state="attached", timeout=15000)
            assert home.is_checked(), "HOME radio failed to persist after rerun"
            print("LOCAL_HOME_CONFIRMED", flush=True)

            analyze = frame.get_by_role("button", name="Analyze Fourth Down")
            assert analyze.count() == 1 and analyze.is_enabled(), (
                "Analyze Fourth Down is disabled or missing"
            )
            started = time.monotonic()
            analyze.click(timeout=15000)
            print("LOCAL_ANALYZE_CLICKED", flush=True)

            # A constrained free host can take much longer than local Docker.
            # Override ONLY the browser wait, never the simulation budget.
            analyze_limit = int(os.environ.get(
                "FOURTH_DOWN_ANALYZE_TIMEOUT_SECONDS", "100"
            ))
            assert 30 <= analyze_limit <= 360, "Invalid browser time limit"
            deadline = time.monotonic() + analyze_limit
            consecutive = 0
            last_body = ""
            while time.monotonic() < deadline:
                last_body = frame.locator("body").inner_text(timeout=5000)
                has_result = (
                    "Recommendation" in last_body
                    and "Win probability if chosen" in last_body
                    and "Q4 estimates use corrected standard-35 kickoff inputs"
                    in last_body
                )
                if (
                    "Older percentages have NOT been silently substituted"
                    in last_body
                ):
                    raise AssertionError("Local Q4 native estimate failed")
                consecutive = consecutive + 1 if has_result else 0
                if consecutive >= 2:
                    print(
                        "LOCAL_Q4_HOME_RESULT_PASS", {
                            "seconds": round(time.monotonic() - started, 2),
                            "close_call": "Close call" in last_body,
                            "corrected_q4": True,
                        }, flush=True,
                    )
                    return
                page.wait_for_timeout(1200)
            raise AssertionError(
                f"Q4 HOME analysis failed to render in {analyze_limit} s. "
                "Body tail: " + repr(last_body[-850:])
            )
        finally:
            try:
                page.screenshot(path=str(SCREENSHOT), full_page=True)
            except Exception as exc:
                print("SCREENSHOT_ERROR", str(exc)[:160], flush=True)
            browser.close()


def main() -> None:
    if os.environ.get("FOURTH_DOWN_SMOKE_BASE_URL"):
        wait_for_server(None, seconds=120)
        print("EXTERNAL_STREAMLIT_SERVER_READY", BASE_URL, flush=True)
        run_browser()
        return

    with tempfile.TemporaryDirectory(prefix="fourth-down-local-smoke-") as tmp:
        log_path = Path(tmp) / "streamlit.log"
        with log_path.open("w", encoding="utf-8") as logfile:
            command = [
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.headless=true",
                f"--server.port={PORT}",
                "--server.address=127.0.0.1",
                "--browser.gatherUsageStats=false",
            ]
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                wait_for_server(process)
                run_browser()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print("LOCAL_SERVER_LOG_TAIL", "\n".join(lines[-35:]), flush=True)


if __name__ == "__main__":
    main()
