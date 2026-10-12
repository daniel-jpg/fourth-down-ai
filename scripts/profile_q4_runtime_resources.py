"""Profile the shipped Q4 engine's runtime without modifying frozen artifacts.

Uses historical frozen model files already in this repository, not future
outcomes or protected validation scripts. This is diagnostic only.
"""
from __future__ import annotations

import contextlib
import gc
import importlib.util
import io
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "src" / "42_build_decision_engine.py"


def memory_mib() -> tuple[float | None, float]:
    current = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) / 1024.0
                break
    except FileNotFoundError:
        pass
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return current, peak_mib


def mark(label: str, began: float) -> None:
    current, peak = memory_mib()
    now = time.monotonic() - began
    print(
        f"PROFILE {label} elapsed_s={now:.2f} "
        f"rss_mib={current if current is not None else 'unknown'} "
        f"peak_rss_mib={peak:.1f}",
        flush=True,
    )


def main() -> None:
    began = time.monotonic()
    mark("start", began)

    spec = importlib.util.spec_from_file_location("fourth_down_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen fourth-down engine")
    engine = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(engine)
    mark("engine_loaded", began)

    from src.streamlit_kickoff_preview import (
        isolated_engine_from_loaded_engine,
        preview_native_kickoff_feature_candidate,
    )
    mark("q4_preview_imported", began)

    base_state = {
        "qtr": 4,
        "game_seconds_remaining": 458.0,
        "yardline_100": 30,
        "ydstogo": 3.0,
        "score_differential": -3.0,
        "posteam_timeouts_remaining": 3.0,
        "defteam_timeouts_remaining": 3.0,
        "roof": "outdoors",
    }
    for site in ("AWAY", "HOME"):
        state = dict(base_state, site=site)
        before = time.monotonic()
        private = isolated_engine_from_loaded_engine(engine)
        mark(f"{site}_namespace_ready", began)
        result = preview_native_kickoff_feature_candidate(
            state, simulations=1200, action_seed=43, isolated_engine=private
        )
        seconds = time.monotonic() - before
        rows = result["results"]
        valid = rows.loc[
            rows["eligible"]
            & ~rows["action"].astype(str).str.startswith("FAKE_")
        ]
        if valid.empty or not valid["expected_win_probability"].notna().all():
            raise RuntimeError(f"{site}: frozen normal-action outputs incomplete")
        winner = valid.sort_values("expected_win_probability", ascending=False).iloc[0]
        print(
            f"PROFILE {site}_Q4_SUCCESS simulation_s={seconds:.2f} "
            f"best_action={winner['action']}",
            flush=True,
        )
        mark(f"{site}_q4_complete", began)
        del result, rows, valid, private
        gc.collect()

    print("PROFILE COMPLETE: no models or protected datasets changed", flush=True)


if __name__ == "__main__":
    main()
