"""Cross-quarter weather isolation QA using the actual frozen engine.

Only synthetic pre-decision states and previously inspected 2014-2024
development models/weather coefficients are loaded. The public policy is
not updated; no outcomes, future-season files, network or retraining.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
    preview_native_kickoff_feature_candidate,
)
from src.weather_fg_sensitivity import (
    compare_weather_scenario,
    load_weather_research,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/qa_weather_regulation_quarters.json"
N = 1200
SEED = 43


def make_state(
    *, qtr: int, seconds_in_quarter: int, yards_to_goal: int,
    ydstogo: int, score: int, site: str, roof: str = "outdoors",
    receiver: str | None = None,
) -> dict:
    state = {
        "qtr": qtr,
        "game_seconds_remaining": float(
            (4 - qtr) * 900 + seconds_in_quarter
        ),
        "yardline_100": float(yards_to_goal),
        "ydstogo": float(ydstogo),
        "score_differential": float(score),
        "posteam_timeouts_remaining": 3.0,
        "defteam_timeouts_remaining": 3.0,
        "site": site,
        "roof": roof,
    }
    if receiver is not None:
        state["second_half_receiver"] = receiver
    return state


# Synthetic situations, not logged plays or future-season outcomes.
SCENARIOS = (
    {
        "name": "Q1 tied, opponent 30, outdoor HOME",
        "state": make_state(
            qtr=1, seconds_in_quarter=600, yards_to_goal=30,
            ydstogo=3, score=0, site="HOME",
        ),
        "temperature_f": 45, "wind_mph": 25, "conditions": "Rain",
    },
    {
        "name": "Q2 final 25 seconds, opponent 22, AWAY",
        "state": make_state(
            qtr=2, seconds_in_quarter=25, yards_to_goal=22,
            ydstogo=7, score=-3, site="AWAY", receiver="DEFENSE",
        ),
        "temperature_f": 27, "wind_mph": 20, "conditions": "Snow",
    },
    {
        "name": "Q3 trailing 7, opponent 30, retractable roof OPEN",
        "state": make_state(
            qtr=3, seconds_in_quarter=375, yards_to_goal=30,
            ydstogo=4, score=-7, site="NEUTRAL", roof="open",
        ),
        "temperature_f": -5, "wind_mph": 30, "conditions": "Fog",
    },
    {
        "name": "Q4 down 3, opponent 30, HOME",
        "state": make_state(
            qtr=4, seconds_in_quarter=458, yards_to_goal=30,
            ydstogo=3, score=-3, site="HOME",
        ),
        "temperature_f": 32, "wind_mph": 25, "conditions": "Snow",
    },
    {
        "name": "Q4 down 4, opponent 6, AWAY",
        "state": make_state(
            qtr=4, seconds_in_quarter=299, yards_to_goal=6,
            ydstogo=4, score=-4, site="AWAY",
        ),
        "temperature_f": 35, "wind_mph": 25, "conditions": "Rain",
    },
)


def frozen_answer(engine: dict, state: dict) -> dict:
    if state["qtr"] == 4:
        # Matches the app's original corrected standard-35 Q4 kickoff path.
        private = isolated_engine_from_loaded_engine(engine)
        return preview_native_kickoff_feature_candidate(
            dict(state), simulations=N, action_seed=SEED,
            isolated_engine=private,
        )
    # App's regulation Q1-Q3 path, with the full original per-action budget.
    return engine["recommend"](
        dict(state), n_simulations=N, seed=SEED,
    )


def assert_same_frozen_answer(before: dict, after: dict, name: str) -> None:
    for key in ("recommendation", "pre_decision_wp"):
        assert before[key] == after[key], (
            f"{name}: weather research changed frozen engine {key}"
        )
    columns = (
        "action", "eligible", "expected_win_probability", "mc_se",
    )
    a = before["results"].loc[:, columns].set_index("action").sort_index()
    b = after["results"].loc[:, columns].set_index("action").sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=True, obj=name)


def run() -> dict:
    research = load_weather_research()
    assert research["validation_supports_official_weather_use"] is False
    assert research["brier_improvement_baseline_minus_weather"] < 0
    assert research["2025_heldout_data_used"] is False
    assert research["2026_or_2027_data_used"] is False

    engine = isolated_engine_for_preview()
    original_recommend = engine["recommend"]
    original_fg = engine["field_goal_probabilities"]
    original_wp = engine["evaluate_post_play_states"]
    rows = []
    for case in SCENARIOS:
        state = dict(case["state"])
        frozen_before = frozen_answer(engine, state)
        observation = {
            "roof": state["roof"],
            "temperature_f": case["temperature_f"],
            "wind_mph": case["wind_mph"],
            "conditions": case["conditions"],
            "wind_relative_direction": "Unknown",
        }
        experimental = compare_weather_scenario(
            dict(state), weather=observation, live_engine=engine,
            simulations=N, seed=SEED,
        )
        frozen_after = frozen_answer(engine, state)

        assert_same_frozen_answer(
            frozen_before, frozen_after, case["name"]
        )
        assert engine["recommend"] is original_recommend
        assert engine["field_goal_probabilities"] is original_fg
        assert engine["evaluate_post_play_states"] is original_wp
        assert original_recommend.__globals__["field_goal_probabilities"] is original_fg
        assert case["state"] == state, "Synthetic input mutated"

        assert experimental["applicable"] is True, case["name"]
        assert experimental["official_model_unchanged"] is True
        assert experimental["model_retrained"] is False
        assert experimental["real_2027_data_accessed"] is False
        assert experimental["go_and_punt_weather_adjusted"] is False
        assert experimental["adjusted_rain_snow_numeric_effect"] is False
        assert experimental["wind_direction_measured"] is False
        assert experimental["validation_shows_brier_improvement"] is False
        assert experimental["data_source"] == (
            "NFLVERSE_2014_2024_DEVELOPMENT_ONLY"
        )
        before_make = experimental["original_made_probability"]
        after_make = experimental["weather_made_probability"]
        fg_wp = experimental["weather_fg_win_probability"]
        assert (
            math.isfinite(before_make) and
            math.isfinite(after_make) and
            math.isfinite(fg_wp)
        )
        assert 0 <= before_make <= 1 and 0 <= after_make <= 1
        assert after_make < before_make, (
            f"{case['name']}: negative cold/wind contrast should lower FG make"
        )
        assert 0 <= fg_wp <= 1
        assert experimental["applied_log_odds_change"] < 0

        entry = {
            "case": case["name"],
            "quarter": state["qtr"],
            "roof": state["roof"],
            "site": state["site"],
            "official_recommendation": frozen_after["recommendation"],
            "official_pre_decision_wp": frozen_after["pre_decision_wp"],
            "official_output_unchanged": True,
            "original_fg_made": round(before_make, 6),
            "research_fg_made": round(after_make, 6),
            "research_fg_action_wp": round(fg_wp, 6),
            "research_validation_failed": True,
            "original_simulations_per_action": N,
        }
        rows.append(entry)
        print(
            f"PASS {entry['case']}: official {entry['official_recommendation']}; "
            f"FG make {100*before_make:.2f}% -> {100*after_make:.2f}% "
            f"(research only); q={state['qtr']}",
            flush=True,
        )

    assert {r["quarter"] for r in rows} == {1, 2, 3, 4}
    assert any(r["roof"] == "open" for r in rows)
    report = {
        "status": "PASS_WEATHER_FROZEN_MODEL_CROSS_QUARTER_ISOLATION",
        "scope": "Synthetic inputs with frozen 2014-2024 development only",
        "quarters": [1, 2, 3, 4],
        "cases": rows,
        "simulations_per_action": N,
        "seed": SEED,
        "official_model_mutated": False,
        "model_weights_retrained": False,
        "future_2026_or_2027_outcomes_accessed": False,
        "validates_weather_coaching_policy": False,
        "weather_brier_improved": False,
    }
    return report


def main() -> None:
    report = run()
    REPORT_PATH.parent.mkdir(exist_ok=True, parents=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: {len(report['cases'])} cold/windy regulation-quarter "
        "scenarios without mutation of frozen official recommendations. "
        "Weather remains unvalidated FG-only research.",
        flush=True,
    )
    print("Audit output:", REPORT_PATH.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
