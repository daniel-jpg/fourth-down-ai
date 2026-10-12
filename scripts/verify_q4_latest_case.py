"""CI invariant: the Q4 demo's native-input action WP is recalculated.

Uses only checked-in models and a fixed 4:59 / fourth-and-4 game state.
Does not download future data, refit weights or run the Step 66/83 gates.
"""
from __future__ import annotations

import math

from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    isolated_engine_from_loaded_engine,
    preview_native_kickoff_feature_candidate,
)

STATE = {
    "qtr": 4,
    "game_seconds_remaining": 299.0,
    "yardline_100": 6.0,
    "ydstogo": 4.0,
    "score_differential": -4.0,
    "posteam_timeouts_remaining": 3.0,
    "defteam_timeouts_remaining": 3.0,
    "site": "AWAY",
    "roof": "outdoors",
}
N = 1200
SEED = 43
ACTIONS = ["NORMAL_GO_PASS", "NORMAL_GO_RUN", "FIELD_GOAL", "PUNT"]


def main() -> None:
    private = isolated_engine_for_preview()
    legacy = private["recommend"](STATE, n_simulations=N, seed=SEED)
    lightweight = isolated_engine_from_loaded_engine(private)
    updated = preview_native_kickoff_feature_candidate(
        STATE, simulations=N, action_seed=SEED, isolated_engine=lightweight,
    )
    old = legacy["results"].set_index("action")
    new = updated["results"].set_index("action")

    assert old.index.tolist() == new.index.tolist()
    assert legacy["pre_decision_wp"] == updated["pre_decision_wp"]
    print("Q4 4:59, 4th-and-4 opponent 6, trailing 27-31; Away; timeout 3-3")
    print("Same frozen model weights, simulation budget 1200/action, seed 43")
    print("Action           Legacy WP     Corrected native WP  Delta (pp)")
    changed = 0
    for action in ACTIONS:
        assert bool(old.loc[action, "eligible"]) == bool(
            new.loc[action, "eligible"]
        ), f"Eligibility changed unexpectedly: {action}"
        if not bool(new.loc[action, "eligible"]):
            print(f"{action:<16} unavailable   unavailable")
            continue
        a = float(old.loc[action, "expected_win_probability"])
        b = float(new.loc[action, "expected_win_probability"])
        assert math.isfinite(a) and math.isfinite(b) and 0 <= a <= 1 and 0 <= b <= 1
        delta = 100 * (b - a)
        print(f"{action:<16} {100*a:9.2f}%       {100*b:9.2f}%    {delta:+8.2f}")
        if abs(delta) > .1:
            changed += 1
    assert changed >= 1, (
        "Corrected native kickoff representation did not change any "
        "eligible action WP by >0.1 percentage point"
    )
    print("PASS: latest Q4 demo estimates differ from legacy for an eligible action.")
    print("These retrospective input corrections are not a prospectively validated policy.")


if __name__ == "__main__":
    main()
