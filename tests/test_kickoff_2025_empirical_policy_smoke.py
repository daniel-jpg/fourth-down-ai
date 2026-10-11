"""Real frozen 2025 donor + fourth-down decision integration smoke.

Run against the versioned CSV, rather than synthetic examples or a 2024
return pool. Tests correctness and isolation, NOT forecast validity.
"""

import numpy as np
import pandas as pd
import pytest

from src.kickoff_2025_empirical import (
    FROZEN_PATH, load_frozen_pool,
)
from src.streamlit_kickoff_preview import (
    isolated_engine_for_preview,
    preview_observed_2025_recommendation,
    action_comparison, preview_rank_with_uncertainty,
)


RAMS = {
    "qtr": 4, "game_seconds_remaining": 299.,
    "yardline_100": 6., "ydstogo": 4.,
    "score_differential": -4.,
    "posteam_timeouts_remaining": 3.,
    "defteam_timeouts_remaining": 3.,
    "site": "AWAY", "roof": "outdoors",
}


def test_committed_2025_ordinary_pool_has_auditable_rule_aware_distribution():
    pool = load_frozen_pool()
    assert FROZEN_PATH.is_file()
    assert len(pool) == 488
    assert pool["game_id"].nunique() == 251
    tb = pool["transition_type"].eq("touchback")
    assert tb.sum() == 156
    assert set(pool.loc[tb, "own_yardline"]).issubset({20., 35.})
    assert len(pool.loc[~tb]) == 332
    assert pool["season"].eq(2025).all()


def test_real_rams_decision_integrates_2025_donors_without_mutating_engine():
    untouched = isolated_engine_for_preview()
    scorer = untouched["recommend"]
    original_evaluator = untouched["evaluate_post_play_states"]
    production = scorer(RAMS, n_simulations=1200, seed=42)
    scenario = preview_observed_2025_recommendation(
        RAMS, simulations=2400, action_seed=42,
    )
    # Same isolated production engine is unchanged after preview runs.
    assert untouched["evaluate_post_play_states"] is original_evaluator
    baseline_again = scorer(RAMS, n_simulations=1200, seed=42)
    cols = ["action", "expected_win_probability", "eligible"]
    pd.testing.assert_frame_equal(
        production["results"][cols],
        baseline_again["results"][cols],
        check_exact=True,
    )

    rows = action_comparison(production, scenario)
    assert set(rows["Action"]) == {"FIELD GOAL", "PASS / DROPBACK"}
    assert rows[["Production WP (%)", "2025+ scenario WP (%)"]].notna().all().all()
    assert np.isfinite(rows["2025+ scenario WP (%)"]).all()
    assert rows["2025+ scenario WP (%)"].between(0, 100).all()
    fg = rows.loc[rows["Action"].eq("FIELD GOAL")].iloc[0]
    # Sanity check that the new kickoff resolver actually affects the made-FG
    # continuation in a consequential, late-game state.
    assert fg["Production WP (%)"] - fg["2025+ scenario WP (%)"] > 2.
    rank = preview_rank_with_uncertainty(scenario)
    assert rank["leader"] in {"FIELD GOAL", "PASS / DROPBACK"}
    assert isinstance(rank["near_tie"], bool)
