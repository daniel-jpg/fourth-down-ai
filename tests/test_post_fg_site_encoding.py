"""Verify exact made-FG kickoff site attribution without action-level claims."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.kickoff_2025_resolved import load_resolved_2025_pool
from src.streamlit_kickoff_preview import isolated_engine_for_preview

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"src/79_audit_post_fg_site_encoding.py"
spec=importlib.util.spec_from_file_location("site_attribution_79",SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_every_donor_exactly_once_never_random():
    cycle=mod.CompleteDonorCycle(538)
    assert np.array_equal(cycle.integers(0,538,size=538),np.arange(538))
    with pytest.raises(ValueError,match="every donor"):
        cycle.integers(0,538,size=200)


def test_conditional_made_fg_uses_same_clock_score_and_538_donors():
    engine=isolated_engine_for_preview()
    donors=load_resolved_2025_pool()
    away=mod.one_site("AWAY",engine,donors)
    home=mod.one_site("HOME",engine,donors)
    neutral=mod.one_site("NEUTRAL",engine,donors)

    for res in (away,home,neutral):
        assert res["post_fg_clock_seconds"]==453.
        assert res["post_fg_original_team_score_margin"]==0.
        assert res["donors"]==538
        assert res["simulations_or_random_sampling"] is False
        assert res["is_kickoff_feature_at_pending"]==1
        assert 0<=res["synthetic_pending_win_probability"]<=1
        assert 0<=res["all_538_first_snap_mean_win_probability"]<=1
        assert res["total_first_snap_minus_synthetic_pp"]==pytest.approx(
            res["synthetic_to_native_difference_pp"]
            +res["native_to_first_snap_difference_pp"]
        )
    assert away["receiving_team_home_axis"]==1
    assert home["receiving_team_home_axis"]==0
    assert home["model_coordinate_original_home_axis"]==1
    assert neutral["model_coordinate_original_home_axis"]==1

    by_site=mod.combine([away,home,neutral])
    assert set(by_site)=={
        "synthetic_pending_win_probability",
        "native_like_pending_win_probability",
        "all_538_first_snap_mean_win_probability",
    }
    # At tied score and symmetric 3/3 timeouts, neutral is average of
    # mirrored home/away model coordinates. Check actual model symmetry.
    for data in by_site.values():
        assert data["neutral_minus_home_away_average_pp"]==pytest.approx(
            0.,abs=1e-7,
        )


def test_input_selection_never_uses_2026_or_touches_production():
    s=SCRIPT.read_text()
    assert "load_resolved_2025_pool()" in s
    assert "isolated_engine_for_preview()" in s
    assert "not_action_level_fourth_down_value" in s
    assert "2026_outcomes_used" in s
    assert "nfl.load_pbp([2026])" not in s
    assert "joblib.dump(" not in s
    assert "app.py" not in s
