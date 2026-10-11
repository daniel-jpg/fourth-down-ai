"""Step 72 guards the actual post-made-FG context and 2025 donor state mapping."""

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest

from src.kickoff_2025_empirical import (
    load_frozen_pool, resolve_post_score_q4_2025,
)

SCRIPT=Path(__file__).resolve().parents[1]/"src/72_audit_post_made_fg_wp.py"
spec=importlib.util.spec_from_file_location("step72_post_fg",SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def synthetic_actions():
    def row(game,play,kind,team,fg=None):
        return dict(season=2025,season_type="REG",game_id=game,
                    play_id=float(play),play_type=kind,posteam=team,
                    field_goal_result=fg,qtr=4)
    return pd.DataFrame([
        row("g1",1,"field_goal","RAM","made"),
        row("g1",2,"kickoff","SEA"),
        row("g1",3,"run","SEA"),
        row("g1",4,"kickoff","RAM"),
        row("g2",1,"field_goal","JAX","missed"),
        row("g2",2,"kickoff","HOU"),
        row("g3",1,"field_goal","LAR","made"),
        row("g3",2,"no_play","LAR"),
        row("g3",3,"kickoff","SEA"),
        row("g4",1,"field_goal","NYJ","made"),
        row("g4",2,"run","NYJ"),
        row("g4",3,"kickoff","NE"),
        row("g5",1,"field_goal","LAR","made"),
        row("g5",2,"kickoff","LAR"),
    ])


def test_only_next_opponent_kickoffs_after_actual_made_field_goals():
    found=mod.kickoff_after_made_field_goal(synthetic_actions())
    assert set(zip(found["game_id"], found["play_id_kick"])) == {
        ("g1",2.), ("g3",3.)
    }
    assert found.loc[found["game_id"].eq("g1"),
                     "preceding_made_fg_play_id"].iloc[0] == 1.


def test_preceding_scoring_action_cannot_leak_between_games():
    data=synthetic_actions().iloc[:2].copy()
    other=pd.DataFrame([{
        "season":2025,"season_type":"REG",
        "game_id":"other","play_id":1.,"play_type":"kickoff",
        "posteam":"B","field_goal_result":None,"qtr":4
    }])
    found=mod.kickoff_after_made_field_goal(pd.concat([data,other]))
    assert len(found)==1
    bad=synthetic_actions().assign(season=2026)
    with pytest.raises(ValueError,match="Only 2024-25"):
        mod.kickoff_after_made_field_goal(bad)


def test_report_is_observed_action_only_and_flags_low_support():
    observed=pd.DataFrame({
        "game_id":["g1","g1","g2"],
        "actual_home_win":[1.,0.,1.],
        "p_kickoff_synthetic":[.5,.4,.7],
        "p_kickoff_native":[.6,.3,.8],
        "p_first_snap":[.6,.3,.9],
    })
    ci=lambda delta,games:(float(delta.min()),float(delta.max()))
    score=mod.grouped_observed_scoring_calibration(
        "chosen only",observed,cluster_ci=ci)
    assert score["n"]==3
    assert score["games"]==2
    assert score["status"]=="LOW SUPPORT"
    assert score["pending_brier"]==pytest.approx(
        np.mean((observed["actual_home_win"]-
                 observed["p_kickoff_synthetic"])**2))
    assert score["native_pending_brier"]==pytest.approx(
        np.mean((observed["actual_home_win"]-
                 observed["p_kickoff_native"])**2))
    assert score["native_vs_synthetic_brier_improvement"]==pytest.approx(
        score["pending_brier"]-score["native_pending_brier"]
    )
    none=mod.grouped_observed_scoring_calibration(
        "empty",observed.iloc[:0],cluster_ci=ci)
    assert none["status"]=="NO DATA"


def test_exact_snapshots_equal_2025_real_scenario_resolver_first_donor():
    engine=mod.import_script(mod.ENGINE_PATH,"step72_real_engine_test")
    pool=load_frozen_pool()
    base=engine.normalize_state(mod.RAMS_EXAMPLE)
    pending=engine.make_wp_state(
        base,False,np.nan,np.nan,np.nan,
        elapsed_seconds=5.,score_change_original=3.,is_kickoff=True,
    )
    generated=mod.all_frozen_donor_snapshots(pending,pool)
    assert len(generated)==len(pool)
    assert all(s["is_kickoff"]==0 and s["down"]==1 for s in generated)
    assert all(0<=s["game_seconds_remaining"]<=pending["game_seconds_remaining"]
               for s in generated)
    assert all(s["is_home_posteam"]==pending["is_home_posteam"]
               for s in generated)

    class ControlledRng:
        def integers(self,lo,hi,size):
            assert lo==0 and hi==len(pool) and size==1
            return np.array([0])

    actual=resolve_post_score_q4_2025([pending],pool,
                                       rng=ControlledRng())[0]
    for key in [
        "is_kickoff","yardline_100","down","ydstogo","goal_to_go",
        "game_seconds_remaining","half_seconds_remaining",
        "final_five_minutes","is_home_posteam","_kickoff_transition_type",
    ]:
        assert generated[0][key]==actual[key]


def test_rams_exact_decomposition_conserves_probability_mixture():
    engine=mod.import_script(mod.ENGINE_PATH,"step72_model_attribution")
    pool=load_frozen_pool()
    result=mod.exact_made_fg_decomposition(
        engine.__dict__,pool,mod.RAMS_EXAMPLE)
    assert result["ordinary_donors"]==488
    assert result["year_of_donors"]==2025
    assert 0<result["native_like_pending_wp"]<1
    assert (
        result["synthetic_to_native_encoding_difference"]
        + result["native_encoding_to_post_snap_difference"]
    )==pytest.approx(result["difference_integrated_minus_pending"])
    assert result["post_fg_margin_for_kicker"]==-1.
    assert 0<result["touchback_rate"]<1
    assert 0<result["kickoff_pending_wp"]<1
    assert 0<result["ordinary_kickoff_integrated_wp"]<1
    # Real previously observed large difference cannot silently disappear.
    assert result["kickoff_pending_wp"] - (
        result["ordinary_kickoff_integrated_wp"]
    )>.03


def test_never_reads_2026_or_changes_live_models():
    source=SCRIPT.read_text()
    assert 'SEASONS = (2024, 2025)' in source
    assert "nfl.load_pbp([year])" in source
    assert "load_pbp([2026])" not in source
    assert "joblib.dump(" not in source
    assert "app.py" not in source
