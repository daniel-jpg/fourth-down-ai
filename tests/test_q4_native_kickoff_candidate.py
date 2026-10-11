"""Research-only Q4 native kickoff feature repair, strict isolation and parity."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.kickoff_q4_native_features_candidate import (
    NATIVE_STANDARD_35,native_q4_normal_kickoff,
    use_q4_native_kickoff_candidate,
)
from src.streamlit_kickoff_preview import isolated_engine_for_preview

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"src/80_evaluate_q4_native_kickoff_candidate.py"
spec=importlib.util.spec_from_file_location("native_q4_candidate_80",SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def raw_pending(**kwargs):
    s={
        "qtr":4, "is_kickoff":1., "game_seconds_remaining":453.,
        "_score_change_original":3.,
        "down":np.nan, "ydstogo":np.nan, "yardline_100":np.nan,
        "goal_to_go":0., "is_home_posteam":1.,
    }
    s.update(kwargs)
    return s


def test_only_normal_q4_post_score_missing_features_are_reencoded():
    row=raw_pending()
    native=native_q4_normal_kickoff(row)
    assert native["yardline_100"]==NATIVE_STANDARD_35==35.
    assert native["ydstogo"]==0.
    assert np.isnan(native["down"])
    assert native["is_home_posteam"]==row["is_home_posteam"]
    assert native["_score_change_original"]==3.
    assert np.isnan(row["yardline_100"])
    for variation in (
        {"qtr":3},
        {"is_kickoff":0.},
        {"game_seconds_remaining":0.},
        {"_score_change_original":0.},
        {"_kickoff_from_50":True},
        {"yardline_100":35.},
        {"ydstogo":0.},
    ):
        changed=raw_pending(**variation)
        out=native_q4_normal_kickoff(changed)
        assert set(out)==set(changed)
        assert (
            not np.isfinite(out["yardline_100"])
            if np.isnan(changed["yardline_100"])
            else out["yardline_100"]==changed["yardline_100"]
        )


def test_native_patch_equals_direct_native_encoded_wp_for_made_fg():
    engine=isolated_engine_for_preview()
    original=engine["evaluate_post_play_states"]
    for site in ("AWAY","HOME","NEUTRAL"):
        base=engine["normalize_state"]({
            "qtr":4,"game_seconds_remaining":458.,
            "yardline_100":30.,"ydstogo":3.,
            "score_differential":-3.,
            "posteam_timeouts_remaining":3.,
            "defteam_timeouts_remaining":3.,
            "site":site,"roof":"outdoors",
        })
        pending=engine["make_wp_state"](
            base,False,np.nan,np.nan,np.nan,
            elapsed_seconds=5.,score_change_original=3.,is_kickoff=True,
        )
        directly=engine["evaluate_post_play_states"](
            base,[native_q4_normal_kickoff(pending)]
        )[0]
        with use_q4_native_kickoff_candidate(engine):
            patched=original.__globals__["evaluate_post_play_states"](
                base,[pending]
            )[0]
            assert patched==pytest.approx(directly,abs=1e-12)
        assert original.__globals__["evaluate_post_play_states"] is original


def test_nested_wrapper_fails_and_always_restores_private_model():
    engine=isolated_engine_for_preview()
    original=engine["evaluate_post_play_states"]
    with pytest.raises(RuntimeError,match="explicit failure"):
        with use_q4_native_kickoff_candidate(engine):
            with pytest.raises(RuntimeError,match="Nested"):
                with use_q4_native_kickoff_candidate(engine):
                    pass
            raise RuntimeError("explicit failure")
    assert original.__globals__["evaluate_post_play_states"] is original


def test_predeclared_grid_is_reused_not_optimized_after_results():
    step=mod.step78()
    cases=step.build_states()
    assert len(cases)==14
    assert cases[0][0]=="reference"
    assert cases[-1][0]=="neutral_site"
    assert mod.SIMULATIONS==1200
    assert mod.SEED==42


def test_native_only_candidate_outputs_valid_win_probabilities():
    step=mod.step78()
    state=step.build_states()[0][1]
    candidate=mod.model_native_only(state)
    snap=step.snapshot(candidate)
    assert snap["supported"]
    assert 0<=snap["go_wp"]<=1
    assert 0<=snap["fg_wp"]<=1
    assert snap["top_strategy"] in ("GO FOR IT","FIELD GOAL","PUNT")


def test_no_2026_data_or_production_files_edited():
    src=SCRIPT.read_text()
    assert "models_frozen" not in src or "frozen" in src.lower()
    assert "2026_outcomes_read" in src
    assert '"promotion":"NOT AUTHORIZED"' in src
    assert "nfl.load_pbp([2026])" not in src
    assert "joblib.dump(" not in src
    assert "app.py" not in src
