"""Step 83 test suite: all 2026 game data are synthetic unit-test fixtures.

Do NOT fetch, score, or inspect any actual 2026 W6-18 game in these tests.
The existing pre-registered Step 66 forward validation must fire first.
"""
import importlib.util
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STEP83 = ROOT / "src/83_forward_2026_native_challenger_once.py"
spec = importlib.util.spec_from_file_location("once_challenger_83", STEP83)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
step66 = audit.load_fixed_step66()
helper61 = runpy.run_path(str(ROOT/"src/61_audit_observed_kickoff_wp.py"))
FEATURES = list(helper61["feature_frame"](
    pd.DataFrame({
        f"{name}_kick": [value] for name, value in {
            "game_seconds_remaining": 250, "half_seconds_remaining":250,
            "qtr":4, "down":np.nan, "ydstogo":0, "yardline_100":35,
            "goal_to_go":0, "posteam_score":17, "defteam_score":20,
            "posteam_timeouts_remaining":2, "defteam_timeouts_remaining":3,
            "posteam":"HOME","home_team":"HOME","result":7,
        }.items()
    }), "kick"
).columns)


class FakeWP:
    classes_ = np.array([0, 2])

    def predict_proba(self, matrix):
        yard = matrix[:, FEATURES.index("yardline_100")]
        receiver_is_home = matrix[:, FEATURES.index("is_home_posteam")]
        # The test model deliberately favors the native feature, but the
        # scoring rule must work for ANY result, not only this direction.
        p = np.where(np.isnan(yard), .32, .64)
        p = p + .02 * receiver_is_home
        return np.column_stack([1-p,p])


def mock_play(game, play_id, typ, week, *, receiver="HOME", season=2026,
              home_result=7, sec=250, desc="", onside=False):
    is_kickoff = typ=="kickoff"
    return {
        "game_id": game,
        "play_id": float(play_id),
        "play_type":typ,
        "season":season,
        "season_type":"REG",
        "week":int(week),
        "qtr":4,
        "game_seconds_remaining":float(sec if is_kickoff else sec-4),
        "half_seconds_remaining":float(sec if is_kickoff else sec-4),
        "down":np.nan if is_kickoff else 1.,
        "ydstogo":0. if is_kickoff else 10.,
        "yardline_100":35. if is_kickoff else 65.,
        "goal_to_go":0.,
        "posteam_score":17,
        "defteam_score":20,
        "posteam_timeouts_remaining":2.,
        "defteam_timeouts_remaining":3.,
        "posteam":receiver,
        "home_team":"HOME",
        "touchback":int(is_kickoff),
        "desc":("Onside kick" if onside else desc),
        "result":float(home_result),
    }


def fixture_2026_games(n=110):
    rows=[]
    for i in range(n):
        receiver="HOME" if i%2 else "AWAY"
        game=f"test_only_game_{i:04d}"
        wk=6+i%13
        rows.append(mock_play(game,1,"kickoff",wk,receiver=receiver))
        rows.append(mock_play(game,2,"run",wk,receiver=receiver))
    return pd.DataFrame(rows)


def test_gate_requires_frozen_step66_output_even_before_importing_real_games(tmp_path):
    final=tmp_path/"2026_native.json"
    gate=tmp_path/"2026_gate.txt"
    with pytest.raises(RuntimeError,match="not yet complete"):
        audit.check_existing_gate_report(gate,final)
    gate.write_text("NOT READY: need >= 100 unique games.\n")
    with pytest.raises(RuntimeError,match="incomplete"):
        audit.check_existing_gate_report(gate,final)
    gate.write_text(
        "=== STEP 66: FROZEN 2026 W6-18 FORWARD VALIDATION ===\n"
        "PREDECLARED RESEARCH DECISION: RESEARCH GATE FAIL/INCONCLUSIVE\n"
        "evaluation_commit=abc0123\n"
    )
    assert "PREDECLARED" in audit.check_existing_gate_report(gate,final)
    final.write_text("{}")
    with pytest.raises(RuntimeError,match="already evaluated"):
        audit.check_existing_gate_report(gate,final)


def test_original_source_artifacts_are_still_pinned():
    audit.enforce_frozen_provenance(step66)


def test_prespecified_gate_rejects_99_games_without_scoring():
    before=fixture_2026_games(99)
    with pytest.raises(RuntimeError,match="NOT READY: 99"):
        audit.eligible_pairs_only(before, step66, helper61["FIELDS"])


def test_locked_ordinary_q4_sample_requires_100_unique_games_and_week6_plus():
    earlier=fixture_2026_games(110)
    excluded=pd.DataFrame([
        mock_play("week5",1,"kickoff",5),
        mock_play("week5",2,"run",5),
        mock_play("explicit_onside",1,"kickoff",6,onside=True),
        mock_play("explicit_onside",2,"run",6),
    ])
    games=pd.concat([earlier,excluded],ignore_index=True)
    pairs=audit.eligible_pairs_only(games,step66,helper61["FIELDS"])
    assert len(pairs)==110
    assert pairs["game_id"].nunique()==110
    assert pairs["week"].between(6,18).all()
    assert pairs["qtr_kick"].eq(4).all()
    assert not pairs[["game_id","play_id_kick"]].duplicated().any()


def test_forecasts_use_only_before_kickoff_information_and_frozen_features():
    pairs=audit.eligible_pairs_only(
        fixture_2026_games(110),step66,helper61["FIELDS"]
    )
    original=audit.paired_native_forecasts(
        pairs, FakeWP(), FEATURES, helper61
    )
    assert set(original.columns)==set(audit.SCORE_COLUMNS)
    assert len(original)==110
    assert original["kicker_home"].sum()==55
    assert original["game_id"].nunique()==110
    assert original["native_wp"].between(0,1).all()
    assert original["synthetic_wp"].between(0,1).all()
    assert not original[["native_wp","synthetic_wp"]].equals(
        original[["synthetic_wp","native_wp"]]
    )

    tampered=pairs.copy()
    tampered["yardline_100_snap"]=95.
    tampered["game_seconds_remaining_snap"]=200.
    tampered["result_kick"]=-7.
    tampered["result_snap"]=-7.
    rescored=audit.paired_native_forecasts(
        tampered, FakeWP(), FEATURES, helper61
    )
    np.testing.assert_array_equal(original["native_wp"],rescored["native_wp"])
    np.testing.assert_array_equal(
        original["synthetic_wp"],rescored["synthetic_wp"]
    )
    assert original["actual_home_win"].eq(1).all()
    assert rescored["actual_home_win"].eq(0).all()


def test_wrong_kickoff_native_encoding_fails_without_imputation():
    pairs=audit.eligible_pairs_only(
        fixture_2026_games(110),step66,helper61["FIELDS"]
    )
    bad=pairs.copy()
    bad.loc[0,"yardline_100_kick"]=50.
    with pytest.raises(ValueError,match="Unexpected 2026"):
        audit.paired_native_forecasts(bad,FakeWP(),FEATURES,helper61)


def test_game_level_bootstrap_is_deterministic_and_clumps_kicks():
    differences=np.array([.1,.1,.5,.5])
    game_ids=np.array(["one","one","two","two"])
    a=audit.game_cluster_gain_ci(differences,game_ids,draws=600,seed=83)
    b=audit.game_cluster_gain_ci(differences,game_ids,draws=600,seed=83)
    assert a==b
    assert .099999<=a[0]<=a[1]<=.500001
    with pytest.raises(ValueError,match="paired finite"):
        audit.game_cluster_gain_ci(np.array([]),np.array([]))


def fake_scored(native_wp=.8,old_wp=.2):
    n=120
    return pd.DataFrame({
        "game_id":[f"g{i//2}" for i in range(n)],
        "week":[6+i%13 for i in range(n)],
        "seconds":[200 if i%2 else 500 for i in range(n)],
        "kicker_home":[i%2==0 for i in range(n)],
        "receiver_margin":[0 for i in range(n)],
        "actual_home_win":[1. for i in range(n)],
        "synthetic_wp":[old_wp for i in range(n)],
        "native_wp":[native_wp for i in range(n)],
    },columns=audit.SCORE_COLUMNS)


def test_research_screen_requires_overall_brier_gain_and_both_site_gains():
    scored=fake_scored()
    assert scored["game_id"].nunique()==60
    with pytest.raises(RuntimeError,match="ungated"):
        audit.summarize_once(scored)
    # One row per game satisfies the >=100 completed game threshold.
    scored["game_id"]=[f"g{i}" for i in range(len(scored))]
    s=audit.summarize_once(scored)
    assert s["sample_games"]==120
    assert s["paired_results"]["overall"]["native_brier_gain_positive_better"]>0
    assert s["research_screen"]["preliminary_reliability_support"]
    assert not s["research_screen"]["promotion_to_production_authorized"]
    assert s["research_screen"]["decision"].startswith("PRELIMINARY")
    assert s["paired_results"]["kicker_home"]["games"]==60
    assert s["paired_results"]["kicker_away"]["games"]==60

    bad=scored.copy()
    bad.loc[~bad["kicker_home"],"native_wp"]=.01
    s=audit.summarize_once(bad)
    assert not s["research_screen"]["both_site_point_brier_gains_nonnegative"]
    assert not s["research_screen"]["preliminary_reliability_support"]
    assert s["research_screen"]["decision"].startswith("INCONCLUSIVE")


def test_workflow_source_blob_lock_matches_this_registered_challenger():
    workflow = (
        ROOT / ".github/workflows/forward_2026_native_challenger_once.yml"
    ).read_text()
    actual = step66.git_blob_sha(STEP83)
    assert len(actual) == 40
    assert f'expected="{actual}"' in workflow
    assert audit.PINNED_SOURCE_GIT_BLOBS[
        "src/66_forward_2026_continuation_wp.py"
    ] == step66.git_blob_sha(
        ROOT / "src/66_forward_2026_continuation_wp.py"
    )


def test_one_time_workflow_is_not_an_early_or_manual_future_data_check():
    workflow=(ROOT/".github/workflows/forward_2026_native_challenger_once.yml").read_text()
    source=STEP83.read_text()
    assert "workflow_run:" in workflow
    assert "Frozen 2026 kickoff validation (automatic, once)" in workflow
    assert "on: schedule:" not in workflow
    assert "workflow_dispatch:" not in workflow
    assert "reports/2026_forward_kickoff_wp_final.txt" in workflow
    assert "reports/2026_native_kickoff_challenger_final.json" in workflow
    assert "PREDECLARED RESEARCH DECISION:" in source
    assert "nfl.load_pbp([2026])" in source
    assert "check_existing_gate_report()" in source
    assert source.index("check_existing_gate_report()") < source.index(
        "nfl.load_pbp([2026])"
    )
    assert "PINNED_SOURCE_GIT_BLOBS" in source
    assert "NOT AUTHORIZED" not in source or "NO AUTOMATIC PROMOTION" in source
