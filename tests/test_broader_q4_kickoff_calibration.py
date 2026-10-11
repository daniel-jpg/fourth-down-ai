"""Broad Q4 scoring-kickoff analysis must use only pre-kickoff conditions."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.streamlit_kickoff_preview import isolated_engine_for_preview

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/81_broader_post_score_kickoff_calibration.py"
spec = importlib.util.spec_from_file_location("broad_2025_q4_reliability_81", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def small_pbp():
    out = []
    def event(game, pid, kind, team, *, yr=2025, q=4,
              fg=None, desc="", two=0, own=35., pen=0,
              td=0, home_score=24, away_score=27, score_team=24):
        return {
            "season": yr, "season_type": "REG",
            "game_id": f"{yr}_{game}",
            "play_id": float(pid), "play_type": kind,
            "qtr": q, "posteam": team,
            "home_team": "HOME", "away_team": "AWAY",
            "result": float(3 if game!="wrong_final" else -3),
            "game_seconds_remaining": 458.,
            "half_seconds_remaining": 458.,
            "down":np.nan, "ydstogo":0.,
            "yardline_100":own, "goal_to_go":np.nan,
            "posteam_score":score_team, "defteam_score":27,
            "posteam_timeouts_remaining":3.,
            "defteam_timeouts_remaining":2.,
            "field_goal_result":fg, "two_point_attempt":two,
            "desc":desc, "penalty":pen, "return_touchdown":td,
            "home_score":home_score,"away_score":away_score,
        }

    # Actual scoring action FG -> normal kickoff; penalty + return TD
    # outcomes are intentionally kept, never filtered using future results.
    out.append(event("fg",100,"field_goal","HOME",fg="made"))
    out.append(event("fg",110,"kickoff","AWAY",
                     desc="Kickoff penalty; returned for touchdown",
                     pen=1,td=1))

    # PAT following touchdown, then kickoff by that same team.
    out.append(event("xp",100,"extra_point","HOME"))
    out.append(event("xp",110,"kickoff","AWAY"))

    # A recorded two-point try as a pass; previously known attempt status.
    out.append(event("two",100,"pass","HOME",two=1))
    out.append(event("two",110,"kickoff","AWAY"))

    # Second kickoff after a scoring action is a repeat and cannot be
    # counted again in the population.
    out.append(event("rekick",100,"field_goal","HOME",fg="made"))
    out.append(event("rekick",110,"kickoff","AWAY",pen=1))
    out.append(event("rekick",120,"kickoff","AWAY"))

    # The team's explicitly declared onside attempt is an excluded policy.
    out.append(event("onside",100,"field_goal","HOME",fg="made"))
    out.append(event("onside",110,"kickoff","AWAY",
                     desc="Declared onside kick"))

    # An arbitrary run followed by kickoff does not imply a made score.
    out.append(event("non_score",100,"run","HOME"))
    out.append(event("non_score",110,"kickoff","AWAY"))

    # Missed FG isn't a post-score kickoff; cannot be incorrectly included.
    out.append(event("miss",100,"field_goal","HOME",fg="missed"))
    out.append(event("miss",110,"kickoff","AWAY"))

    # Wrong kickoff spot belongs to a different special-kick rule.
    out.append(event("wrongspot",100,"field_goal","HOME",fg="made"))
    out.append(event("wrongspot",110,"kickoff","AWAY",own=40.))

    # Kickoff clock <=40s is intentionally out of this preregistered scope.
    out.append(event("late",100,"field_goal","HOME",fg="made"))
    row=event("late",110,"kickoff","AWAY")
    row["game_seconds_remaining"]=40.
    out.append(row)

    # Receiving team home => original kicking team away.
    out.append(event("away_kicker",100,"field_goal","AWAY",fg="made"))
    out.append(event("away_kicker",110,"kickoff","HOME"))

    # Different year and game: prior scoring team must not leak across games.
    out.append(event("fg2023",100,"field_goal","HOME",fg="made",yr=2023))
    out.append(event("fg2023",110,"kickoff","AWAY",yr=2023))

    return pd.DataFrame(out)


def test_fixed_prior_event_selection_includes_scoring_kick_outcomes_not_future_filter():
    data=small_pbp()
    result=audit.historical_post_score_kickoffs(data)
    names=set(result["game_id"])
    assert names == {
        "2025_fg", "2025_xp", "2025_two", "2025_rekick",
        "2025_away_kicker", "2023_fg2023",
    }
    assert len(result)==6
    assert result.loc[result["game_id"].eq("2025_rekick"),
                      "play_id"].tolist()==[110.]
    assert result.loc[result["game_id"].eq("2025_fg"),
                      "return_touchdown"].iloc[0]==1
    assert result.loc[result["game_id"].eq("2025_fg"),
                      "penalty"].iloc[0]==1
    assert result.loc[result["game_id"].eq("2025_away_kicker"),
                      "kicker_site"].iloc[0]=="AWAY"
    assert result["post_score_type"].value_counts().to_dict()=={
        "made_fg":4, "post_extra_point":1,"post_two_point":1,
    }


def test_engine_original_and_fixed_candidate_use_identical_context():
    selected=audit.historical_post_score_kickoffs(small_pbp())
    old=audit.import_2024_feature_helpers()
    baseline,aligned,native=audit.kickoff_model_features(selected,old)
    assert len(baseline)==len(aligned)==len(native)==6
    assert baseline["yardline_100"].isna().all()
    assert baseline["ydstogo"].isna().all()
    assert aligned["yardline_100"].eq(35.).all()
    assert aligned["ydstogo"].eq(0.).all()
    assert aligned["down"].isna().all()
    assert aligned["is_kickoff"].eq(1).all()
    assert aligned["qtr"].eq(4).all()
    assert (aligned["is_home_posteam"]==baseline["is_home_posteam"]).all()
    for column in aligned.columns.difference(["yardline_100","ydstogo"]):
        pd.testing.assert_series_equal(baseline[column],aligned[column])
    scored=audit.score_candidate(selected,isolated_engine_for_preview(),old)
    assert len(scored)==len(selected)
    assert scored["p_synthetic"].between(0,1).all()
    assert scored["p_native_candidate"].between(0,1).all()
    assert set(scored["kicker_site"])=={"HOME","AWAY"}


def test_2026_seasons_and_bad_schema_are_rejected():
    data=small_pbp()
    with pytest.raises(ValueError,match="Only 2023-25"):
        audit.historical_post_score_kickoffs(data.assign(season=2026))
    with pytest.raises(ValueError,match="Missing historical PBP"):
        audit.historical_post_score_kickoffs(data.drop(columns=["result"]))
    with pytest.raises(ValueError,match="Duplicate game/play"):
        audit.historical_post_score_kickoffs(
            pd.concat([data,data.iloc[[0]]],ignore_index=True)
        )


def test_game_bootstrap_is_repeatable_and_groups_whole_games():
    delta=np.array([.1,.1,.5,.5])
    games=np.array(["g1","g1","g2","g2"])
    low,high=audit.game_cluster_ci(delta,games,repeats=200,seed=123)
    low2,high2=audit.game_cluster_ci(delta,games,repeats=200,seed=123)
    assert (low,high)==(low2,high2)
    assert .099999<=low<=high<=.500001
    with pytest.raises(ValueError,match="Nonempty"):
        audit.game_cluster_ci(np.array([]),np.array([]))
    with pytest.raises(ValueError,match="Finite"):
        audit.game_cluster_ci(np.array([np.nan]),np.array(["x"]))


def test_calibration_scores_correct_sign_and_zero_support():
    fake=pd.DataFrame({
        "game_id":["g1","g2"],
        "actual_home_win":[1.,0.],
        "p_synthetic":[.1,.9],
        "p_native_candidate":[.8,.2],
        "p_native_pbp":[.8,.2],
    })
    score=audit.metrics("chosen",fake,ci=True)
    assert score["n"]==2 and score["games"]==2
    assert score["candidate_brier_gain_positive_is_better"]>0
    assert score["candidate_brier"]<score["legacy_brier"]
    assert score["candidate_brier_gain_game_cluster_ci95"][0]>0
    assert audit.metrics("empty",fake.iloc[:0])["support"]=="NO DATA"


def test_no_future_event_selection_or_model_promotion():
    src=SCRIPT.read_text()
    assert "nfl.load_pbp([yr])" in src
    assert "YEARS = (2023, 2024, 2025)" in src
    assert "nfl.load_pbp([2026])" not in src
    assert "2026_game_data_used" in src
    assert "production_promotion" in src
    assert "NOT AUTHORIZED" in src
    assert "joblib.dump(" not in src
    assert "git push" not in src
    assert "app.py" not in src
