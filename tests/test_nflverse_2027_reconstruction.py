"""Fake 2027 and pinned-history-only nflverse intake regression checks."""
from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from src import nflverse_2027_reconstruction as intake


def synthetic_rows():
    base = {
        "season": 2027, "season_type": "REG", "week": 4,
        "game_id": "synthetic_2027_04_AAA_BBB", "play_id": 101.0,
        "down": 4.0, "qtr": 4.0, "game_seconds_remaining": 299.,
        "yardline_100": 6., "ydstogo": 4., "goal_to_go": 0.,
        "score_differential": -4., "posteam_timeouts_remaining": 2.,
        "defteam_timeouts_remaining": 3., "posteam": "AAA",
        "home_team": "AAA", "away_team": "BBB", "location": "Home",
        "roof": "outdoors", "play_type": "field_goal", "qb_scramble": 0,
        "desc": "Field goal attempt", "aborted_play": 0, "safety": 0,
        "result": 3., "yards_gained": -1000., "epa": 2000.,
    }
    second = deepcopy(base)
    second.update(play_id=202., play_type="pass", posteam="BBB",
                  score_differential=4., desc="pass completion")
    third = deepcopy(base)
    third.update(play_id=303., qtr=2., game_seconds_remaining=2000.,
                 play_type="run", qb_scramble=1, result=3.)
    return pd.DataFrame([base, second, third])


def test_preplay_feature_projection_never_sees_outcome_fields():
    original = intake.reconstruct_source_preplay(
        synthetic_rows(), mode=intake.MODE_SYNTHETIC
    )
    changed = synthetic_rows()
    changed["play_type"] = "run"
    changed["qb_scramble"] = 0
    changed["desc"] = "fake punt interception touchdown"
    changed["result"] = -90
    changed["yards_gained"] = 54321
    changed["epa"] = 321
    after = intake.reconstruct_source_preplay(changed, mode=intake.MODE_SYNTHETIC)
    pd.testing.assert_frame_equal(original["preplay_features"], after["preplay_features"])
    assert original["audit"] == after["audit"]
    assert len(original["preplay_features"]) == 3
    assert original["audit"]["source_labels_accessed"] is False
    assert original["audit"]["source_fourth_down_candidates"] == 3
    assert original["audit"]["context_fields_present"] == 2
    assert original["audit"]["missing_model_context_counts"] == {
        "Q2_SECOND_HALF_RECEIVER_UNAVAILABLE": 1,
    }
    assert all(c not in original["preplay_features"] for c in (
        "play_type", "result", "epa", "yards_gained", "desc",
        "observed_action", "offense_win",
    ))
    assert set(original["preplay_features"]["feature_asof"]) == {
        "POSTGAME_RECONSTRUCTED_NOT_WITNESSED",
    }


def test_postplay_actions_and_source_margin_never_misrepresented_as_verified():
    out = intake.project_postplay_labels(synthetic_rows(), mode=intake.MODE_SYNTHETIC)
    assert out["observed_actions"]["observed_action"].tolist() == [
        "FIELD_GOAL", "NORMAL_GO_PASS", "NORMAL_GO_PASS"
    ]
    assert len(out["source_game_result_claims"]) == 1
    assert out["source_game_result_claims"].iloc[0]["source_claimed_home_margin"] == 3
    assert out["source_game_result_claims"].iloc[0]["verification_status"] == "UNVERIFIED_SOURCE_CLAIM"
    assert not out["audit"]["final_game_results_independently_verified"]


@pytest.mark.parametrize("year,mode", [
    (2026, intake.MODE_SYNTHETIC),
    (2026, intake.MODE_HISTORICAL),
    (2027, intake.MODE_HISTORICAL),
    (2025, intake.MODE_SYNTHETIC),
])
def test_rejects_future_2026_or_wrong_season_before_filter(year, mode):
    rows = synthetic_rows()
    rows.loc[0, "season"] = year
    with pytest.raises(ValueError, match="forbidden season"):
        intake.reconstruct_source_preplay(rows, mode=mode)


def test_real_2027_game_ids_cannot_enter_synthetic_preview():
    rows = synthetic_rows()
    rows.loc[0, "game_id"] = "2027_04_AAA_BBB"
    with pytest.raises(ValueError, match="synthetic_"):
        intake.reconstruct_source_preplay(rows, mode=intake.MODE_SYNTHETIC)


def test_all_fourth_down_opportunities_accounted_for_even_when_invalid():
    rows = synthetic_rows()
    overtime = rows.iloc[[0]].copy()
    overtime.loc[:, "play_id"] = 404.
    overtime.loc[:, "qtr"] = 5.
    invalid = rows.iloc[[0]].copy()
    invalid.loc[:, "play_id"] = 505.
    invalid.loc[:, "yardline_100"] = -1.0
    postseason = rows.iloc[[0]].copy()
    postseason.loc[:, "play_id"] = 606.
    postseason.loc[:, "season_type"] = "POST"
    rows = pd.concat([rows, overtime, invalid, postseason], ignore_index=True)
    obj = intake.reconstruct_source_preplay(rows, mode=intake.MODE_SYNTHETIC)
    assert obj["audit"]["source_fourth_down_candidates"] == 6
    assert obj["audit"]["reconstructed_core_states"] == 3
    assert obj["audit"]["excluded_states"] == 3
    assert obj["audit"]["exclusion_counts"] == {
        "MISSING_OR_INVALID_PREPLAY_STATE": 1,
        "NOT_REGULAR_SEASON": 1,
        "OVERTIME": 1,
    }
    assert set(obj["excluded"]["play_id"]) == {"404", "505", "606"}


def test_mixed_game_duplicate_ids_cannot_be_silently_deduped():
    rows = synthetic_rows()
    rows.loc[1, "play_id"] = 101.0
    with pytest.raises(ValueError, match="Duplicate"):
        intake.reconstruct_source_preplay(rows, mode=intake.MODE_SYNTHETIC)


def test_missing_context_stays_missing_without_posthoc_kickoff_or_roof_default():
    rows = synthetic_rows()
    rows.loc[2, "roof"] = None
    obj = intake.reconstruct_source_preplay(rows, mode=intake.MODE_SYNTHETIC)
    q2 = obj["preplay_features"].loc[lambda x: x["quarter"] == 2].iloc[0]
    assert not bool(q2["model_context_complete"])
    assert q2["model_context_gaps"] == (
        "ROOF_UNAVAILABLE_OR_NOT_REVIEWED;Q2_SECOND_HALF_RECEIVER_UNAVAILABLE"
    )
    assert pd.isna(q2["roof"])
    assert obj["audit"]["missing_model_context_counts"] == {
        "Q2_SECOND_HALF_RECEIVER_UNAVAILABLE": 1,
        "ROOF_UNAVAILABLE_OR_NOT_REVIEWED": 1,
    }


def test_fake_no_play_and_scrambles_are_postplay_only_labels():
    rows = synthetic_rows()
    rows.loc[0, "play_type"] = "run"
    rows.loc[0, "desc"] = "fake punt, punt formation scramble"
    rows.loc[1, "play_type"] = "no_play"
    labels = intake.project_postplay_labels(rows, mode=intake.MODE_SYNTHETIC)
    assert labels["observed_actions"]["observed_action"].tolist() == [
        "FAKE_OR_SPECIAL_TEAMS", "NO_PLAY", "NORMAL_GO_PASS",
    ]
    assert labels["audit"]["classified_normal_actions"] == 1
    before = intake.reconstruct_source_preplay(synthetic_rows(), mode=intake.MODE_SYNTHETIC)
    after = intake.reconstruct_source_preplay(rows, mode=intake.MODE_SYNTHETIC)
    pd.testing.assert_frame_equal(before["preplay_features"], after["preplay_features"])


def test_conflicting_final_source_margins_block_label_preview():
    rows = synthetic_rows()
    rows.loc[1, "result"] = -4
    with pytest.raises(ValueError, match="Conflicting source final margins"):
        intake.project_postplay_labels(rows, mode=intake.MODE_SYNTHETIC)


def test_missing_source_columns_fail_closed():
    rows = synthetic_rows().drop(columns=["roof"])
    with pytest.raises(ValueError, match="Missing nflverse source columns"):
        intake.reconstruct_source_preplay(rows, mode=intake.MODE_SYNTHETIC)


def test_only_a_verified_historical_archive_may_be_read(monkeypatch, tmp_path):
    fake = tmp_path / "data" / "fourth_downs_raw.parquet"
    fake.parent.mkdir()
    fake.write_bytes(b"wrong-or-future-season-content")
    monkeypatch.setattr(intake, "ROOT", tmp_path)
    monkeypatch.setattr(intake, "HISTORICAL_SOURCE", fake)
    called = []
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: called.append(True))
    with pytest.raises(ValueError, match="changed"):
        intake.audit_pinned_historical_archive()
    assert not called


def test_census_is_deterministic_independent_of_source_order():
    rows = synthetic_rows()
    first = intake.reconstruct_source_preplay(rows, mode=intake.MODE_SYNTHETIC)
    second = intake.reconstruct_source_preplay(
        rows.iloc[::-1], mode=intake.MODE_SYNTHETIC
    )
    pd.testing.assert_frame_equal(first["preplay_features"], second["preplay_features"])
    assert first["audit"] == second["audit"]


def test_original_nflreadpy_polars_frame_is_supported():
    """nflreadpy.load_pbp returns Polars, not pandas, in this project."""
    pl = pytest.importorskip("polars")
    raw = synthetic_rows()
    pandas_result = intake.reconstruct_source_preplay(raw, mode=intake.MODE_SYNTHETIC)
    polars_result = intake.reconstruct_source_preplay(
        pl.from_pandas(raw), mode=intake.MODE_SYNTHETIC,
    )
    pd.testing.assert_frame_equal(
        pandas_result["preplay_features"],
        polars_result["preplay_features"],
        check_dtype=False,
    )
    assert pandas_result["audit"] == polars_result["audit"]
    pandas_labels = intake.project_postplay_labels(raw, mode=intake.MODE_SYNTHETIC)
    polars_labels = intake.project_postplay_labels(
        pl.from_pandas(raw), mode=intake.MODE_SYNTHETIC,
    )
    pd.testing.assert_frame_equal(
        pandas_labels["observed_actions"],
        polars_labels["observed_actions"],
        check_dtype=False,
    )
