"""Historical-only SHADOW intake; no 2026/2027 PBP, no model fitting."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.historical_fourth_down_cohort import (
    ACTION_COLUMNS, OUTCOME_COLUMNS, PRESNAP_COLUMNS,
    extract_historical_cohort, fingerprint_frame,
)
from src.shadow_historical_data_refresh import (
    DESTINATION, FROZEN_POLICY, SOURCE, SOURCE_GIT_BLOB,
    FORBIDDEN_PRESNAP, git_blob_sha1, historical_role,
    plan_historical_snapshot, require_pinned_source,
    require_same_keys, sha256_bytes, version_manifest,
    write_snapshot,
)


def historical_sample():
    base = {
        "season": 2020, "season_type": "REG", "week": 8,
        "game_id": "2020_08_BBB_AAA", "play_id": 100.0,
        "down": 4.0, "qtr": 4.0, "game_seconds_remaining": 299.0,
        "yardline_100": 6.0, "ydstogo": 4.0, "goal_to_go": 0.0,
        "score_differential": -4.0, "posteam_timeouts_remaining": 3.0,
        "defteam_timeouts_remaining": 2.0, "posteam": "AAA",
        "home_team": "AAA", "away_team": "BBB", "location": "Home",
        "action": "NORMAL_GO_PASS", "result": 4.0,
    }
    rows = [base]
    for year, action, gid, play in [
        (2023, "FIELD_GOAL", "2023_10_DDD_CCC", 200.0),
        (2024, "PUNT", "2024_11_FFF_EEE", 300.0),
        (2025, "NORMAL_GO_RUN", "2025_12_HHH_GGG", 400.0),
    ]:
        extra = deepcopy(base)
        extra.update(
            season=year, action=action, game_id=gid, play_id=play,
            week=10, home_team=gid[-3:], away_team="XYZ",
            posteam=gid[-3:], score_differential=1.0,
        )
        # This is still historical/retrospective, NEVER live fourth-down evidence.
        rows.append(extra)
    return pd.DataFrame(rows)


def test_sample_roles_are_chronological_and_2025_not_fresh_holdout():
    cohort=extract_historical_cohort(historical_sample())
    p=plan_historical_snapshot(cohort)
    assert p["included_decisions"]==4
    assert p["excluded_decisions"]==0
    assert p["counts_by_role"]=={
        "previously_seen_2025_development":1,
        "train_2014_2022":1,
        "validation_2023_2024":2,
    }
    assert p["years"]==[2020,2023,2024,2025]
    assert list(cohort["pre_snap"].columns)==PRESNAP_COLUMNS
    assert list(cohort["observed_actions"].columns)==ACTION_COLUMNS
    assert list(cohort["final_labels"].columns)==OUTCOME_COLUMNS
    assert not FORBIDDEN_PRESNAP.intersection(cohort["pre_snap"].columns)
    assert set(p["membership"]["development_role"])==set(p["counts_by_role"])


@pytest.mark.parametrize("year,role",[
    (2014,"train_2014_2022"),(2022,"train_2014_2022"),
    (2023,"validation_2023_2024"),(2024,"validation_2023_2024"),
    (2025,"previously_seen_2025_development"),
])
def test_allowed_historical_years(year,role):
    assert historical_role(year)==role


@pytest.mark.parametrize("year", [2012,2013,2026,2027,2028,2030])
def test_unreviewed_seasons_cannot_be_staged(year):
    with pytest.raises(ValueError,match="unapproved"):
        historical_role(year)
    bad=historical_sample()
    bad.loc[0,"season"]=year
    with pytest.raises(ValueError,match="2014-2025"):
        extract_historical_cohort(bad)


@pytest.mark.parametrize("value",[2020.0,"2020",None,True,False])
def test_no_coerced_or_ambiguous_years(value):
    with pytest.raises(ValueError,match="exact integer"):
        historical_role(value)


def test_actions_or_outcomes_cannot_be_smuggled_into_presnap_snapshot():
    cohort=extract_historical_cohort(historical_sample())
    malformed=deepcopy(cohort)
    malformed["pre_snap"]["offense_win"]=1
    with pytest.raises(ValueError,match="schemas"):
        plan_historical_snapshot(malformed)
    malformed=deepcopy(cohort)
    malformed["pre_snap"]["feature_asof"]="post_snap"
    with pytest.raises(ValueError,match="pre-snap"):
        plan_historical_snapshot(malformed)
    malformed=deepcopy(cohort)
    malformed["observed_actions"].loc[0,"observed_action"]=None
    with pytest.raises(ValueError,match="Missing observed"):
        plan_historical_snapshot(malformed)


def test_observed_actions_and_winners_remain_disjoint_from_features():
    src=historical_sample()
    original=extract_historical_cohort(src)
    changed=src.copy(deep=True)
    changed["result"]=-changed["result"]
    changed.loc[0,"action"]="PUNT"
    later=extract_historical_cohort(changed)
    assert fingerprint_frame(original["pre_snap"])==fingerprint_frame(later["pre_snap"])
    assert fingerprint_frame(original["observed_actions"])!=fingerprint_frame(later["observed_actions"])
    assert fingerprint_frame(original["final_labels"])!=fingerprint_frame(later["final_labels"])


def test_key_mismatch_and_duplicate_membership_fail_closed():
    cohort=extract_historical_cohort(historical_sample())
    broken=deepcopy(cohort)
    broken["observed_actions"].loc[0,"play_id"]="surprise"
    with pytest.raises(ValueError,match="differ"):
        plan_historical_snapshot(broken)
    broken=deepcopy(cohort)
    broken["final_labels"].loc[0,"play_id"]=broken["final_labels"].loc[1,"play_id"]
    broken["final_labels"].loc[0,"game_id"]=broken["final_labels"].loc[1,"game_id"]
    with pytest.raises(ValueError,match="Duplicate"):
        plan_historical_snapshot(broken)


def test_every_excluded_fake_and_overtime_row_is_accounted_for():
    raw=historical_sample()
    fake=raw.iloc[[0]].copy()
    fake["game_id"]="2020_11_FAKE_ABC"
    fake["action"]="FAKE_PUNT_PASS"
    fake["play_id"]=124.0
    ot=raw.iloc[[0]].copy()
    ot["game_id"]="2020_12_OT_ABC"
    ot["qtr"]=5.0
    ot["play_id"]=125.0
    cohort=extract_historical_cohort(pd.concat([raw,fake,ot],ignore_index=True))
    p=plan_historical_snapshot(cohort)
    assert p["source_candidates"]==6
    assert p["included_decisions"]==4
    assert p["excluded_decisions"]==2
    assert p["exclusion_counts"]=={
        "fake_or_unclassified_action":1, "overtime":1,
    }


def test_can_stage_only_separated_artifacts_with_hash_manifest(tmp_path):
    cohort=extract_historical_cohort(historical_sample())
    plan=plan_historical_snapshot(cohort)
    mock={
        "status":"SHADOW_HISTORICAL_DEVELOPMENT_NOT_A_NEW_EVALUATION",
        "contains_2026_or_2027_rows":False,
        "registered_2027_policy_commit_UNCHANGED":FROZEN_POLICY,
    }
    dest=tmp_path/"shadow-historical-test"
    result=write_snapshot(dest,cohort,plan,mock)
    assert (dest/"manifest.json").exists()
    saved=json.loads((dest/"manifest.json").read_text())
    assert result==saved
    assert len(saved["staged_tables"])==5
    for name,meta in saved["staged_tables"].items():
        data=(dest/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==meta["sha256"]
    assert list(pd.read_parquet(dest/"presnap_features.parquet"))==PRESNAP_COLUMNS
    assert "offense_win" not in pd.read_parquet(dest/"presnap_features.parquet")
    assert list(pd.read_parquet(dest/"postgame_outcomes.parquet"))==OUTCOME_COLUMNS
    assert list(pd.read_parquet(dest/"observed_actions.parquet"))==ACTION_COLUMNS
    with pytest.raises(FileExistsError,match="overwritten"):
        write_snapshot(dest,cohort,plan,mock)


def test_shadow_writer_rejects_tracked_production_paths():
    from src.shadow_historical_data_refresh import ROOT
    cohort=extract_historical_cohort(historical_sample())
    plan=plan_historical_snapshot(cohort)
    for path in (ROOT/"models"/"not-for-production", ROOT/"data"/"unsafe",
                 ROOT/"reports"/"unsafe"):
        with pytest.raises(ValueError,match="tracked source/model/reports"):
            write_snapshot(path,cohort,plan,{"research_only":True})


def test_manifest_refuses_cherrypicked_historical_source_and_unverified_final():
    source=require_pinned_source()
    plan=plan_historical_snapshot(extract_historical_cohort(historical_sample()))
    status={
        "status":"ALL_HISTORICAL_FINAL_SCORES_MATCH",
        "source_game_count":3151,
        "source_fourth_down_rows":47942,
        "game_keys_sha256":"0"*64,
        "game_margins_sha256":"0"*64,
    }
    with pytest.raises(ValueError,match="All frozen"):
        version_manifest(plan,source,status)
    plan["source_candidates"]=47942
    plan["years"]=list(range(2014,2026))
    status["status"]="UNVERIFIED"
    with pytest.raises(ValueError,match="All frozen"):
        version_manifest(plan,source,status)


def test_source_must_have_original_git_blob_not_a_new_weekly_file(tmp_path):
    assert SOURCE.is_file()
    assert require_pinned_source()["git_blob_sha1"]==SOURCE_GIT_BLOB
    assert git_blob_sha1(SOURCE.read_bytes())==SOURCE_GIT_BLOB
    new_source=tmp_path/"downloaded_2026_pbp.parquet"
    new_source.write_bytes(b"2026 PBP should never be parsed")
    with pytest.raises(ValueError,match="original checked-in"):
        require_pinned_source(new_source)


def test_changed_source_hash_blocks_run_before_any_parquet_read(monkeypatch):
    import src.shadow_historical_data_refresh as m
    monkeypatch.setattr(m,"SOURCE_GIT_BLOB","0"*40)
    def fail_if_opened(*a,**k):
        raise AssertionError("PARQUET OPENED PRIOR TO FREEZE HASH CHECK")
    monkeypatch.setattr(m.pd,"read_parquet",fail_if_opened)
    with pytest.raises(ValueError,match="Refusing to parse"):
        m.run()


def test_no_read_latest_season_or_auto_model_promote():
    src=Path("src/shadow_historical_data_refresh.py").read_text()
    assert "nflverse" in src  # safety restriction described explicitly
    for bad in (
        "nfl.load_pbp(", "requests.get(", "urllib.request",
        "load_pbp([2026", "load_pbp([2027",
        "joblib.dump(", "model.fit(", "Step 66", "Step 83",
    ):
        assert bad not in src
    assert "SHADOW_HISTORICAL_DEVELOPMENT_NOT_A_NEW_EVALUATION" in src
