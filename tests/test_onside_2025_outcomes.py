"""Verify 2025 onside recovery labels and actual next-possession matching."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"src/74_freeze_2025_onside_outcomes.py"
spec=importlib.util.spec_from_file_location("step74_onside",SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sample_onside_2025():
    rows=[]
    for i in range(41):
        # Three kicker recoveries, deliberately one penalty-affected.
        success = i in (0, 9, 30)
        kicker = "LAR"
        receiver = "SEA"
        kickoff = {
            "season":2025, "season_type":"REG", "game_id":f"g{i}",
            "play_id":100., "week":1+i%18,"play_type":"kickoff",
            "qtr":4,"yardline_100":35.,"game_seconds_remaining":200.,
            "home_score":31,"away_score":30,"posteam":receiver,
            "posteam_score":31,"defteam_score":30,"desc":"declared onside kick",
            "return_touchdown":0, "penalty":int(i==30),
            "own_kickoff_recovery":int(success),"down":np.nan,
        }
        first = dict(kickoff)
        first.update({
            "play_id":110., "play_type":"run",
            "game_seconds_remaining":195.,
            "posteam":kicker if success else receiver,
            "yardline_100":57. if success else 43.,
            "down":1.,
            "desc":"First scrimmage snap after kickoff",
        })
        rows.extend([kickoff, first])
    # An ordinary historical kickoff for Step 73 denominator reconciliation.
    normal=rows[1].copy()
    normal.update({
        "game_id":"ordinary","play_id":100.,"play_type":"kickoff",
        "desc":"A touchback", "own_kickoff_recovery":0,
        "yardline_100":35., "down":np.nan, "penalty":0,
    })
    rows.append(normal)
    frame=pd.DataFrame(rows)
    donors=pd.DataFrame({"game_id":["ordinary"],"kick_play_id":[100.]})
    return frame, donors


def test_full_2025_onside_denominator_and_verifiable_possession():
    raw,donors=sample_onside_2025()
    step73=mod.load_step73()
    onside=mod.classify_onside(raw,donors,step73)
    assert len(onside)==41
    paired=mod.pair_first_snaps(raw,onside)
    report=mod.summarize(onside,paired)
    assert report["all_on_side_eligible"]==41
    assert report["flag_known"]==41
    assert report["official_kicker_recoveries"]==3
    assert report["matched_reliable"]==40
    assert report["exclusion_reason_counts"]["penalty_on_onside_attempt"]==1
    assert report["reliable_first_snap"]["kicker_recovers"]["n"]==2
    assert report["reliable_first_snap"]["receiver_recovers"]["n"]==38
    assert report["historical_only"] and report["no_2026_data"]


def test_onside_success_must_change_possession_and_never_assume_receiver():
    raw,donors=sample_onside_2025()
    raw.loc[
        (raw["game_id"]=="g0") & raw["play_type"].eq("run"),
        "posteam",
    ]="SEA"  # erroneously unchanged after recorded kicking-team recovery
    onside=mod.classify_onside(raw,donors,mod.load_step73())
    paired=mod.pair_first_snaps(raw,onside)
    wrong=paired.loc[paired["game_id"].eq("g0")].iloc[0]
    assert not bool(wrong["matched_reliable"])
    assert wrong["unmatched_reason"]=="official_flag_vs_first_snap_disagreement"


def test_no_next_snap_is_not_silently_counted_as_failed_onside():
    raw,donors=sample_onside_2025()
    raw=raw.loc[
        ~((raw["game_id"]=="g0") & raw["play_type"].eq("run"))
    ].copy()
    onside=mod.classify_onside(raw,donors,mod.load_step73())
    paired=mod.pair_first_snaps(raw,onside)
    g0=paired.loc[paired["game_id"].eq("g0")].iloc[0]
    assert not bool(g0["matched_reliable"])
    assert g0["unmatched_reason"]=="no_clean_next_snap"
    assert mod.summarize(onside,paired)["official_kicker_recoveries"]==3


def test_missing_official_recovery_label_cannot_be_guessed():
    raw,donors=sample_onside_2025()
    raw.loc[
        (raw["game_id"]=="g0") & raw["play_type"].eq("kickoff"),
        "own_kickoff_recovery",
    ]=np.nan
    onside=mod.classify_onside(raw,donors,mod.load_step73())
    paired=mod.pair_first_snaps(raw,onside)
    report=mod.summarize(onside,paired)
    assert report["flag_missing"]==1
    assert report["official_kicker_recoveries"]==2
    assert report["exclusion_reason_counts"]["unknown_recovery_flag"]==1


def test_exclude_future_seasons_and_require_explicit_schema():
    raw,donors=sample_onside_2025()
    step=mod.load_step73()
    with pytest.raises(ValueError,match="2025-only"):
        mod.classify_onside(raw.assign(season=2026),donors,step)
    with pytest.raises(ValueError,match="fields missing"):
        mod.classify_onside(raw.drop(columns=["own_kickoff_recovery"]),
                            donors,step)
    with pytest.raises(ValueError,match="Missing first-snap"):
        mod.pair_first_snaps(raw.drop(columns=["home_score"]),raw.iloc[:1])


def test_event_classification_is_not_a_2026_calibration_or_live_model_edit():
    s=SCRIPT.read_text()
    assert "nfl.load_pbp([2025])" in s
    assert "nfl.load_pbp([2026])" not in s
    assert "own_kickoff_recovery" in s
    assert "posteam_snap" in s
    assert "model.fit(" not in s
    assert "joblib.dump(" not in s
