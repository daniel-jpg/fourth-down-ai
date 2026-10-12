"""No-lookahead retrospective NFL PBP data completeness census."""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.historical_prespnap_evidence_inventory import (
    AUDITED_FIELDS, SCOPE_FIELDS, PBP, EXPECTED_BLOB_SHA1,
    TRUSTED_CAPTURE_FIELDS, KICKOFF_DIRECTION_CANDIDATES,
    STATE_FIELDS, inventory_retrospective_fields, _git_blob_sha1,
)


def source_fixture():
    f=pd.DataFrame([
        dict(season=2014,season_type="REG",down=4,qtr=1,
             game_id="2014_01_ABC_DEF",play_id=124,
             game_seconds_remaining=3400,
             location="Home",roof="outdoors"),
        dict(season=2025,season_type="REG",down=4,qtr=4,
             game_id="2025_09_DEF_ABC",play_id=456,
             game_seconds_remaining=300,
             location="Neutral",roof=None),
        dict(season=2025,season_type="POST",down=4,qtr=4,
             game_id="post_game",play_id=1,
             game_seconds_remaining=300,location="Home",roof="closed"),
        dict(season=2025,season_type="REG",down=3,qtr=4,
             game_id="third_down",play_id=1,
             game_seconds_remaining=300,location="Home",roof="closed"),
        dict(season=2025,season_type="REG",down=4,qtr=5,
             game_id="overtime",play_id=1,
             game_seconds_remaining=0,location="Home",roof="dome"),
    ])
    return f


def audit(f):
    raw_cols=set(f.columns)|{"result","play_type","kicker_player_id","yards_gained"}
    return inventory_retrospective_fields(f,raw_columns=raw_cols)


def test_pre_snap_coverage_is_only_a_retrospective_inventory():
    report=audit(source_fixture())
    assert report["regulation_fourth_down_rows"]==2
    assert report["distinct_games"]==2
    assert report["fields"]["roof"]["nonmissing"]==1
    assert report["fields"]["roof"]["coverage_fraction"]==.5
    assert report["fields"]["location"]["nonmissing"]==2
    assert report["fields"]["goal_to_go"]["source_column_present"] is False
    assert report["fields"]["game_seconds_remaining"]["nonmissing"]==2
    assert report["by_season"]["2014"]["regulation_fourth_down_rows"]==1
    assert report["by_season"]["2025"]["regulation_fourth_down_rows"]==1
    assert report["status"].endswith("NOT_DECISION_TIME_CERTIFIED")
    assert report["safe_for_model_input"] is False


def test_missing_receiver_coin_toss_and_independent_clock_never_certified():
    report=audit(source_fixture())
    assert all(not report["fields"][k]["source_column_present"]
               for k in KICKOFF_DIRECTION_CANDIDATES)
    assert all(not report["fields"][k]["source_column_present"]
               for k in TRUSTED_CAPTURE_FIELDS)
    assert all(not report["fields"][k]["availability_certified_before_snap"]
               for k in AUDITED_FIELDS)
    assert report["risk_summary"]["source_facts_with_independent_utc_signatures"]==0
    assert report["permits_2026_gate_or_2027_prospective_evaluation"] is False


def test_post_snap_columns_are_only_reported_as_prohibited_metadata():
    report=audit(source_fixture())
    assert {"result","play_type","yards_gained","kicker_player_id"}.issubset(
        set(report["post_snap_excluded_from_projected_frame"])
    )
    frame=source_fixture()
    frame["result"]=2
    with pytest.raises(ValueError,match="potentially post-snap"):
        audit(frame)


def test_future_reserved_year_is_rejected_even_on_filtered_out_rows():
    f=source_fixture()
    f.loc[0,"season"]=2026
    with pytest.raises(ValueError,match="Forbidden 2026"):
        audit(f)
    f=source_fixture()
    f.loc[0,"season"]=2027
    with pytest.raises(ValueError,match="Forbidden 2026"):
        audit(f)


def test_missing_scope_or_wrong_types_rejected():
    f=source_fixture().drop(columns="down")
    with pytest.raises(ValueError,match="date/down/quarter"):
        audit(f)
    with pytest.raises(ValueError,match="nonempty"):
        audit(source_fixture().iloc[:0])


def test_original_historical_raw_source_is_pinned_and_has_no_2026_events():
    assert PBP.is_file()
    assert _git_blob_sha1(PBP)==EXPECTED_BLOB_SHA1
    cols=set(pq.ParquetFile(PBP).schema.names)
    required=set(SCOPE_FIELDS)
    assert required.issubset(cols)
    raw=pd.read_parquet(
        PBP, columns=sorted(cols & (set(AUDITED_FIELDS)|required))
    )
    audit_row=inventory_retrospective_fields(raw,raw_columns=cols)
    assert audit_row["years"]==list(range(2014,2026))
    assert audit_row["regulation_fourth_down_rows"]>40000
    assert audit_row["distinct_games"]>3000
    assert audit_row["safe_for_model_input"] is False
