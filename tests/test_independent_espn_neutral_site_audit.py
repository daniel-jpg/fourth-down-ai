"""ESPN physical neutral-site indicators are independent, retrospective only."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.historical_scoreboard_verification import GAME_COLUMNS
from src.independent_espn_margin_archive import ARCHIVE, PBP
from src.independent_espn_neutral_site_audit import (
    audit_pinned_source, independent_neutral_site_audit,
    load_archived_neutral_flags,
)


def test_3151_archived_espn_neutral_flags_have_unchanged_sha():
    flags=load_archived_neutral_flags()
    assert len(flags)==3151
    assert sum(flags)==49
    assert set(flags)=={0,1}


def test_independent_venue_flag_comparison_is_explicitly_not_presnap_proof():
    report=audit_pinned_source()
    assert report["games"]==3151
    assert report["espn_neutral_site_games"]==49
    assert report["changes_to_frozen_policy"] is False
    assert report["offline_archive_verified"] is True
    assert "actual snap-time or model forecast timestamps" in report["not_certified"]
    assert report["status"] in (
        "ALL_RETROSPECTIVE_SITE_FLAGS_MATCH",
        "SITE_DISAGREEMENTS_REQUIRE_INDEPENDENT_REVIEW",
    )
    assert report["mismatch_count"] == len(report["mismatch_games"])


def test_corrupted_neutral_archive_flags_are_not_trusted(tmp_path):
    obj=json.loads(ARCHIVE.read_text())
    obj["neutral_site_provenance"]["flag_vector_base64"]="AAAA"+obj["neutral_site_provenance"]["flag_vector_base64"][4:]
    fake=tmp_path/"corrupt_espn_archive.json"
    fake.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match="Invalid|integrity"):
        load_archived_neutral_flags(fake)


def test_unrecognized_historical_site_is_never_silently_called_home():
    src=pd.read_parquet(PBP,columns=list(GAME_COLUMNS)+["location"])
    original=src["game_id"].iloc[0]
    src.loc[src["game_id"].eq(original),"location"]="unknown"
    result=independent_neutral_site_audit(src)
    assert len(result["source_location_missing_or_unrecognized"])>=1
    assert result["status"]=="SITE_DISAGREEMENTS_REQUIRE_INDEPENDENT_REVIEW"
    assert any(x["game_id"]==original for x in result["source_location_missing_or_unrecognized"])
