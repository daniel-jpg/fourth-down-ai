"""ESPN physical neutral-site indicators are independent, retrospective only."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.historical_scoreboard_verification import GAME_COLUMNS
from src.independent_espn_margin_archive import ARCHIVE, PBP
from src.independent_espn_neutral_site_audit import (
    audit_pinned_source, independent_neutral_site_audit,
    load_archived_neutral_flags, load_reviewed_relocations,
    OFFICIAL_RELOCATION_FILE,
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
    assert report["status"] == "FOUR_REVIEWED_RELOCATIONS_EXPLAIN_PROVIDER_FLAGS"
    assert report["mismatch_count"] == 4
    assert report["mismatch_count"] == len(report["mismatch_games"])
    assert report["official_relocation_review_count"] == 4
    assert report["unexpected_site_disagreements"] == []
    assert report["reviewed_relocations_missing_from_current_source"] == []
    assert report["pbp_neutral_site_games"] == 53
    assert {x["game_id"] for x in report["official_relocation_reviewed_cases"]} == {
        "2020_13_BUF_SF", "2020_14_WAS_SF",
        "2020_17_SEA_SF", "2021_01_GB_NO",
    }


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
    assert result["status"]=="SITE_ANOMALIES_NEED_ADDITIONAL_REVIEW"
    assert result["unexpected_site_disagreements"] or result["source_location_missing_or_unrecognized"]
    assert any(x["game_id"]==original for x in result["source_location_missing_or_unrecognized"])



def test_relocated_venues_are_proven_by_source_links_not_overwritten():
    reference=load_reviewed_relocations()
    assert set(reference) == {
        "2020_13_BUF_SF", "2020_14_WAS_SF",
        "2020_17_SEA_SF", "2021_01_GB_NO",
    }
    assert all(x["espn_neutral_site"] is False for x in reference.values())
    assert all(x["nflverse_location"] == "Neutral" for x in reference.values())
    assert all(x["official_source"].startswith("https://www.") for x in reference.values())


def test_unreviewed_new_non_neutral_source_mismatch_would_be_reported():
    src=pd.read_parquet(PBP,columns=list(GAME_COLUMNS)+["location"])
    to_flip = src.loc[src["location"].eq("Home"), "game_id"].iloc[0]
    src.loc[src["game_id"].eq(to_flip), "location"] = "Neutral"
    result=independent_neutral_site_audit(src)
    assert result["status"]=="SITE_ANOMALIES_NEED_ADDITIONAL_REVIEW"
    assert any(x["game_id"]==to_flip for x in result["unexpected_site_disagreements"])


def test_relocating_venue_evidence_without_source_is_rejected(tmp_path):
    original=json.loads(OFFICIAL_RELOCATION_FILE.read_text())
    original["expected_reviewed_disagreements"][0]["official_source"]="https://example.com/fake"
    out=tmp_path/"untrusted-reviews.json"
    out.write_text(json.dumps(original))
    with pytest.raises(ValueError,match="Invalid, undocumented"):
        load_reviewed_relocations(out)


def test_do_not_relabel_a_known_relocated_game_to_nominal_home():
    src=pd.read_parquet(PBP,columns=list(GAME_COLUMNS)+["location"])
    src.loc[src["game_id"].eq("2021_01_GB_NO"),"location"]="Home"
    result=independent_neutral_site_audit(src)
    assert result["status"]=="SITE_ANOMALIES_NEED_ADDITIONAL_REVIEW"
    assert "2021_01_GB_NO" in result[
        "reviewed_relocations_missing_from_current_source"
    ]
