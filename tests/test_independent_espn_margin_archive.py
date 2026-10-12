"""Reproduce ALL historical ESPN scores offline with no retained Actions ZIP."""
from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import pytest

from src.historical_scoreboard_verification import (
    GAME_COLUMNS, pbp_game_census,
)
from src.independent_espn_margin_archive import (
    ARCHIVE, EXPECTED_GAMES, EXPECTED_MARGINS_SHA256,
    EXPECTED_KEYS_SHA256, PBP, load_archived_margins,
    replay_score_audit, replay_pinned_source,
)


def source_census():
    return pbp_game_census(
        pd.read_parquet(PBP,columns=list(GAME_COLUMNS))
    )


def test_static_espn_archive_reproduces_every_2014_2025_pbp_game_result():
    assert ARCHIVE.is_file()
    metadata, margins = load_archived_margins()
    assert len(margins) == EXPECTED_GAMES == 3151
    assert metadata["margin_vector_raw_sha256"] == EXPECTED_MARGINS_SHA256
    assert metadata["ordered_matchup_key_sha256"] == EXPECTED_KEYS_SHA256
    assert metadata["source_week_count"] == 209
    assert metadata["source_game_count"] == 3151
    assert metadata["source_pbp_git_blob_sha1"] == "d7230df63cad1c24579ee40cabaacd0b330e4784"
    result = replay_pinned_source()
    assert result["status"] == "ALL_HISTORICAL_FINAL_SCORES_MATCH"
    assert result["source_game_count"] == 3151
    assert result["source_fourth_down_rows"] == 47942
    assert result["prospective_or_causal_claims_authorized"] is False


def test_archive_escapes_2026_and_2027_and_never_fetches():
    metadata,_ = load_archived_margins()
    assert metadata["years"] == list(range(2014,2026))
    assert "2026" not in str(metadata["years"])
    assert "2027" not in str(metadata["years"])
    import pathlib
    source = pathlib.Path("src/independent_espn_margin_archive.py").read_text()
    assert "urllib" not in source
    assert "requests.get" not in source
    assert "nflreadpy" not in source
    assert "joblib.load" not in source
    assert "fit_nuisances(" not in source


def test_wrong_game_team_week_or_sort_order_cannot_be_independently_verified():
    census=source_census()
    idx=census.index[0]
    census.loc[idx,"home"]="NYG"
    with pytest.raises(ValueError,match="identities/order"):
        replay_score_audit(census)


def test_wrong_final_winner_cannot_match_espn_archive():
    census=source_census()
    census.loc[0,"pbp_home_margin"] = -float(census.loc[0,"pbp_home_margin"])+1
    with pytest.raises(ValueError,match="scores differ"):
        replay_score_audit(census)


def test_missing_source_game_cannot_be_claimed_as_full_archive():
    census=source_census().iloc[:-1].copy()
    with pytest.raises(ValueError,match="incomplete"):
        replay_score_audit(census)


def test_cannot_tamper_with_archive_vector_even_if_manifest_copy_exists(tmp_path):
    archive=json.loads(ARCHIVE.read_text())
    archive["margin_vector_base64"]="AAAA"+archive["margin_vector_base64"][4:]
    p=tmp_path/"fake_archive.json"
    p.write_text(json.dumps(archive))
    with pytest.raises(ValueError,match="Malformed|SHA-256"):
        load_archived_margins(p)


def test_cannot_change_reported_count_or_cross_source_keys(tmp_path):
    archive=json.loads(ARCHIVE.read_text())
    archive["source_game_count"]=3150
    p=tmp_path/"fake_archive.json"
    p.write_text(json.dumps(archive))
    with pytest.raises(ValueError,match="metadata/freeze"):
        load_archived_margins(p)
    archive["source_game_count"]=3151
    archive["ordered_matchup_key_sha256"]="0"*64
    p.write_text(json.dumps(archive))
    with pytest.raises(ValueError,match="metadata/freeze"):
        load_archived_margins(p)


def test_score_archive_has_no_recommendation_or_presnap_feature():
    archive=json.loads(ARCHIVE.read_text())
    assert archive["archive_status"].endswith("NOT_PRESNAP")
    assert archive["nuisance_training_model_changed"] is False
    assert len(archive["nonfinal_provider_placeholders"]) == 3
    assert set(x["status"] for x in archive["nonfinal_provider_placeholders"]) == {
        "STATUS_POSTPONED", "STATUS_CANCELED"
    }
    assert "q_wp_by_action" not in archive
    assert "propensity_policy" not in archive
    assert "mu_policy" not in archive
