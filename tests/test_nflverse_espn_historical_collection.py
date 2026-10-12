"""Historical-only source manifest and independent ESPN margin regressions."""
from __future__ import annotations

import hashlib
import json

import pytest

from src import nflverse_espn_historical_collection as packet


@pytest.fixture(scope="module")
def real_archive_report():
    # This reads only the original 2014-25 pinned Parquets, plus the 3151
    # archived ESPN scoreboard margins. No network / 2026 / 2027 access.
    return packet.historical_2025_source_evidence(
        year=2025, mode=packet.MODE
    )


def test_real_2025_nflverse_and_independent_espn_archive(real_archive_report):
    report = real_archive_report
    assert report["status"] == "HISTORICAL_2025_SOURCE_PROVENANCE_VERIFIED"
    assert report["season"] == 2025
    assert report["source"] == packet.ALLOWED_SCHEMA_SOURCE
    assert report["policy_commit"] == packet.FROZEN_POLICY
    assert report["reg_game_count"] == 272
    assert sum(report["weeks"].values()) == 272
    assert set(report["weeks"]) == set(range(1, 19))
    assert report["weeks"][18] == 16
    scores = report["archived_independent_scores"]
    assert scores["reference_games_verified"] == 3151
    assert scores["provider"] == "ESPN historical score archive (2014-2025)"
    assert scores["original_margins_sha256"] == packet.EXPECTED_MARGINS_SHA256
    assert scores["original_keys_sha256"] == packet.EXPECTED_KEYS_SHA256
    assert len(report["season_2025_matchup_sha256"]) == 64
    assert len(report["pbp_original_sha256"]) == 64
    assert len(report["preplay_schema_sha256"]) == 64
    assert report["preplay_source_field_names"] == sorted(
        packet.PREPLAY_SOURCE_FIELDS
    )
    assert report["independent_historical_score_margin_verified"] is True
    for flag in (
        "quarterback_and_action_labels_accessed",
        "actual_2027_outcomes_accessed",
        "real_2027_collection_authorized",
        "timestamped_pre_snap_claim_authorized",
    ):
        assert report[flag] is False


def test_archival_packet_is_exclusive_and_deterministic(tmp_path, real_archive_report):
    out = tmp_path / "source_evidence.json"
    saved = packet.save_historical_2025_evidence(
        out, year=2025, mode=packet.MODE,
    )
    assert saved == real_archive_report
    assert json.loads(out.read_text()) == saved
    before = hashlib.sha256(out.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="must be new"):
        packet.save_historical_2025_evidence(
            out, year=2025, mode=packet.MODE,
        )
    assert hashlib.sha256(out.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("year,mode", [
    (2026, packet.MODE),
    (2027, packet.MODE),
    (2028, packet.MODE),
    (2025, "completed_season"),
    (2025, "real_2027"),
    (2025, None),
    ("2025", packet.MODE),
    (True, packet.MODE),
])
def test_future_season_denied_before_archive_read(monkeypatch, year, mode):
    def forbidden(*a, **kw):
        raise AssertionError("Attempted to read protected source!")
    monkeypatch.setattr(packet, "_read_pinned", forbidden)
    with pytest.raises(PermissionError, match="2025 only"):
        packet.historical_2025_source_evidence(year=year, mode=mode)


def test_tampered_original_raw_pbp_blocked_before_parquet_read(
    monkeypatch, tmp_path,
):
    broken = tmp_path / "changed_historical_source.parquet"
    broken.write_bytes(b"an invented dataset, not original git Parquet")
    monkeypatch.setattr(packet, "HISTORICAL_SOURCE", broken)
    import pandas as pd
    monkeypatch.setattr(
        pd, "read_parquet",
        lambda *args, **kw: (_ for _ in ()).throw(
            AssertionError("Parsed tampered Parquet!")
        ),
    )
    with pytest.raises(ValueError, match="original frozen source"):
        packet.historical_2025_source_evidence(year=2025, mode=packet.MODE)


def test_symlink_original_file_blocked(monkeypatch, tmp_path):
    target = tmp_path / "source.parquet"
    target.write_bytes(b"arbitrary")
    link = tmp_path / "source-link.parquet"
    link.symlink_to(target)
    monkeypatch.setattr(packet, "HISTORICAL_SOURCE", link)
    with pytest.raises(ValueError, match="symlinked"):
        packet.historical_2025_source_evidence(year=2025, mode=packet.MODE)
