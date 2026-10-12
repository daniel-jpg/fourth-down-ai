"""Regression checks for the read-only release preflight.

The test suite uses checked-in artifacts only and never fetches NFL outcomes.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_preflight", ROOT / "scripts/release_preflight.py"
)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_release_preflight_matches_checked_in_files():
    assert preflight.check() == []


def test_release_preflight_detects_frozen_source_hash_drift(monkeypatch):
    modified = dict(preflight.FROZEN_BLOBS)
    modified["src/83_forward_2026_native_challenger_once.py"] = "0" * 40
    monkeypatch.setattr(preflight, "FROZEN_BLOBS", modified)
    errors = preflight.check()
    assert any("Frozen blob mismatch:" in error for error in errors)
    assert any("workflow source pin differs" in error for error in errors)


def test_release_preflight_detects_model_spec_hash_drift(monkeypatch):
    modified = dict(preflight.FROZEN_BLOBS)
    modified["data/win_probability_spec.json"] = "0" * 40
    monkeypatch.setattr(preflight, "FROZEN_BLOBS", modified)
    errors = preflight.check()
    assert any(
        "Frozen blob mismatch: data/win_probability_spec.json" in error
        for error in errors
    )
