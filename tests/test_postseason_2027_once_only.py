"""Postseason 2027: independent finals, one-time seal, no future holdout."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.rehearse_2027_postseason_once import fabricate_whole_season
from src.postseason_2027_admission import (
    MODE, EXPECTED_GAMES, verify_fabricated_complete_season,
)
from src.postseason_2027_once_only import (
    run_fabricated_postseason_once, validate_label_free_forecasts,
)


@pytest.fixture(scope="module")
def fixture():
    return fabricate_whole_season()


def test_full_fake_season_has_all_272_scores_crosschecked(fixture):
    checked = verify_fabricated_complete_season(
        fixture["pbp_games"], fixture["reference_games"], mode=MODE,
    )
    assert checked["status"] == "FABRICATED_FULL_SEASON_SCORE_CROSSCHECK_PASSED"
    assert len(checked["final_games"]) == EXPECTED_GAMES == 272
    audit = checked["audit"]
    assert audit["score_matches"] == EXPECTED_GAMES
    assert set(audit["source_weeks"]) == set(range(1, 19))
    assert audit["source_weeks"][18] == 16
    assert audit["games_per_team"] == 17
    assert audit["authentic_espn_response_verified"] is False
    assert audit["external_source_independence_certified"] is False
    assert audit["2027_season_currently_complete"] is False
    assert audit["future_evaluation_authorized"] is False
    assert len(audit["nflverse_source_sha256"]) == 64
    assert len(audit["independent_shaped_score_sha256"]) == 64


@pytest.mark.parametrize("corruption,match", [
    ("missing_game", "272"),
    ("nonfinal", "not final"),
    ("score_mismatch", "final-score mismatch"),
    ("wrong_home", "does not match"),
    ("duplicate_event", "Duplicate"),
    ("outside_season", "season"),
    ("untrusted_provider", "fabricated ESPN"),
    ("2026_source", "season"),
])
def test_dangerous_or_incomplete_seasons_are_rejected(fixture, corruption, match):
    obj = deepcopy(fixture)
    pbp = obj["pbp_games"]
    ref = obj["reference_games"]
    if corruption == "missing_game":
        ref.pop()
    elif corruption == "nonfinal":
        ref[0]["status"] = "in_progress"
    elif corruption == "score_mismatch":
        ref[0]["away_score_final"] += 1
    elif corruption == "wrong_home":
        # Preserve a legal team code, change one matchup identity.
        ref[0]["home_team"] = ref[1]["home_team"]
    elif corruption == "duplicate_event":
        ref[1]["event_id"] = ref[0]["event_id"]
    elif corruption == "outside_season":
        ref[0]["season"] = 2028
    elif corruption == "untrusted_provider":
        ref[0]["provider"] = "REAL_ESPN_LIVE"  # never silently assume trust
    elif corruption == "2026_source":
        pbp[0]["season"] = 2026
    with pytest.raises(ValueError, match=match):
        verify_fabricated_complete_season(pbp, ref, mode=MODE)


def test_team_week_duplications_and_missing_week18_are_rejected(fixture):
    obj = deepcopy(fixture)
    pbp, ref = obj["pbp_games"], obj["reference_games"]
    # Move the entire last week to Week 17: same matchups/score copies but
    # one team now plays twice in the week and Week 18 disappears.
    for row in pbp + ref:
        if row["week"] == 18:
            row["week"] = 17
    with pytest.raises(ValueError, match="twice|incomplete"):
        verify_fabricated_complete_season(pbp, ref, mode=MODE)


@pytest.mark.parametrize("bad_mode", [
    "real_2027", "prospective", "retrospective", "", None, True,
])
def test_no_real_holdout_mode_even_inspects_input(bad_mode):
    class Spy:
        def __getattribute__(self, attr):
            raise AssertionError("Inspected forbidden football source!")
    with pytest.raises(PermissionError, match="Genuine 2027 holdout admission"):
        verify_fabricated_complete_season(Spy(), Spy(), mode=bad_mode)
    with pytest.raises(PermissionError, match="Genuine 2027 holdout admission"):
        run_fabricated_postseason_once(
            Spy(), Path("/dev/null/future"), mode=bad_mode,
        )


def test_forecast_seal_precedes_any_external_scores_or_actions(tmp_path, fixture):
    obj = deepcopy(fixture)
    dest = tmp_path / "once"
    # A list subclass proves that source outcomes are only ITERATED after
    # the label-free hash has been durably written.
    class FinalSpy(list):
        def __iter__(self):
            assert (dest / "forecast_seal.json").is_file()
            return super().__iter__()
    obj["reference_games"] = FinalSpy(obj["reference_games"])
    obj["observed_plays"] = FinalSpy(obj["observed_plays"])
    out = run_fabricated_postseason_once(obj, dest, mode=MODE)
    assert out["status"] == "SYNTHETIC_POSTSEASON_REHEARSAL_ONLY"
    assert out["joined_decisions"] == 1
    assert out["source_fourth_down_candidates"] == 2
    assert out["exclusion_counts"] == {"unmatched_pre_snap": 1}
    assert out["source_crosscheck"]["score_matches"] == 272
    assert out["evaluated_synthetic_only"]["mode"] == "synthetic"
    assert out["evaluated_synthetic_only"]["n_decisions"] == 1
    assert out["real_2027_evaluated"] is False
    assert out["real_holdout_release_authorized"] is False
    seal = json.loads((dest / "forecast_seal.json").read_text())
    done = json.loads((dest / "done.json").read_text())
    assert seal["trial"] == "SYNTHETIC_ONLY_NO_REAL_2027_OUTCOMES"
    assert done["status"] == "FABRICATED_ONCE_COMPLETE"
    assert done["real_release_approved"] is False
    assert (dest / "synthetic_evaluation.json").is_file()
    with pytest.raises(FileExistsError, match="never rescore"):
        run_fabricated_postseason_once(obj, dest, mode=MODE)


def test_invalid_final_score_seals_but_cannot_publish_evaluation(tmp_path, fixture):
    obj = deepcopy(fixture)
    obj["reference_games"][0]["away_score_final"] += 7
    dest = tmp_path / "invalid"
    with pytest.raises(ValueError, match="final-score mismatch"):
        run_fabricated_postseason_once(obj, dest, mode=MODE)
    assert (dest / "forecast_seal.json").is_file()
    assert (dest / "failed_after_seal.txt").is_file()
    assert not (dest / "synthetic_evaluation.json").exists()
    assert not (dest / "done.json").exists()
    with pytest.raises(FileExistsError, match="never rescore"):
        run_fabricated_postseason_once(obj, dest, mode=MODE)


def test_forecast_hash_invalid_or_input_contains_outcome_before_sealing(
    tmp_path, fixture,
):
    obj = deepcopy(fixture)
    obj["forecast_bundle"]["synthetic_forecasts"][0]["observed_action"] = "PUNT"
    dest = tmp_path / "leak"
    with pytest.raises(ValueError, match="Unexpected forecast input"):
        run_fabricated_postseason_once(obj, dest, mode=MODE)
    assert not dest.exists()
    obj = deepcopy(fixture)
    obj["forecast_bundle"]["synthetic_forecasts"][0]["q_wp_policy"] = 1.
    with pytest.raises(ValueError, match="commitment hash mismatch"):
        run_fabricated_postseason_once(obj, dest, mode=MODE)
    assert not dest.exists()


def test_observed_play_denominator_must_match_sealed_opportunities(tmp_path, fixture):
    obj = deepcopy(fixture)
    obj["observed_plays"].pop()
    dest = tmp_path / "bad_action"
    with pytest.raises(ValueError, match="differ from sealed denominator"):
        run_fabricated_postseason_once(obj, dest, mode=MODE)
    assert (dest / "failed_after_seal.txt").exists()
    assert not (dest / "synthetic_evaluation.json").exists()


def test_outside_roster_policy_game_is_not_silently_scored(tmp_path, fixture):
    obj = deepcopy(fixture)
    forecast = obj["forecast_bundle"]["synthetic_forecasts"][0]
    forecast["game_id"] = "synthetic_2027_unlisted"
    obj["forecast_bundle"]["audit"]["synthetic_prediction_sha256"] = (
        __import__("src.postseason_2027_admission", fromlist=["_canonical_hash"])
        ._canonical_hash([forecast])
    )
    with pytest.raises(ValueError, match="outside verified season"):
        run_fabricated_postseason_once(obj, tmp_path / "not_in_schedule", mode=MODE)


def test_missing_or_duplicated_fourth_down_denominator_is_rejected(fixture):
    obj = deepcopy(fixture)
    obj["forecast_bundle"]["audit"]["source_fourth_down_candidates"] += 1
    with pytest.raises(ValueError, match="denominator"):
        validate_label_free_forecasts(obj["forecast_bundle"])
    obj = deepcopy(fixture)
    obj["forecast_bundle"]["skipped_opportunities"][0]["play_id"] = "101"
    with pytest.raises(ValueError, match="overlaps"):
        validate_label_free_forecasts(obj["forecast_bundle"])


def test_no_network_or_live_future_source_io_in_rehearsal():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/postseason_2027_admission.py",
        "src/postseason_2027_once_only.py",
        "scripts/rehearse_2027_postseason_once.py",
    ):
        source = (root / relative).read_text()
        for forbidden in ("import nflreadpy", "from urllib.request",
                          "requests.get(", "load_pbp([2027])",
                          "src/66_forward_2026_continuation_wp.py"):
            assert forbidden not in source
