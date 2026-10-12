"""Fabricated ESPN-vs-PBP score audits. Never call a network endpoint in CI tests."""
from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from src.historical_scoreboard_verification import (
    FIRST, LAST, RECORD_KEYS, parse_espn_week, pbp_game_census,
    team_code, verify_independent_margins, season_int, week_int,
)


def espn_game(*, season=2025, week=1, home="LAR", away="HOU",
              home_score="20", away_score="17", final=True, event_id="987654"):
    return {
        "id": event_id,
        "season": {"year": season, "type": 2},
        "week": {"number": week},
        "status": {"type": {"completed": final, "name": (
            "STATUS_FINAL" if final else "STATUS_IN_PROGRESS"
        )}},
        "competitions": [{
            "neutralSite": False,
            "competitors": [
                {"homeAway": "home", "score": home_score,
                 "team": {"abbreviation": home}},
                {"homeAway": "away", "score": away_score,
                 "team": {"abbreviation": away}},
            ],
        }],
    }


def pbp_games():
    return pd.DataFrame([
        dict(season=2025, season_type="REG", week=1, game_id="2025_01_HOU_LA",
             home_team="LA", away_team="HOU", result=3.0, play_id=100),
        dict(season=2025, season_type="REG", week=1, game_id="2025_01_HOU_LA",
             home_team="LA", away_team="HOU", result=3.0, play_id=203),
        dict(season=2025, season_type="REG", week=1, game_id="2025_01_JAX_WAS",
             home_team="WAS", away_team="JAX", result=0.0, play_id=100),
        dict(season=2025, season_type="POST", week=1, game_id="ignored_post",
             home_team="LAR", away_team="HOU", result=-5., play_id=101),
    ])


def two_game_payload():
    return {"events": [
        espn_game(),
        espn_game(home="WSH", away="JAC", home_score="31",
                  away_score="31", event_id="987655"),
    ]}


def test_espn_parse_is_final_and_team_aliases_are_explicit():
    rows = parse_espn_week(two_game_payload(), 2025, 1)
    assert len(rows) == 2
    assert set(rows[0]) == set(RECORD_KEYS)
    assert rows[0]["home"] == "LAR"
    assert rows[0]["away"] == "HOU"
    assert rows[0]["margin"] == 3
    assert rows[1]["home"] == "WSH"
    assert rows[1]["away"] == "JAX"
    assert rows[1]["margin"] == 0
    for raw, normalized in [
        ("LA","LAR"),("STL","LAR"),("SD","LAC"),
        ("LV","LV"),("OAK","LV"),("JAC","JAX"),
        ("WSH","WSH"),("WAS","WSH"),
    ]:
        assert team_code(raw) == normalized


def test_pbp_final_margin_census_deduplicates_plays_and_ignores_postseason():
    census = pbp_game_census(pbp_games())
    assert len(census) == 2
    assert census["pbp_fourth_down_rows"].sum() == 3
    a = census.loc[census["home"] == "LAR"].iloc[0]
    assert a["away"] == "HOU"
    assert a["pbp_home_margin"] == 3


def test_independent_audit_matches_multiple_seasons_and_checks_winners():
    census = pbp_game_census(pbp_games())
    out = verify_independent_margins(
        census, parse_espn_week(two_game_payload(), 2025, 1),
        expected_seasons=[2025],
    )
    assert out["status"] == "ALL_MATCHED"
    assert out["matched_final_scores"] == 2
    assert out["pbp_games"] == 2
    assert out["total_fourth_down_rows_in_verified_games"] == 3
    assert len(out["matched_games_sha256"]) == 64
    assert out["issues"] == []
    assert out["not_verified"]


def test_wrong_final_winner_or_margin_is_a_hard_blocker():
    census = pbp_game_census(pbp_games())
    raw = two_game_payload()
    raw["events"][0]["competitions"][0]["competitors"][0]["score"] = "21"
    rows = parse_espn_week(raw, 2025, 1)
    out = verify_independent_margins(census, rows, expected_seasons=[2025])
    assert out["status"] == "BLOCKED_INCOMPLETE_OR_MISMATCHED"
    assert out["matched_final_scores"] == 1
    assert out["issues"][0]["reason"] == "FINAL_MARGIN_MISMATCH"
    assert out["issues"][0]["espn_margin"] == 4


def test_missing_game_or_pbp_final_result_is_never_reported_as_verified():
    census = pbp_game_census(pbp_games())
    reference = parse_espn_week(two_game_payload(), 2025, 1)
    out = verify_independent_margins(
        census, reference[:1], expected_seasons=[2025]
    )
    assert out["status"] == "BLOCKED_INCOMPLETE_OR_MISMATCHED"
    assert out["issues"][0]["reason"] == "NO_INDEPENDENT_GAME_MATCH"
    census.loc[census["home"] == "LAR", "pbp_home_margin"] = None
    out = verify_independent_margins(
        census, reference, expected_seasons=[2025]
    )
    assert len(out["missing_pbp_scores"]) == 1
    assert out["status"] != "ALL_MATCHED"


@pytest.mark.parametrize("change,expected", [
    (("season", {"year": 2026, "type": 2}), "ESPN season"),
    (("week", {"number": 2}), "Wrong ESPN"),
    (("season", {"year": 2025, "type": 3}), "Wrong ESPN"),
])
def test_wrong_week_type_or_future_season_fails(change, expected):
    bad = two_game_payload()
    bad["events"][0][change[0]] = change[1]
    with pytest.raises(ValueError, match=expected):
        parse_espn_week(bad, 2025, 1)


def test_incomplete_unfinished_score_cannot_be_used():
    raw = two_game_payload()
    raw["events"][0]["status"]["type"]["completed"] = False
    with pytest.raises(ValueError, match="not final"):
        parse_espn_week(raw, 2025, 1)


@pytest.mark.parametrize("bad_score", ["", "pending", "-3", None, 1.2, 999])
def test_invalid_score_is_rejected(bad_score):
    obj = two_game_payload()
    obj["events"][0]["competitions"][0]["competitors"][0]["score"] = bad_score
    with pytest.raises(ValueError, match="score"):
        parse_espn_week(obj, 2025, 1)


def test_duplicate_espn_matchups_or_nonteam_identifiers_fail():
    raw = two_game_payload()
    raw["events"].append(deepcopy(raw["events"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        parse_espn_week(raw, 2025, 1)
    with pytest.raises(ValueError, match="Unrecognized"):
        team_code("FUTURE_TEAM")


def test_false_complete_cannot_hide_wrong_home_and_away():
    census = pbp_game_census(pbp_games())
    raw = two_game_payload()
    first = raw["events"][0]["competitions"][0]["competitors"]
    first[0]["homeAway"], first[1]["homeAway"] = "away", "home"
    # This time ESPN is home=HOU, away=LAR — must NOT match
    out = verify_independent_margins(
        census, parse_espn_week(raw, 2025, 1),
        expected_seasons=[2025],
    )
    assert out["matched_final_scores"] == 1
    assert out["status"] == "BLOCKED_INCOMPLETE_OR_MISMATCHED"


def test_source_final_margin_conflict_fails_before_independent_join():
    source = pbp_games()
    source.loc[1, "result"] = 20.
    with pytest.raises(ValueError, match="Conflicting PBP"):
        pbp_game_census(source)


@pytest.mark.parametrize("year", [2013,2026,2027])
def test_future_reserved_years_are_not_available(year):
    with pytest.raises(ValueError, match="2014"):
        season_int(year)


def test_week_18_before_2021_is_prohibited():
    with pytest.raises(ValueError, match="Week"):
        week_int(18,2020)
    assert week_int(18, 2025) == 18


def test_independent_scoreboard_cannot_be_used_as_pre_snap_feature():
    source = __import__("pathlib").Path(
        "src/historical_scoreboard_verification.py"
    ).read_text()
    assert "2026" in source
    assert "requests.get" not in source
    assert "load_pbp" not in source
    assert "joblib.load" not in source
    assert "fit_nuisances" not in source
