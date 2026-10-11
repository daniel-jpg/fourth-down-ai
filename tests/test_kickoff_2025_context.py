import runpy

import numpy as np
import pandas as pd
import pytest

context = runpy.run_path("src/56_audit_2025_kickoff_context.py")
add_margin = context["add_receiver_margin"]
margin_label = context["margin_label"]
stats = context["context_stats"]


def pbp_example():
    return pd.DataFrame([
        {"game_id": "G", "play_id": 1, "play_type": "kickoff", "posteam": "SEA",
         "posteam_score": 31, "defteam_score": 30, "score_differential": 1},
        {"game_id": "G", "play_id": 2, "play_type": "kickoff", "posteam": "SEA",
         "posteam_score": 31, "defteam_score": 34, "score_differential": -3},
        {"game_id": "G", "play_id": 3, "play_type": "kickoff", "posteam": "SEA",
         "posteam_score": 31, "defteam_score": 34, "score_differential": np.nan},
    ])


def audit_example():
    return pd.DataFrame([
        {"game_id": "G", "kick_play_id": 1, "status": "touchback", "own": 35, "elapsed": 0},
        {"game_id": "G", "kick_play_id": 2, "status": "return", "own": 30, "elapsed": 7},
        {"game_id": "G", "kick_play_id": 3, "status": "onside", "own": np.nan, "elapsed": np.nan},
    ])


def test_score_groups_reflect_receiving_team_after_fg_and_td():
    with_margin = add_margin(audit_example(), pbp_example())
    assert with_margin["receiver_margin"].tolist() == [1, -3, -3]
    assert with_margin["receiver_margin"].map(margin_label).tolist() == [
        "receiver leading 1-3", "receiver trailing 1-3", "receiver trailing 1-3"
    ]


def test_tally_distinguishes_ordinary_and_onside():
    enriched = add_margin(audit_example(), pbp_example())
    values = stats(enriched).set_index("receiving_team_score")
    ahead = values.loc["receiver leading 1-3"]
    behind = values.loc["receiver trailing 1-3"]
    assert ahead["ordinary"] == 1
    assert ahead["touchback_rate"] == 1
    assert behind["ordinary"] == 1
    assert behind["touchback_rate"] == 0
    assert behind["onside"] == 1
    assert behind["avg_clock"] == 7


def test_score_conflict_aborts_instead_of_misclassifying():
    raw = pbp_example()
    raw.loc[0, "score_differential"] = -1
    with pytest.raises(ValueError, match="disagrees"):
        add_margin(audit_example(), raw)


def test_margin_categories_and_unknowns():
    vals = [-14, -4, -3, -1, 0, 1, 3, 4, np.nan]
    assert list(map(margin_label, vals)) == [
        "receiver trailing 4+", "receiver trailing 4+",
        "receiver trailing 1-3", "receiver trailing 1-3", "tied",
        "receiver leading 1-3", "receiver leading 1-3",
        "receiver leading 4+", "unknown",
    ]
