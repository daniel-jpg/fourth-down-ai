"""Documentation-shaped Pulse v7 fabricated replay, without live NFL access."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from src.sportradar_pulse_synthetic_adapter import (
    CANDIDATE_PROVIDER, replay_fabricated_pulse,
    audit_fabricated_opportunity_coverage,
)
from src.synthetic_decision_time_seals import (
    digest, make_synthetic_bundle, sign_collector, witness_collector,
    public_key_bytes, verify_witnessed_before_snap, join_labels_only_after_witness,
)
from src.frozen_2027_policy_bridge import (
    load_verified_engine, build_synthetic_policy_forecast,
)
from src.historical_nuisance_portable import load_historical_parameters
from src.fourth_down_policy_study import POLICY_COMMIT

CAPTURE="2027-09-09T20:00:00Z"
SNAP="2027-09-09T20:00:08Z"
PROVIDER_TIME="2027-09-09T19:59:50+00:00"


def pulse_fixture():
    return {
        "mode":"fabricated_only",
        "provider":CANDIDATE_PROVIDER,
        "season":2027,"week":1,
        "collector_received_at_utc":"2027-09-09T19:59:52Z",
        "captured_at_utc":CAPTURE,
        "message":{
            "metadata":{
                "league":"nfl","version":"v7","operation":"update",
                "event_type":"pulse","status":"inprogress",
                "match":"sd:match:doc-fixture-001",
            },
            "payload":{
                "game":{
                    "id":"doc-fixture-001","game_type":"regular",
                    "status":"inprogress","quarter":1,"clock":"06:38",
                    "summary":{
                        "home":{
                            "alias":"ABC","points":27,
                            "remaining_timeouts":3,
                        },
                        "away":{
                            "alias":"DEF","points":31,
                            "remaining_timeouts":2,
                        },
                    },
                },
                "pulse":{
                    "id":"pulse-001","play_id":"p-001",
                    "type":"event","event_type":"players_lining_up",
                    "source":"SRUS",
                    "created_at":PROVIDER_TIME,
                    # This is a TIMESTAMP ASSOCIATED WITH A PRIOR PLAY
                    # according to Sportradar's documentation; must be ignored.
                    "wall_clock":"2027-09-09T19:59:05+00:00",
                    "period":{"number":1},"down":4,
                    "clock":"06:38","yfd":4,
                    "home_points":27,"away_points":31,
                    "game_clock":{"running":False,"time":"06:38"},
                    "possession":{"alias":"ABC"},
                    "location":{"alias":"DEF","yardline":6},
                    "start_situation":{
                        "down":4,"clock":"06:38","yfd":4,
                        "possession":{"alias":"ABC"},
                        "location":{"alias":"DEF","yardline":6},
                    },
                    "details":[],
                    # In an actual Pulse message, this may describe a
                    # previous/current play situation: NEVER used here.
                    "end_situation":{"down":1,"clock":"06:38"},
                },
            },
        },
        "physical_venue":{
            "source_ref":"synthetic://physical_venue/documentation-venue-001",
            "observed_at_utc":"2027-09-09T19:54:00Z",
            "received_at_utc":"2027-09-09T19:54:02Z",
            "physical_stadium_id":"ABC_Home_Stadium",
            "nominal_home_stadium_id":"ABC_Home_Stadium",
            "physical_neutral_site":False,
            "roof":"outdoors",
        },
        "pregame_coin_toss":{
            "source_ref":"synthetic://halftime_receiver/documentation-coin-001",
            "observed_at_utc":"2027-09-09T18:30:00Z",
            "received_at_utc":"2027-09-09T18:30:01Z",
            "second_half_receiver_team":"DEF",
        },
    }


def test_safe_q1_fabricated_documentation_pulse_translates_to_frozen_state():
    raw=pulse_fixture()
    output=replay_fabricated_pulse(raw,simulated_snap_at_utc=SNAP)
    s=output["snapshot"]
    assert s["source_type"]=="fabricated_only"
    assert s["feature_asof"]=="pre_snap"
    assert s["quarter"]==1 and s["down"]==4
    assert s["game_seconds_remaining"]==3098
    assert s["ydstogo"]==4.0 and s["yardline_100"]==6.0
    assert s["score_differential"]==-4.0
    assert s["site"]=="HOME" and s["site_advantage"]==1
    assert s["posteam_timeouts_remaining"]==3
    assert s["defteam_timeouts_remaining"]==2
    assert s["roof"]=="outdoors"
    assert s["second_half_receiver"] is None
    assert set(x["kind"] for x in output["source_events"]) == {
        "play_state","physical_venue","halftime_receiver",
    }
    assert output["source_events"][0]["observed_at_utc"]=="2027-09-09T19:59:50Z"
    assert output["source_events"][0]["received_at_utc"]=="2027-09-09T19:59:52Z"
    assert output["source_wire_sha256"]==digest(raw["message"])
    assert output["wall_clock_used_for_snap"] is False
    assert output["real_provider_connected"] is False
    assert output["real_independent_timestamp_certified"] is False


def test_wall_clock_can_be_arbitrarily_earlier_and_does_not_backdate_forecast():
    first=pulse_fixture()
    first["message"]["payload"]["pulse"]["wall_clock"]="2000-01-01T00:00:00Z"
    out=replay_fabricated_pulse(first,simulated_snap_at_utc=SNAP)
    assert out["provider_created_at_utc"]=="2027-09-09T19:59:50Z"
    assert out["snapshot"]["game_seconds_remaining"]==3098
    bad=deepcopy(first)
    bad["message"]["payload"]["pulse"]["created_at"]="2027-09-09T20:00:11+00:00"
    bad["collector_received_at_utc"]="2027-09-09T20:00:12Z"
    bad["captured_at_utc"]="2027-09-09T20:00:13Z"
    with pytest.raises(ValueError,match="BEFORE"):
        replay_fabricated_pulse(bad,simulated_snap_at_utc=SNAP)


def test_q2_uses_independent_coin_toss_not_next_kickoff():
    x=pulse_fixture()
    x["message"]["payload"]["game"]["quarter"]=2
    x["message"]["payload"]["pulse"]["period"]["number"]=2
    output=replay_fabricated_pulse(x,simulated_snap_at_utc=SNAP)
    assert output["snapshot"]["game_seconds_remaining"]==2198
    assert output["snapshot"]["second_half_receiver"]=="DEFENSE"
    assert output["source_events"][2]["payload"]["second_half_receiver"]=="DEFENSE"
    x["pregame_coin_toss"]["second_half_receiver_team"]="ABC"
    out=replay_fabricated_pulse(x,simulated_snap_at_utc=SNAP)
    assert out["snapshot"]["second_half_receiver"]=="OFFENSE"


def test_away_possession_and_own_yardline_are_correct():
    x=pulse_fixture()
    s=x["message"]["payload"]["pulse"]
    s["start_situation"]["possession"]["alias"]="DEF"
    s["possession"]["alias"]="DEF"
    # DEF is at own 6, 94 yards from ABC's end zone.
    s["start_situation"]["location"]={"alias":"DEF","yardline":6}
    s["location"]={"alias":"DEF","yardline":6}
    out=replay_fabricated_pulse(x,simulated_snap_at_utc=SNAP)
    assert out["snapshot"]["site"]=="AWAY"
    assert out["snapshot"]["site_advantage"]==-1
    assert out["snapshot"]["yardline_100"]==94
    assert out["snapshot"]["score_differential"]==4
    assert out["snapshot"]["posteam_timeouts_remaining"]==2


def test_physically_relocated_game_is_neutral_not_nominal_home():
    x=pulse_fixture()
    venue=x["physical_venue"]
    venue["physical_stadium_id"]="Relocated_Away_Stadium"
    venue["physical_neutral_site"]=True
    out=replay_fabricated_pulse(x,simulated_snap_at_utc=SNAP)
    assert out["snapshot"]["site"]=="NEUTRAL"
    assert out["snapshot"]["site_advantage"]==0
    assert out["source_events"][1]["payload"]["physical_neutral_site"] is True


def test_goal_to_go_on_opponent_fourth_and_six_from_six():
    x=pulse_fixture()
    s=x["message"]["payload"]["pulse"]
    s["yfd"]=6
    s["start_situation"]["yfd"]=6
    out=replay_fabricated_pulse(x,simulated_snap_at_utc=SNAP)
    assert out["snapshot"]["goal_to_go"]==1
    assert out["snapshot"]["yardline_100"]==6
    assert out["snapshot"]["ydstogo"]==6


@pytest.mark.parametrize("label,modify,expected",[
    ("outcome",lambda x:x["message"]["payload"]["pulse"].update({"observed_action":"PUNT"}),"Post-play"),
    ("real_mode",lambda x:x.update({"mode":"prospective"}),"NO REAL"),
    ("bad_type",lambda x:x["message"]["payload"]["pulse"].update({"event_type":"pass"}),"pre-snap"),
    ("play",lambda x:x["message"]["payload"].update({"event":{"play_type":"pass"}}),"pre-snap|Post-play"),
    ("event_outcome",lambda x:x["message"]["payload"]["pulse"].update({"event_context":{"outcome":"first_down"}}),"Post-play"),
    ("details",lambda x:x["message"]["payload"]["pulse"].update({"details":[{"yards":10}]}),"play details"),
    ("metadata",lambda x:x["message"]["metadata"].update({"operation":"delete"}),"event type"),
    ("playoff",lambda x:x["message"]["payload"]["game"].update({"game_type":"playoff"}),"pre-snap"),
    ("same_team",lambda x:x["message"]["payload"]["game"]["summary"]["away"].update({"alias":"ABC"}),"Duplicated"),
    ("missing_roof",lambda x:x["physical_venue"].update({"roof":None}),"venue roof"),
    ("roof_unrecognized",lambda x:x["physical_venue"].update({"roof":"unknown"}),"Actual roof"),
    ("physical_conflict",lambda x:x["physical_venue"].update({"physical_neutral_site":True}),"Physical location"),
    ("late_receive",lambda x:x.update({"collector_received_at_utc":"2027-09-09T20:00:10Z","captured_at_utc":"2027-09-09T20:00:11Z"}),"BEFORE"),
    ("late_source",lambda x:x["message"]["payload"]["pulse"].update({"created_at":"2027-09-09T20:00:01+00:00"}),"AFTER collector"),
    ("stale_play",lambda x:x["message"]["payload"]["pulse"].update({"created_at":"2027-09-09T19:58:00+00:00"}),"Stale"),
    ("stale_venue",lambda x:x["physical_venue"].update({"observed_at_utc":"2027-09-09T18:00:00Z"}),"Stale physical_venue"),
    ("stale_coin",lambda x:x["pregame_coin_toss"].update({"observed_at_utc":"2027-09-09T10:00:00Z","received_at_utc":"2027-09-09T10:00:01Z"}),"Stale halftime_receiver"),
    ("score_mismatch",lambda x:x["message"]["payload"]["pulse"].update({"away_points":32}),"scoreboard"),
    ("quarter_mismatch",lambda x:x["message"]["payload"]["pulse"]["period"].update({"number":2}),"quarters"),
    ("clock_mismatch",lambda x:x["message"]["payload"]["game"].update({"clock":"06:35"}),"clocks disagree"),
    ("field_error",lambda x:x["message"]["payload"]["pulse"]["start_situation"]["location"].update({"yardline":70}),"yard number"),
    ("bad_distance",lambda x:x["message"]["payload"]["pulse"]["start_situation"].update({"yfd":50}),"Pulse yfd|cross opponent goal"),
    ("not_fourth",lambda x:x["message"]["payload"]["pulse"].update({"down":3}),"down"),
    ("unallowed_receipt",lambda x:x["physical_venue"].update({"source_ref":"https://nfl.example.com/stadium"}),"synthetic"),
])
def test_fail_closed_source_errors(label,modify,expected):
    x=pulse_fixture()
    modify(x)
    with pytest.raises(ValueError,match=expected):
        replay_fabricated_pulse(x,simulated_snap_at_utc=SNAP)


@pytest.mark.parametrize("badclock",[
    "16:00","15:01","04:60","T04:59","00:00Z","04:59:10","invalid",
])
def test_game_clock_format_rejected(badclock):
    x=pulse_fixture()
    s=x["message"]["payload"]
    s["game"]["clock"]=badclock
    s["pulse"]["clock"]=badclock
    s["pulse"]["start_situation"]["clock"]=badclock
    s["pulse"]["game_clock"]["time"]=badclock
    with pytest.raises(ValueError,match="clock|minutes"):
        replay_fabricated_pulse(x,simulated_snap_at_utc=SNAP)


def test_missing_pulse_is_recorded_as_a_skipped_decision():
    first=pulse_fixture()
    missing={
        "game_id":"synthetic_unavailable","play_id":"p-002",
        "snap_at_utc":"2027-09-09T20:00:09Z","candidate":None,
    }
    late=deepcopy(first)
    late["collector_received_at_utc"]="2027-09-09T20:00:11Z"
    late["captured_at_utc"]="2027-09-09T20:00:12Z"
    out=audit_fabricated_opportunity_coverage([
        {"game_id":"synthetic_doc-fixture-001","play_id":"p-001",
         "snap_at_utc":SNAP,"candidate":first},
        missing,
        {"game_id":"synthetic_doc-fixture-001","play_id":"p-003",
         "snap_at_utc":"2027-09-09T20:00:10Z",
         "candidate":late},
    ])
    assert out["opportunities"]==3
    assert out["synthetic_source_ready"]==1
    assert out["missing_or_invalid"]==2
    assert out["readiness_fraction"]==round(1/3,6)
    assert out["reason_counts"]=={
        "ARRIVED_AT_OR_AFTER_SNAP":1,
        "MISSING_PULSE_EVENT":1,
        "READY_FOR_SYNTHETIC_FORECAST":1,
    }
    assert out["forecast_count"]==0
    assert all(not r["forecast_available"] for r in out["rows"])
    assert out["real_prospective_approval"] is False
    assert out["externally_witnessed_chain_root"] is False
    assert len(out["local_chain_root_sha256"])==64




def test_audit_classifies_stale_vs_post_snap_vs_wrong_event_with_safe_codes():
    def case(tag, f):
        candidate=pulse_fixture()
        f(candidate)
        return {
            "game_id":"synthetic_doc-fixture-001",
            "play_id":f"{tag}-candidate",
            "snap_at_utc":"2027-09-09T20:00:08Z",
            "candidate":candidate,
        }
    out=audit_fabricated_opportunity_coverage([
        case("stale",lambda x:x["message"]["payload"]["pulse"].update({
            "created_at":"2027-09-09T19:58:01+00:00",
        })),
        case("bad-event",lambda x:x["message"]["payload"]["pulse"].update({
            "event_type":"rush",
        })),
        case("result",lambda x:x["message"]["payload"]["pulse"].update({
            "observed_action":"PUNT",
        })),
    ])
    assert out["synthetic_source_ready"]==0
    assert out["reason_counts"]=={
        "STALE_PLAY_STATE":1,
        "NOT_VERIFIED_PREPLAY_PHASE":1,
        "POST_SNAP_PAYLOAD_REJECTED":1,
    }
    assert all("observed_action" not in str(row) for row in out["rows"])


def test_prospective_opportunity_ledger_deduplicates_and_requires_snap_order():
    row={"game_id":"synthetic_doc-fixture-001","play_id":"p-001",
         "snap_at_utc":SNAP,"candidate":pulse_fixture()}
    with pytest.raises(ValueError,match="Duplicate"):
        audit_fabricated_opportunity_coverage([row,deepcopy(row)])
    another={"game_id":"synthetic_other","play_id":"p-001",
             "snap_at_utc":"2027-09-09T20:00:07Z","candidate":None}
    with pytest.raises(ValueError,match="time ordered"):
        audit_fabricated_opportunity_coverage([row,another])


def test_chain_detects_deletions_or_changed_ids_if_root_is_externally_preserved():
    row={"game_id":"synthetic_doc-fixture-001","play_id":"p-001",
         "snap_at_utc":SNAP,"candidate":pulse_fixture()}
    another={"game_id":"synthetic_missing","play_id":"p-002",
             "snap_at_utc":"2027-09-09T20:00:09Z","candidate":None}
    full=audit_fabricated_opportunity_coverage([row,another])
    short=audit_fabricated_opportunity_coverage([row])
    changed=deepcopy(another)
    changed["play_id"]="p-changed"
    changed2=audit_fabricated_opportunity_coverage([row,changed])
    assert full["local_chain_root_sha256"]!=short["local_chain_root_sha256"]
    assert full["local_chain_root_sha256"]!=changed2["local_chain_root_sha256"]
    assert full["rows"][0]["local_chain_sha256"] == short["rows"][0]["local_chain_sha256"]


def test_known_good_fabricated_pulse_enters_registered_engine_and_signed_snap_gate():
    candidate=replay_fabricated_pulse(
        pulse_fixture(),simulated_snap_at_utc=SNAP
    )
    policy=build_synthetic_policy_forecast(
        {"mode":"synthetic","source_type":"fabricated_only",
         "policy_commit":POLICY_COMMIT},
        candidate["snapshot"],engine=load_verified_engine(),
        nuisance_snapshot=load_historical_parameters(),n_simulations=16,
    )
    bundle=make_synthetic_bundle(
        snapshot=candidate["snapshot"],
        source_events=candidate["source_events"],forecast=policy,
        captured_at_utc=CAPTURE,
        forecast_issued_at_utc="2027-09-09T20:00:02Z",
    )
    collector,witness=Ed25519PrivateKey.generate(),Ed25519PrivateKey.generate()
    collected=sign_collector(
        bundle,sealed_at_utc="2027-09-09T20:00:03Z",
        collector_private_key=collector,
    )
    packet=witness_collector(
        collected,witnessed_at_utc="2027-09-09T20:00:04Z",
        collector_public_key=public_key_bytes(collector),
        witness_private_key=witness,
    )
    verified=verify_witnessed_before_snap(
        packet,snap_at_utc=SNAP,
        trusted_collector_public_key=public_key_bytes(collector),
        trusted_witness_public_key=public_key_bytes(witness),
    )
    assert verified["real_independent_provenance_certified"] is False
    assert verified["source_count"]==3
    # Only after sealed we can join independently fabricated outcome labels.
    post=join_labels_only_after_witness(
        packet,snap_at_utc=SNAP,
        trusted_collector_public_key=public_key_bytes(collector),
        trusted_witness_public_key=public_key_bytes(witness),
        observed_plays=[{
            "game_id":"synthetic_doc-fixture-001","play_id":"p-001",
            "observed_action":"NORMAL_GO_PASS","no_play":False,
        }],
        final_games=[{
            "game_id":"synthetic_doc-fixture-001","season":2027,"week":1,
            "home_team":"ABC","away_team":"DEF",
            "home_score_final":27,"away_score_final":31,
            "status":"final","game_type":"REG",
        }],
    )
    assert post["synthetic_cohort"]["included_decisions"]==1
    assert post["synthetic_cohort"]["records"][0]["offense_win"]==0
    assert post["receipt_validation"]["real_independent_provenance_certified"] is False



def test_documentation_pulse_q4_uses_registered_native_kickoff_correction():
    x=pulse_fixture()
    game=x["message"]["payload"]["game"]
    pulse=x["message"]["payload"]["pulse"]
    game["quarter"]=4
    pulse["period"]["number"]=4
    game["clock"]="04:59"
    pulse["clock"]="04:59"
    pulse["start_situation"]["clock"]="04:59"
    pulse["game_clock"]["time"]="04:59"
    result=replay_fabricated_pulse(x,simulated_snap_at_utc=SNAP)
    assert result["snapshot"]["game_seconds_remaining"]==299
    engine=load_verified_engine()
    forecast=build_synthetic_policy_forecast(
        {"mode":"synthetic","source_type":"fabricated_only",
         "policy_commit":POLICY_COMMIT},
        result["snapshot"],engine=engine,
        nuisance_snapshot=load_historical_parameters(),
        n_simulations=200,  # Native patch smoke budget only; NOT 1,200 output
    )
    assert forecast["audit"]["kickoff_input_mode"]=="Q4_NATIVE_STANDARD_35"
    assert forecast["audit"]["seed"]==43
    assert forecast["audit"]["simulation_count"]==200
    assert forecast["audit"]["full_precision"] is False
    bundle=make_synthetic_bundle(
        snapshot=result["snapshot"],
        source_events=result["source_events"],forecast=forecast,
        captured_at_utc=CAPTURE,
        forecast_issued_at_utc="2027-09-09T20:00:02Z",
    )
    collector,witness=Ed25519PrivateKey.generate(),Ed25519PrivateKey.generate()
    collected=sign_collector(
        bundle,sealed_at_utc="2027-09-09T20:00:03Z",
        collector_private_key=collector,
    )
    witnessed=witness_collector(
        collected,witnessed_at_utc="2027-09-09T20:00:04Z",
        collector_public_key=public_key_bytes(collector),
        witness_private_key=witness,
    )
    report=verify_witnessed_before_snap(
        witnessed,snap_at_utc=SNAP,
        trusted_collector_public_key=public_key_bytes(collector),
        trusted_witness_public_key=public_key_bytes(witness),
    )
    assert report["status"].startswith("TWO_SYNTHETIC_ED25519_RECEIPTS")
    assert report["real_independent_provenance_certified"] is False


def test_no_live_api_fetch_or_2026_outcomes_in_candidate_adapter():
    s=Path("src/sportradar_pulse_synthetic_adapter.py").read_text()
    for forbidden in (
        "requests.get(","urlopen(","urllib.request",
        "nflreadpy.load","load_pbp(","api.sportradar.com/nfl",
        "Ed25519PrivateKey.generate(", "2026_native_kickoff",
    ):
        assert forbidden not in s
    assert "fabricated_only" in s
    assert "wall_clock" in s
