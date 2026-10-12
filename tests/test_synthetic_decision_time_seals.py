"""Fabricated 2027 pre-snap evidence signing and independent-witness rehearsal.

No real 2027 play data, actual witness credentials, hardcoded signing key,
2026 native-kickoff gate, model fitting or production app changes.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pandas as pd
import pytest

from src.frozen_2027_policy_bridge import (
    build_synthetic_policy_forecast, load_verified_engine,
)
from src.historical_nuisance_portable import load_historical_parameters
from src.synthetic_decision_time_seals import (
    FEEDS, STATE_KEYS, VENUE_KEYS, RECEIVER_KEYS, MANIFEST, canonical,
    digest, make_synthetic_bundle, sign_collector, public_key_bytes,
    witness_collector, verify_witnessed_before_snap,
    join_labels_only_after_witness, validate_source_bundle,
)

STAMPS = {
    "play_state": ("2027-09-09T19:59:50Z","2027-09-09T19:59:52Z"),
    "physical_venue": ("2027-09-09T19:54:00Z","2027-09-09T19:54:01Z"),
    "halftime_receiver": ("2027-09-09T18:30:00Z","2027-09-09T18:31:00Z"),
}
CAPTURED = "2027-09-09T20:00:00Z"
ISSUED = "2027-09-09T20:00:02Z"
SEALED = "2027-09-09T20:00:03Z"
WITNESSED = "2027-09-09T20:00:04Z"
SNAPPED = "2027-09-09T20:00:08Z"


def fake_state(**overrides):
    s = {
        "game_id":"synthetic_2027_01_DEF_ABC","play_id":"1234",
        "season":2027,"week":1,"quarter":1,"down":4,
        "offense_team":"ABC","home_team":"ABC","away_team":"DEF",
        "site":"HOME","site_advantage":1,
        "game_seconds_remaining":3300.0,"yardline_100":6.0,
        "ydstogo":4.0,"goal_to_go":0,"score_differential":-4.0,
        "posteam_timeouts_remaining":3,"defteam_timeouts_remaining":2,
        "roof":"outdoors","second_half_receiver":None,
        "feature_asof":"pre_snap","source_type":"fabricated_only",
    }
    s.update(overrides)
    return s


class CheapEngine:
    def normalize_state(self,state):
        return dict(state)

    def action_eligible(self,state,action):
        if action == "NORMAL_GO_RUN" and state["ydstogo"]>3:
            return False,"run unsupported"
        if action == "PUNT" and state["yardline_100"]<20:
            return False,"punt unsupported"
        return True,""

    def recommend(self,state,n_simulations,seed):
        assert seed in (42,43)
        results=[]
        for action,prob in [
            ("NORMAL_GO_RUN",.52),("NORMAL_GO_PASS",.53),
            ("FIELD_GOAL",.58),("PUNT",.36)
        ]:
            ok,_=self.action_eligible(state,action)
            results.append({
                "action":action, "eligible":ok,
                "expected_win_probability":prob if ok else float("nan"),
                "mc_se":.005 if ok else float("nan"),
            })
        table=pd.DataFrame(results)
        table=pd.concat([
            table.loc[table["eligible"]].sort_values(
                "expected_win_probability",ascending=False
            ), table.loc[~table["eligible"]],
        ],ignore_index=True)
        return {"results":table}


def fabricate_events(state):
    venue = {k:state[k] for k in ("home_team","away_team","roof")}
    venue.update({
        "physical_stadium_id":"ABC_USUAL_HOME",
        "nominal_home_stadium_id":"ABC_USUAL_HOME",
        "physical_neutral_site":False,
    })
    payloads={
        "play_state":{k:state[k] for k in STATE_KEYS},
        "physical_venue":venue,
        "halftime_receiver":{k:state[k] for k in RECEIVER_KEYS},
    }
    output=[]
    for kind,(provider,_,_) in FEEDS.items():
        observed,received=STAMPS[kind]
        output.append({
            "kind":kind,"provider_id":provider,
            "source_ref":f"synthetic://{kind}/fabricated-2027-event-001",
            "observed_at_utc":observed,"received_at_utc":received,
            "payload":payloads[kind],
            "payload_sha256":digest(payloads[kind]),
        })
    return output


@pytest.fixture(scope="module")
def models():
    return load_historical_parameters()


@pytest.fixture
def evidence(models):
    state=fake_state()
    forecast=build_synthetic_policy_forecast(
        {"mode":"synthetic","source_type":"fabricated_only",
         "policy_commit":MANIFEST["policy_commit"]},
        state, engine=CheapEngine(),nuisance_snapshot=models,n_simulations=24,
    )
    events=fabricate_events(state)
    return make_synthetic_bundle(
        snapshot=state,source_events=events,forecast=forecast,
        captured_at_utc=CAPTURED,forecast_issued_at_utc=ISSUED,
    )


@pytest.fixture
def credentials():
    # Independently generated, IN MEMORY, fresh EACH test. Never versioned.
    collector=Ed25519PrivateKey.generate()
    witness=Ed25519PrivateKey.generate()
    return collector,witness


@pytest.fixture
def signed(evidence,credentials):
    collector,witness=credentials
    collected=sign_collector(
        evidence,sealed_at_utc=SEALED,collector_private_key=collector,
    )
    packet=witness_collector(
        collected,witnessed_at_utc=WITNESSED,
        collector_public_key=public_key_bytes(collector),
        witness_private_key=witness,
    )
    return packet


def verify(packet,credentials,snap_at=SNAPPED):
    collector,witness=credentials
    return verify_witnessed_before_snap(
        packet,snap_at_utc=snap_at,
        trusted_collector_public_key=public_key_bytes(collector),
        trusted_witness_public_key=public_key_bytes(witness),
    )


def test_whole_fabricated_forecast_is_bound_to_sources_and_two_key_receipts(signed,credentials):
    receipt=verify(signed,credentials)
    assert receipt["source_count"]==3
    assert receipt["game_id"]=="synthetic_2027_01_DEF_ABC"
    assert receipt["status"]=="TWO_SYNTHETIC_ED25519_RECEIPTS_AND_LATER_SNAP_CONSISTENT"
    assert receipt["real_independent_provenance_certified"] is False
    assert receipt["collector_key_fingerprint"] != receipt["witness_key_fingerprint"]
    assert receipt["forecast_sha256"]==digest(signed["bundle"]["forecast"])
    assert receipt["bundle_sha256"]==digest(signed["bundle"])
    assert receipt["full_precision"] is False
    assert receipt["sealed_at_utc"] < receipt["hypothetical_snap_at_utc"]


def test_join_factual_labels_only_after_verified_seals(signed,credentials):
    collector,witness=credentials
    output=join_labels_only_after_witness(
        signed,snap_at_utc=SNAPPED,
        trusted_collector_public_key=public_key_bytes(collector),
        trusted_witness_public_key=public_key_bytes(witness),
        observed_plays=[{
            "game_id":"synthetic_2027_01_DEF_ABC",
            "play_id":"1234","observed_action":"NORMAL_GO_PASS",
            "no_play":False,
        }],
        final_games=[{
            "game_id":"synthetic_2027_01_DEF_ABC",
            "season":2027,"week":1,
            "home_team":"ABC","away_team":"DEF",
            "home_score_final":24,"away_score_final":17,
            "status":"final","game_type":"REG",
        }],
    )
    result=output["synthetic_cohort"]
    assert result["included_decisions"]==1
    assert result["records"][0]["observed_action"]=="NORMAL_GO_PASS"
    assert result["records"][0]["policy_action"]=="FIELD_GOAL"
    assert result["records"][0]["offense_win"]==1
    assert output["receipt_validation"]["real_independent_provenance_certified"] is False


@pytest.mark.parametrize("inject",[
    ("source_events",0,"payload"),("snapshot",None,"score_differential"),
    ("forecast",None,"mu_policy"),
])
def test_post_signature_mutation_never_passes_signature_verification(signed,credentials,inject):
    changed=deepcopy(signed)
    kind,index,field=inject
    if kind=="source_events":
        changed["bundle"]["source_events"][index]["payload"]["score_differential"]+=2
    elif kind=="snapshot":
        changed["bundle"]["snapshot"][field]=-1.0
    else:
        changed["bundle"]["forecast"]["pre_snap"][field]=.21
    with pytest.raises(ValueError):
        verify(changed,credentials)


def test_modifying_non_predictive_signed_audit_is_still_detected(signed,credentials):
    changed=deepcopy(signed)
    changed["bundle"]["forecast"]["audit"]["default_kicker_context"]="TAMPERED"
    with pytest.raises(ValueError,match="commit|authentic"):
        verify(changed,credentials)


@pytest.mark.parametrize("actual_snap",[
    "2027-09-09T20:00:00Z",
    "2027-09-09T20:00:03Z",
    "2027-09-09T20:00:04Z",
    "2027-09-09T19:59:59Z",
])
def test_stale_or_post_snap_witness_prohibited(signed,credentials,actual_snap):
    with pytest.raises(ValueError,match="BEFORE"):
        verify(signed,credentials,actual_snap)


def test_forged_untrusted_witness_or_collector_fails(signed,credentials):
    different=Ed25519PrivateKey.generate()
    c,w=credentials
    with pytest.raises(ValueError,match="authenticity"):
        verify_witnessed_before_snap(
            signed,snap_at_utc=SNAPPED,
            trusted_collector_public_key=public_key_bytes(c),
            trusted_witness_public_key=public_key_bytes(different),
        )
    with pytest.raises(ValueError,match="authenticity"):
        verify_witnessed_before_snap(
            signed,snap_at_utc=SNAPPED,
            trusted_collector_public_key=public_key_bytes(different),
            trusted_witness_public_key=public_key_bytes(w),
        )


def test_cannot_act_as_own_independent_witness(evidence,credentials):
    c,_=credentials
    collected=sign_collector(
        evidence,sealed_at_utc=SEALED,collector_private_key=c,
    )
    with pytest.raises(ValueError,match="own collector"):
        witness_collector(
            collected,witnessed_at_utc=WITNESSED,
            collector_public_key=public_key_bytes(c),
            witness_private_key=c,
        )


def test_future_witness_receipt_cannot_be_backdated_before_collector(evidence,credentials):
    c,w=credentials
    collected=sign_collector(
        evidence,sealed_at_utc=SEALED,collector_private_key=c,
    )
    with pytest.raises(ValueError,match="before receipt"):
        witness_collector(
            collected,witnessed_at_utc=ISSUED,
            collector_public_key=public_key_bytes(c),
            witness_private_key=w,
        )


@pytest.mark.parametrize("kind",["play_state","physical_venue","halftime_receiver"])
def test_each_source_event_must_arrive_before_forecast(evidence,kind):
    changed=deepcopy(evidence)
    ev=next(x for x in changed["source_events"] if x["kind"]==kind)
    ev["received_at_utc"]="2027-09-09T20:00:05Z"
    with pytest.raises(ValueError,match="after snapshot"):
        validate_source_bundle(changed)


@pytest.mark.parametrize("kind",["play_state","physical_venue","halftime_receiver"])
def test_source_age_must_not_exceed_fixed_freshness_budget(evidence,kind):
    changed=deepcopy(evidence)
    ev=next(x for x in changed["source_events"] if x["kind"]==kind)
    minutes={"play_state":3,"physical_venue":180,"halftime_receiver":500}[kind]
    ev["observed_at_utc"]=(
        datetime.fromisoformat(CAPTURED.replace("Z","+00:00"))-
        timedelta(minutes=minutes)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    with pytest.raises(ValueError,match="Stale"):
        validate_source_bundle(changed)


@pytest.mark.parametrize("evil",[
    "observed_action","offense_win","converted","result",
    "yards_gained","post_snap_epa","final_home_margin",
])
def test_future_factual_data_rejected_from_source_payload(evidence,evil):
    changed=deepcopy(evidence)
    ev=changed["source_events"][0]
    ev["payload"][evil]=123
    ev["payload_sha256"]=digest(ev["payload"])
    with pytest.raises(ValueError,match="EXACT"):
        validate_source_bundle(changed)


def test_pseudo_source_reference_cannot_claim_live_espn(evidence):
    changed=deepcopy(evidence)
    changed["source_events"][0]["source_ref"]="https://site.api.espn.com/pretend"
    with pytest.raises(ValueError,match="fabricated"):
        validate_source_bundle(changed)


def test_source_hash_tampering_is_blocked(evidence):
    changed=deepcopy(evidence)
    changed["source_events"][0]["payload_sha256"]="a"*64
    with pytest.raises(ValueError,match="integrity"):
        validate_source_bundle(changed)


def test_missing_roof_or_kickoff_source_does_not_fall_back_to_defaults(evidence):
    changed=deepcopy(evidence)
    changed["source_events"][1]["payload"].pop("roof")
    with pytest.raises(ValueError,match="EXACT"):
        validate_source_bundle(changed)
    changed=deepcopy(evidence)
    changed["source_events"].pop()
    with pytest.raises(ValueError,match="Exactly three"):
        validate_source_bundle(changed)


def test_nominal_home_team_is_not_physical_neutral_venue(evidence):
    changed=deepcopy(evidence)
    changed["snapshot"]["site"]="NEUTRAL"
    changed["snapshot"]["site_advantage"]=0
    # Deliberately leave venue in normal home stadium
    changed["forecast"]["pre_snap"]["site"]="NEUTRAL"
    with pytest.raises(ValueError,match="Neutral-site"):
        validate_source_bundle(changed)


def test_documented_relocation_style_neutral_venue_has_site_zero(models):
    state=fake_state(
        site="NEUTRAL",site_advantage=0,
    )
    forecast=build_synthetic_policy_forecast(
        {"mode":"synthetic","source_type":"fabricated_only",
         "policy_commit":MANIFEST["policy_commit"]},
        state,engine=CheapEngine(),nuisance_snapshot=models,n_simulations=24,
    )
    events=fabricate_events(state)
    events[1]["payload"]["physical_stadium_id"]="RELOCATED_OTHER_CITY"
    events[1]["payload"]["physical_neutral_site"]=True
    events[1]["payload_sha256"]=digest(events[1]["payload"])
    bundle=make_synthetic_bundle(
        snapshot=state,source_events=events,forecast=forecast,
        captured_at_utc=CAPTURED,forecast_issued_at_utc=ISSUED,
    )
    assert validate_source_bundle(bundle)["source_count"]==3
    assert bundle["snapshot"]["site_advantage"]==0


def test_q2_receiver_source_known_from_before_halftime(models):
    state=fake_state(
        quarter=2,game_seconds_remaining=2100.0,
        second_half_receiver="DEFENSE",
    )
    forecast=build_synthetic_policy_forecast(
        {"mode":"synthetic","source_type":"fabricated_only",
         "policy_commit":MANIFEST["policy_commit"]},
        state,engine=CheapEngine(),nuisance_snapshot=models,n_simulations=24,
    )
    events=fabricate_events(state)
    bundle=make_synthetic_bundle(
        snapshot=state,source_events=events,forecast=forecast,
        captured_at_utc=CAPTURED,forecast_issued_at_utc=ISSUED,
    )
    assert validate_source_bundle(bundle)["source_count"]==3
    wrong=deepcopy(bundle)
    wrong["source_events"][2]["payload"]["second_half_receiver"]=None
    wrong["source_events"][2]["payload_sha256"]=digest(wrong["source_events"][2]["payload"])
    with pytest.raises(ValueError,match="receiver"):
        validate_source_bundle(wrong)


@pytest.mark.parametrize("wrong",[
    "2027-09-09T20:00:00+00:00","2027-09-09 20:00:00",
    "2027-09-09T20:00:00","2026-09-09T20:00:00Z",
    "2028-09-09T20:00:00Z","2027-02-30T20:00:00Z",
])
def test_malformed_or_non_2027_utc_timestamps_rejected(evidence,wrong):
    changed=deepcopy(evidence)
    changed["captured_at_utc"]=wrong
    with pytest.raises(ValueError,match="captured_at_utc"):
        validate_source_bundle(changed)


def test_unknown_pinned_policy_or_non_synthetic_mode_blocked(evidence):
    changed=deepcopy(evidence)
    changed["manifest"]["mode"]="prospective"
    with pytest.raises(ValueError,match="fabricated"):
        validate_source_bundle(changed)
    changed=deepcopy(evidence)
    changed["manifest"]["policy_commit"]="main"
    with pytest.raises(ValueError,match="fabricated"):
        validate_source_bundle(changed)


def test_no_real_data_network_or_private_keys_baked_into_research_code():
    source=Path("src/synthetic_decision_time_seals.py").read_text()
    for prohibited in (
        "urlopen(", "requests.get(", "load_pbp(",
        "nflreadpy", "private_bytes(", "Ed25519PrivateKey.from_private_bytes(",
        "2026-native", "Step 66", "Step 83",
    ):
        assert prohibited not in source
    assert "fabricated_only" in source
    assert "Ed25519" in source


def test_real_registered_q1_engine_can_be_forecast_and_then_sealed(models,credentials):
    # Slow real-engine integration: only fabricated 2027-shaped inputs,
    # deliberately lower n=16 than frozen app's 300/1200 precision.
    state=fake_state()
    actual=build_synthetic_policy_forecast(
        {"mode":"synthetic","source_type":"fabricated_only",
         "policy_commit":MANIFEST["policy_commit"]},
        state,engine=load_verified_engine(),
        nuisance_snapshot=models,n_simulations=16,
    )
    bundle=make_synthetic_bundle(
        snapshot=state,source_events=fabricate_events(state),
        forecast=actual,captured_at_utc=CAPTURED,
        forecast_issued_at_utc=ISSUED,
    )
    c,w=credentials
    collected=sign_collector(
        bundle,sealed_at_utc=SEALED,collector_private_key=c
    )
    packet=witness_collector(
        collected,witnessed_at_utc=WITNESSED,
        collector_public_key=public_key_bytes(c),
        witness_private_key=w,
    )
    verified=verify(packet,credentials)
    assert verified["forecast_sha256"]==digest(actual)
    assert verified["real_independent_provenance_certified"] is False
