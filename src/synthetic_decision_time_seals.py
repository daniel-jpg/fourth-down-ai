"""Pre-snap *synthetic-only* source provenance and two-party forecast receipts.

Distinct acquisition feeds supply (1) live play state, (2) physical venue
and roof, (3) pre-known halftime receiver. Source timestamps and per-source
data digest are bound into the forecast body. A collector then signs the body
with Ed25519; a DISTINCT independent witness signs the collector receipt.
After the play, a separately supplied snap timestamp must be LATER than the
witness receipt before any labels can be joined.

Both parties and all times are FABRICATED in current tests. Real independent
timing attestation requires a reviewed real-time provider, external witness
with trusted time, public-key custody and an independently timestamped snap.
Never treat a user-provided time or self-issued signature as proof of
prospective sealing, and never use this module on 2026 or real 2027 data.

NO network, scheduler, outcome fetching, public app mutation, model fitting,
or live inference: existing pinned-engine forecasts are injected explicitly.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

from src.fourth_down_cohort_builder import (
    PRE_SNAP_KEYS, REQUIRED_MANIFEST, build_synthetic_cohort,
)
from src.fourth_down_policy_study import POLICY_COMMIT
from src.frozen_2027_policy_bridge import (
    SNAPSHOT_KEYS, PINNED_ENGINE_BLOB, REGISTERED_APP_BLOB,
    validate_synthetic_snapshot,
)
from src.historical_nuisance_portable import EXPECTED_STATUS

VERSION = 1
MANIFEST = {
    "mode": "synthetic", "source_type": "fabricated_only",
    "policy_commit": POLICY_COMMIT, "schema_version": VERSION,
}
STATE_KEYS = frozenset({
    "game_id", "play_id", "season", "week", "quarter", "down",
    "offense_team", "game_seconds_remaining", "yardline_100",
    "ydstogo", "goal_to_go", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
})
VENUE_KEYS = frozenset({
    "home_team", "away_team", "roof", "physical_stadium_id",
    "nominal_home_stadium_id", "physical_neutral_site",
})
RECEIVER_KEYS = frozenset({"second_half_receiver"})
FEEDS = {
    "play_state": ("fabricated-playfeed", STATE_KEYS, 40),
    "physical_venue": ("fabricated-venuefeed", VENUE_KEYS, 3600),
    "halftime_receiver": ("fabricated-kickofffeed", RECEIVER_KEYS, 21600),
}
EVENT_KEYS = frozenset({
    "kind", "provider_id", "source_ref", "observed_at_utc",
    "received_at_utc", "payload_sha256", "payload",
})
BUNDLE_KEYS = frozenset({
    "manifest", "snapshot", "source_events", "forecast",
    "captured_at_utc", "forecast_issued_at_utc",
})
# No outcome/action field can hide inside seemingly harmless audit
# metadata. Keep in lockstep with build_synthetic_policy_forecast.
FORECAST_AUDIT_KEYS = frozenset({
    "source", "frozen_policy_commit", "engine_blob_sha1",
    "historical_nuisance", "simulation_count", "seed",
    "full_precision", "registered_app_blob", "kickoff_input_mode",
    "adaptive_escalated", "pilot_action_gap_z", "pilot_simulations",
    "default_kicker_context", "site", "goal_to_go", "roof",
    "second_half_receiver", "close_call_rule",
    "observed_action_access", "outcome_access", "prospectively_validated",
})
COLLECTOR_KEYS = frozenset({"bundle", "collector_receipt"})
FINAL_KEYS = frozenset({"bundle", "collector_receipt", "witness_receipt"})
RECEIPT_KEYS = frozenset({"signed", "signature_base64"})
COLLECTOR_SIGNED_KEYS = frozenset({
    "type", "bundle_sha256", "sealed_at_utc", "collector_key_fingerprint",
})
WITNESS_SIGNED_KEYS = frozenset({
    "type", "collected_sha256", "witnessed_at_utc", "witness_key_fingerprint",
})
TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z"
)
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _exact(obj: Any, fields: frozenset, label: str) -> None:
    if not isinstance(obj, dict) or set(obj) != fields:
        raise ValueError(f"{label} must have EXACT allowed keys; outcomes/post-snap fields forbidden")


def canonical(obj: Any) -> bytes:
    """Strict canonical encoding; never hash Python dict repr or allow NaN."""
    try:
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (ValueError, TypeError) as err:
        raise ValueError("Unhashable, non-JSON, or nonfinite forecast/evidence data") from err


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not TIMESTAMP_RE.fullmatch(value):
        raise ValueError(f"{name} must be strict UTC RFC3339 ending in Z")
    try:
        dt = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"Invalid UTC {name}") from exc
    if dt.year != 2027 or dt.tzinfo != timezone.utc:
        raise ValueError(f"{name} must be fabricated 2027 UTC chronology")
    return dt


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"Invalid {name}")
    if len(value) > 300:
        raise ValueError(f"Oversized {name}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX_SHA.fullmatch(value):
        raise ValueError(f"{label} must be hex SHA-256")
    return value


def _public_key(raw: bytes) -> Ed25519PublicKey:
    if not isinstance(raw, bytes) or len(raw) != 32:
        raise ValueError("Trusted Ed25519 public key must be exactly 32 raw bytes")
    return Ed25519PublicKey.from_public_bytes(raw)


def _fingerprint(public: bytes) -> str:
    _public_key(public)
    return hashlib.sha256(public).hexdigest()


def public_key_bytes(signing: Ed25519PrivateKey) -> bytes:
    if not isinstance(signing, Ed25519PrivateKey):
        raise ValueError("Ed25519 private key required for fabricated receipt")
    return signing.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _sign(fields: Mapping[str, Any], private: Ed25519PrivateKey) -> dict:
    if not isinstance(private, Ed25519PrivateKey):
        raise ValueError("Ed25519 signing key required")
    return {
        "signed": dict(fields),
        "signature_base64": base64.b64encode(
            private.sign(canonical(fields))
        ).decode("ascii"),
    }


def _verify_signature(receipt: Any, keys: frozenset, trusted_key: bytes) -> dict:
    _exact(receipt, RECEIPT_KEYS, "signed receipt")
    signed = receipt["signed"]
    _exact(signed, keys, "signed receipt body")
    sig = receipt["signature_base64"]
    if not isinstance(sig, str) or len(sig) > 100:
        raise ValueError("Malformed detached Ed25519 signature")
    try:
        decoded = base64.b64decode(sig, validate=True)
        _public_key(trusted_key).verify(decoded, canonical(signed))
    except (InvalidSignature, ValueError) as err:
        raise ValueError("Signed receipt authenticity check failed") from err
    return signed


def validate_source_bundle(bundle: Any) -> dict:
    """Prove fabricated state matches three hashed source payloads, as of capture.

    This validates claims internal to the signed file; independently
    trusted sources/timestamps do NOT exist in the synthetic fixture.
    """
    _exact(bundle, BUNDLE_KEYS, "forecast bundle")
    if bundle["manifest"] != MANIFEST:
        raise ValueError("Only explicitly fabricated pinned 2027 source mode allowed")
    raw = bundle["snapshot"]
    if not isinstance(raw, dict) or set(raw) != SNAPSHOT_KEYS:
        raise ValueError("Exact frozen synthetic snapshot required; no post-snap labels")
    validate_synthetic_snapshot(raw)
    if raw["source_type"] != "fabricated_only" or raw["feature_asof"] != "pre_snap":
        raise ValueError("Only synthetic pre-snap state supported")
    events = bundle["source_events"]
    if not isinstance(events, list) or len(events) != len(FEEDS):
        raise ValueError("Exactly three disjoint fabricated source events required")
    capture = _timestamp(bundle["captured_at_utc"], "captured_at_utc")
    issued = _timestamp(bundle["forecast_issued_at_utc"], "forecast_issued_at_utc")
    if capture > issued:
        raise ValueError("Forecast predates source capture")
    combined = {}
    kinds = set()
    for ev in events:
        _exact(ev, EVENT_KEYS, "source event")
        kind = ev["kind"]
        if kind not in FEEDS or kind in kinds:
            raise ValueError("Missing, duplicate or unknown source kind")
        kinds.add(kind)
        provider, fields, age_seconds = FEEDS[kind]
        if ev["provider_id"] != provider:
            raise ValueError("Unverified provider identity or cross-source alias")
        reference = _nonblank(ev["source_ref"], "source_ref")
        if not reference.startswith(f"synthetic://{kind}/"):
            raise ValueError("Only fabricated source identifiers allowed")
        payload = ev["payload"]
        _exact(payload, fields, "provider-specific source payload")
        if digest(payload) != _sha(ev["payload_sha256"], "payload_sha256"):
            raise ValueError("Source payload integrity mismatch")
        observed = _timestamp(ev["observed_at_utc"], "provider observed time")
        received = _timestamp(ev["received_at_utc"], "source receipt time")
        if not observed <= received <= capture:
            raise ValueError("Source received or observed after snapshot capture")
        if capture-observed > timedelta(seconds=age_seconds):
            raise ValueError(f"Stale {kind} source: failed decision-time freshness")
        # Disjoint source fields have an explicit owner and cannot be
        # inferred from outcomes or duplicated under another feed.
        if set(combined) & set(payload):
            raise RuntimeError("State field claimed by multiple source owners")
        combined.update(payload)

    if set(kinds) != set(FEEDS):
        raise ValueError("One of three source streams is absent")
    venue = events[[e["kind"] for e in events].index("physical_venue")]["payload"]
    if (not isinstance(venue["physical_neutral_site"], bool)
            or not all(_nonblank(venue[k],k) for k in (
                "physical_stadium_id","nominal_home_stadium_id"
            ))):
        raise ValueError("Physical stadium evidence incomplete")
    physical_neutral = venue["physical_neutral_site"]
    # In this fabricated test schema, venue location is explicitly
    # coded as neutral only when different from usual home stadium.
    if physical_neutral != (
        venue["physical_stadium_id"] != venue["nominal_home_stadium_id"]
    ):
        raise ValueError("Physical venue and nominal-home venue conflict")
    if (raw["site"] == "NEUTRAL") != physical_neutral:
        raise ValueError("Neutral-site flag must use physical stadium, not nominal home designation")
    offense = raw["offense_team"]
    expected_site = (
        "NEUTRAL" if physical_neutral else
        ("HOME" if offense == raw["home_team"] else "AWAY")
    )
    expected_advantage = {"HOME":1, "AWAY":-1, "NEUTRAL":0}[expected_site]
    if raw["site"] != expected_site or raw["site_advantage"] != expected_advantage:
        raise ValueError("Game-site orientation disagrees with physical stadium and offense")
    for k,v in combined.items():
        if k in ("physical_stadium_id", "nominal_home_stadium_id", "physical_neutral_site"):
            continue
        if k not in raw or type(raw[k]) is not type(v) and (
            isinstance(v,bool) or isinstance(raw[k],bool)
        ) or raw[k] != v:
            raise ValueError(f"Snap state {k} disagrees with source provider record")
    required_source_facts = (
        SNAPSHOT_KEYS-{"source_type","feature_asof","site","site_advantage"}
    )
    if (set(combined) - {
        "physical_stadium_id","nominal_home_stadium_id","physical_neutral_site"
    }) != required_source_facts:
        raise ValueError("Missing decision-time fact owner or extra provider field")
    prediction = bundle["forecast"]
    _exact(prediction, frozenset({"pre_snap", "audit"}), "pinned forecast")
    presnap = prediction["pre_snap"]
    _exact(presnap, PRE_SNAP_KEYS, "pre-snap action forecasts")
    if presnap["policy_commit"] != POLICY_COMMIT:
        raise ValueError("Forecast policy differs from pinned 2027 decision engine")
    for k in PRE_SNAP_KEYS.intersection(raw):
        if presnap[k] != raw[k]:
            raise ValueError(f"Frozen forecast input diverges from provider state: {k}")
    audit = prediction["audit"]
    _exact(audit, FORECAST_AUDIT_KEYS, "frozen policy audit metadata")
    if (audit["source"] != "SYNTHETIC ONLY; NO REAL FUTURE OUTCOMES"
            or audit["prospectively_validated"] is not False
            or audit["observed_action_access"] is not False
            or audit["outcome_access"] is not False
            or audit["frozen_policy_commit"] != POLICY_COMMIT
            or audit["historical_nuisance"] != EXPECTED_STATUS
            or audit["engine_blob_sha1"] != PINNED_ENGINE_BLOB
            or audit["registered_app_blob"] != REGISTERED_APP_BLOB
            or audit["default_kicker_context"] !=
               "PINNED ENGINE MODEL PRIOR, NO INDIVIDUAL KICKER DATA"
            or audit["roof"] != raw["roof"]
            or audit["goal_to_go"] != raw["goal_to_go"]
            or audit["second_half_receiver"] != raw["second_half_receiver"]
            or audit["site"] != raw["site"]
            or type(audit["full_precision"]) is not bool
            or type(audit["adaptive_escalated"]) is not bool
            or type(audit["simulation_count"]) is not int
            or not 2 <= audit["simulation_count"] <= 1200
            or type(audit["seed"]) is not int
            or audit["seed"] not in (42,43)
            or type(audit["pilot_simulations"]) is not int
            or not 0 <= audit["pilot_simulations"] <= 300
            or audit["kickoff_input_mode"] != (
                "Q4_NATIVE_STANDARD_35" if raw["quarter"] == 4
                else "NO_Q4_KICKOFF_PATCH"
            )):
        raise ValueError("Forecast evidence has unsafe mode or mismatched provenance")
    from src.fourth_down_cohort_builder import _validate_prespnap
    _validate_prespnap(presnap)
    if len(canonical(bundle)) > 200_000:
        raise ValueError("Oversized source or forecast evidence bundle")
    return {
        "game_id":raw["game_id"], "play_id":raw["play_id"],
        "source_count":len(events), "bundle_sha256":digest(bundle),
        "captured_at_utc":bundle["captured_at_utc"],
        "forecast_issued_at_utc":bundle["forecast_issued_at_utc"],
        "forecast_sha256":digest(prediction),
        "full_precision":audit.get("full_precision") is True,
        "status":"FABRICATED_PRE_SNAP_SOURCE_CONSISTENCY_ONLY",
    }


def make_synthetic_bundle(
    *, snapshot: dict, source_events: list, forecast: dict,
    captured_at_utc: str, forecast_issued_at_utc: str,
) -> dict:
    bundle={
        "manifest":dict(MANIFEST), "snapshot":snapshot,
        "source_events":source_events, "forecast":forecast,
        "captured_at_utc":captured_at_utc,
        "forecast_issued_at_utc":forecast_issued_at_utc,
    }
    validate_source_bundle(bundle)
    return bundle


def sign_collector(
    bundle: dict, *, sealed_at_utc: str,
    collector_private_key: Ed25519PrivateKey,
) -> dict:
    validate_source_bundle(bundle)
    seal_time = _timestamp(sealed_at_utc,"sealed_at_utc")
    issued = _timestamp(bundle["forecast_issued_at_utc"],"forecast_issued_at_utc")
    if seal_time < issued:
        raise ValueError("Collector seal precedes forecast issuance")
    public = public_key_bytes(collector_private_key)
    signed={
        "type":"fabricated_collector_receipt_v1",
        "bundle_sha256":digest(bundle),
        "sealed_at_utc":sealed_at_utc,
        "collector_key_fingerprint":_fingerprint(public),
    }
    return {
        "bundle":bundle,
        "collector_receipt":_sign(signed,collector_private_key),
    }


def witness_collector(
    collected: dict, *, witnessed_at_utc: str,
    collector_public_key: bytes, witness_private_key: Ed25519PrivateKey,
) -> dict:
    _exact(collected,COLLECTOR_KEYS,"collected pre-snap bundle")
    validate_source_bundle(collected["bundle"])
    witness_public=public_key_bytes(witness_private_key)
    if _fingerprint(witness_public)==_fingerprint(collector_public_key):
        raise ValueError("Witness cannot be its own collector")
    collector=_verify_signature(
        collected["collector_receipt"],COLLECTOR_SIGNED_KEYS,collector_public_key
    )
    if (collector["type"]!="fabricated_collector_receipt_v1"
            or collector["collector_key_fingerprint"]!=_fingerprint(collector_public_key)
            or collector["bundle_sha256"]!=digest(collected["bundle"])):
        raise ValueError("Collector signature not bound to this forecast bundle")
    sealed=_timestamp(collector["sealed_at_utc"],"sealed_at_utc")
    witness_time=_timestamp(witnessed_at_utc,"witnessed_at_utc")
    issued=_timestamp(collected["bundle"]["forecast_issued_at_utc"],"forecast_issued_at_utc")
    if not issued <= sealed <= witness_time:
        raise ValueError("Witness cannot attest before receipt or issuance of sealed forecast")
    receipt={
        "type":"fabricated_independent_witness_v1",
        "collected_sha256":digest(collected),
        "witnessed_at_utc":witnessed_at_utc,
        "witness_key_fingerprint":_fingerprint(witness_public),
    }
    return {**collected,"witness_receipt":_sign(receipt,witness_private_key)}


def verify_witnessed_before_snap(
    packet: Any, *, snap_at_utc: str,
    trusted_collector_public_key: bytes,
    trusted_witness_public_key: bytes,
) -> dict:
    """Verify both detached signatures, source timestamps and later snap.

    snap_at_utc is a SEPARATE factual reference in a hypothetical dry
    run. In real operation it would require independently attested
    play-clock/snap records, not a caller's assertion. This API never
    authorizes real mode.
    """
    _exact(packet,FINAL_KEYS,"witnessed synthetic forecast")
    if trusted_collector_public_key == trusted_witness_public_key:
        raise ValueError("Collector and independent witness must have different keys")
    info=validate_source_bundle(packet["bundle"])
    collected={
        "bundle":packet["bundle"],
        "collector_receipt":packet["collector_receipt"],
    }
    collector=_verify_signature(
        packet["collector_receipt"],COLLECTOR_SIGNED_KEYS,
        trusted_collector_public_key,
    )
    witness=_verify_signature(
        packet["witness_receipt"],WITNESS_SIGNED_KEYS,
        trusted_witness_public_key,
    )
    if (collector["type"]!="fabricated_collector_receipt_v1"
            or collector["collector_key_fingerprint"]!=_fingerprint(trusted_collector_public_key)
            or collector["bundle_sha256"]!=info["bundle_sha256"]
            or witness["type"]!="fabricated_independent_witness_v1"
            or witness["witness_key_fingerprint"]!=_fingerprint(trusted_witness_public_key)
            or witness["collected_sha256"]!=digest(collected)):
        raise ValueError("Signed receipts do not commit to source and forecast")
    issued=_timestamp(info["forecast_issued_at_utc"],"forecast_issued_at_utc")
    sealed=_timestamp(collector["sealed_at_utc"],"sealed_at_utc")
    witnessed=_timestamp(witness["witnessed_at_utc"],"witnessed_at_utc")
    snap=_timestamp(snap_at_utc,"independent snap_at_utc")
    if not issued <= sealed <= witnessed < snap:
        raise ValueError("Prediction was not witnessed BEFORE the snap")
    return {
        **info,
        "status":"TWO_SYNTHETIC_ED25519_RECEIPTS_AND_LATER_SNAP_CONSISTENT",
        "collector_receipt_sha256":digest(packet["collector_receipt"]),
        "witness_receipt_sha256":digest(packet["witness_receipt"]),
        "collector_key_fingerprint":collector["collector_key_fingerprint"],
        "witness_key_fingerprint":witness["witness_key_fingerprint"],
        "sealed_at_utc":collector["sealed_at_utc"],
        "witnessed_at_utc":witness["witnessed_at_utc"],
        "hypothetical_snap_at_utc":snap_at_utc,
        "real_independent_provenance_certified":False,
    }


def join_labels_only_after_witness(
    packet: dict, *, snap_at_utc: str,
    trusted_collector_public_key: bytes,
    trusted_witness_public_key: bytes,
    observed_plays: list[dict], final_games: list[dict],
) -> dict:
    """After proof validation only, hand signed pre-snap to synthetic builder."""
    receipt=verify_witnessed_before_snap(
        packet,snap_at_utc=snap_at_utc,
        trusted_collector_public_key=trusted_collector_public_key,
        trusted_witness_public_key=trusted_witness_public_key,
    )
    result=build_synthetic_cohort({
        "manifest":dict(REQUIRED_MANIFEST),
        "pre_snap":[packet["bundle"]["forecast"]["pre_snap"]],
        "observed_plays":observed_plays, "final_games":final_games,
    })
    return {"receipt_validation":receipt,"synthetic_cohort":result}
