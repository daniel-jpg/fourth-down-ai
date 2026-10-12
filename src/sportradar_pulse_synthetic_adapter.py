"""Offline Sportradar NFL Push Pulse *candidate* mapping and loss ledger.

No Sportradar subscription, network call, real-time NFL games, 2026 results,
or 2027 observations are accessed here. This consumes FABRICATED v7-shaped
messages and refuses all real/prospective source modes.

Crucial provider detail: Pulse wall_clock can refer to the PREVIOUS play.
Only locally recorded message receipt and a separately supplied simulated
snap reference are used for time-order checks. A Pulse-created-at timestamp
is still just a provider claim; neither is an independent time authority.

The sanitized snapshot and source events can be fed to
src.synthetic_decision_time_seals AFTER computing a registered-engine
synthetic forecast. They are not a real 2027 football data connector.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
import math
import re
from typing import Any, Mapping

from src.frozen_2027_policy_bridge import (
    SNAPSHOT_KEYS, validate_synthetic_snapshot,
)
from src.synthetic_decision_time_seals import (
    STATE_KEYS, VENUE_KEYS, RECEIVER_KEYS, FEEDS, _timestamp, digest,
)

CANDIDATE_PROVIDER = "sportradar_nfl_push_pulse_v7_DOCUMENTATION_FIXTURE_ONLY"
SOURCE_URL = "https://developer.sportradar.com/football/reference/nfl-push-pulse"
REQUIRED_ENVELOPE = frozenset({
    "mode", "provider", "season", "week", "message",
    "collector_received_at_utc", "captured_at_utc",
    "physical_venue", "pregame_coin_toss",
})
VENUE_METADATA = frozenset({
    "source_ref", "observed_at_utc", "received_at_utc",
    "physical_stadium_id", "nominal_home_stadium_id", "roof",
    "physical_neutral_site",
})
TOSS_METADATA = frozenset({
    "source_ref", "observed_at_utc", "received_at_utc",
    "second_half_receiver_team",
})
OPPORTUNITY_KEYS = frozenset({
    "game_id", "play_id", "snap_at_utc", "candidate",
})
POSTPLAY_FORBIDDEN = frozenset({
    "observed_action", "play_type", "yards_gained", "scoring_play",
    "outcome", "converted", "result", "epa", "statistics",
    "final_home_score", "final_away_score", "snap_at_utc",
    "touchdown", "fumble", "interception",
})
CLOCK = re.compile(r"^(?:[0-9]|0[0-9]|1[0-5]):[0-5][0-9]$")
ALIAS = re.compile(r"^[A-Z]{2,3}$")
MAX_MESSAGE = 75_000


def _exact(data: Any, keys: frozenset, label: str) -> None:
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError(f"{label} requires exact fabricated pre-snap schema")


def _record(data: Any, label: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be object")
    return data


def _integer(data: Any, label: str, low: int, high: int) -> int:
    if type(data) is not int or not low <= data <= high:
        raise ValueError(f"{label} must be exact integer {low}..{high}")
    return data


def _alias(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ALIAS.fullmatch(value):
        raise ValueError(f"{label} must be team alias")
    return value


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 90:
        raise ValueError(f"Missing/oversized {label}")
    if any(c in value for c in ("\n", "\r", " ", "/")):
        raise ValueError(f"Invalid characters in {label}")
    return value


def _clock(value: Any) -> int:
    if not isinstance(value, str) or not CLOCK.fullmatch(value):
        raise ValueError("Expected game clock M:SS or MM:SS within 15:00")
    m, s = map(int, value.split(":"))
    secs = m * 60 + s
    if secs > 900:
        raise ValueError("Game clock cannot exceed 15 minutes")
    return secs


def _provider_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("+00:00"):
        raise ValueError(f"{label} must be strict provider UTC +00:00")
    normalized = value[:-6] + "Z"
    return _timestamp(normalized, label)


def _safe_json_tree(value: Any) -> None:
    """Reject outcome-bearing keys even when hidden inside nested metadata."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in POSTPLAY_FORBIDDEN:
                raise ValueError(f"Post-play {k} field cannot enter pre-snap evidence")
            _safe_json_tree(v)
    elif isinstance(value, list):
        for item in value:
            _safe_json_tree(item)


def _wall_time_unused(pulse: Mapping[str, Any]) -> None:
    # Provider docs: Pulse wall_clock may be the snap of the PREVIOUS
    # play and may be updated only after a later event message is sent.
    # Do NOT compare this field with the candidate's snap or use it as
    # provider observation time. It is never exported into the snapshot.
    if "wall_clock" in pulse and not isinstance(pulse["wall_clock"], str):
        raise ValueError("Malformed unused previous-play wall_clock")


def _event(kind: str, payload: dict, observed: str, received: str,
           reference: str) -> dict:
    provider, expected_keys, _ = FEEDS[kind]
    _exact(payload, expected_keys, kind)
    if not reference.startswith(f"synthetic://{kind}/"):
        raise ValueError("All offline replay source refs must be synthetic")
    return {
        "kind": kind, "provider_id": provider, "source_ref": reference,
        "observed_at_utc": observed, "received_at_utc": received,
        "payload_sha256": digest(payload), "payload": payload,
    }


def replay_fabricated_pulse(envelope: Mapping[str, Any], *,
                            simulated_snap_at_utc: str) -> dict:
    """Build exact synthetic policy snapshot from Pulse-shaped pre-play data.

    Throws on missing or contradictory values; never guesses roof,
    timeout counts or second-half receiver. The snap timestamp is an
    explicitly FABRICATED independent reference, not authenticated proof.
    """
    _exact(envelope, REQUIRED_ENVELOPE, "source envelope")
    if (envelope["mode"] != "fabricated_only" or
            envelope["provider"] != CANDIDATE_PROVIDER):
        raise ValueError("NO REAL NFL feed access or prospective mode authorized")
    _integer(envelope["season"], "season", 2027, 2027)
    week = _integer(envelope["week"], "week", 1, 18)
    snap = _timestamp(simulated_snap_at_utc, "fabricated snap")
    received = _timestamp(envelope["collector_received_at_utc"], "collector receipt")
    captured = _timestamp(envelope["captured_at_utc"], "collector capture")
    if not received <= captured < snap:
        raise ValueError("Pulse receipt or capture not BEFORE synthetic snap")
    message = _record(envelope["message"], "Sportradar-shaped message")
    if len(json.dumps(message, separators=(",", ":"), allow_nan=False)) > MAX_MESSAGE:
        raise ValueError("Oversized unsafe candidate provider message")
    _exact(message, frozenset({"metadata", "payload"}), "Pulse message")
    meta = _record(message["metadata"], "Pulse metadata")
    if (meta.get("league") != "nfl" or
            meta.get("event_type") != "pulse" or
            meta.get("operation") != "update" or
            meta.get("version") != "v7" or
            meta.get("status") != "inprogress"):
        raise ValueError("Wrong provider event type, version or lifecycle")
    payload = _record(message["payload"], "Pulse payload")
    _exact(payload, frozenset({"game", "pulse"}), "Pulse pre-snap payload")
    _safe_json_tree(message)
    game = _record(payload["game"], "game")
    pulse = _record(payload["pulse"], "pulse")
    if (game.get("status") != "inprogress" or
            game.get("game_type") != "regular" or
            pulse.get("event_type") != "players_lining_up" or
            pulse.get("type") != "event" or
            pulse.get("source") != "SRUS"):
        raise ValueError("Only documented pre-snap players_lining_up event permitted")
    # The sample Pulse "details" array is empty on lineup events. Any
    # play statistics, later outcome or nonempty detail is NOT safe.
    if pulse.get("details", []) != []:
        raise ValueError("Nonempty Pulse play details cannot enter pre-snap evidence")
    _wall_time_unused(pulse)
    created = _provider_time(pulse.get("created_at"), "Pulse created_at")
    if created > received:
        raise ValueError("Pulse provider created_at AFTER collector receipt")
    if captured-created > timedelta(seconds=40):
        raise ValueError("Stale Pulse lineup update: cannot presume current fourth down")
    game_id = _id(game.get("id"), "provider game id")
    pulse_id = _id(pulse.get("id"), "provider pulse id")
    play_id = _id(pulse.get("play_id"), "provider play id")
    if game_id not in str(meta.get("match", "")):
        raise ValueError("Pulse metadata match ID does not identify game")
    home = _alias(_record(
        _record(game.get("summary"), "score summary").get("home"), "home summary"
    ).get("alias"), "home")
    away = _alias(_record(
        game["summary"].get("away"), "away summary"
    ).get("alias"), "away")
    if home == away:
        raise ValueError("Duplicated home and away teams")
    summary = game["summary"]
    for side in ("home", "away"):
        _integer(summary[side].get("points"), f"{side} score", 0, 150)
        _integer(summary[side].get("remaining_timeouts"),
                 f"{side} timeouts", 0, 3)
    home_points = summary["home"]["points"]
    away_points = summary["away"]["points"]
    if (pulse.get("home_points") != home_points or
            pulse.get("away_points") != away_points):
        raise ValueError("Inconsistent pre-snap scoreboard update")
    q = _integer(game.get("quarter"), "quarter", 1, 4)
    period = _record(pulse.get("period"), "Pulse period")
    if _integer(period.get("number"), "Pulse quarter", 1, 4) != q:
        raise ValueError("Game and Pulse quarters disagree")
    down = _integer(pulse.get("down"), "down", 4, 4)
    situation = _record(pulse.get("start_situation"), "starting play situation")
    if situation.get("down") != down or situation.get("clock") != pulse.get("clock"):
        raise ValueError("Pre-snap start situation and Pulse clock/down disagree")
    if game.get("clock") != pulse.get("clock"):
        raise ValueError("Game and Pulse clocks disagree: stale/out-of-order")
    current_clock = _clock(pulse.get("clock"))
    game_clock = _record(pulse.get("game_clock"), "game_clock")
    if game_clock.get("time") != pulse.get("clock") or game_clock.get("running") is not False:
        raise ValueError("Lineup update has running or inconsistent clock")
    offense = _alias(_record(
        situation.get("possession"), "start possession"
    ).get("alias"), "possessing offense")
    location = _record(situation.get("location"), "line of scrimmage")
    owning_side = _alias(location.get("alias"), "yardline ownership")
    if offense not in (home, away) or owning_side not in (home, away):
        raise ValueError("Missing or invalid possession/line team")
    line = _integer(location.get("yardline"), "line yard number", 1, 50)
    yfd = _integer(situation.get("yfd"), "yards to first down", 1, 100)
    if pulse.get("yfd") != yfd:
        raise ValueError("Pulse yfd and start situation disagree")
    field_distance = 100 - line if owning_side == offense else line
    if yfd > field_distance:
        raise ValueError("First down would cross opponent goal; inconsistent Pulse state")
    # Only the pre-play start_situation may supply location, never the
    # post-play end_situation. Match coarse source fields if present.
    if pulse.get("possession", {}).get("alias", offense) != offense:
        raise ValueError("Pulse possession and starting situation differ")
    if pulse.get("location", {}).get("alias", owning_side) != owning_side:
        raise ValueError("Pulse field-side location and starting situation differ")
    if pulse.get("location", {}).get("yardline", line) != line:
        raise ValueError("Pulse yardline and starting situation differ")
    venue = envelope["physical_venue"]
    _exact(venue, VENUE_METADATA, "physical venue evidence")
    toss = envelope["pregame_coin_toss"]
    _exact(toss, TOSS_METADATA, "pregame second-half receiving evidence")
    if type(venue["physical_neutral_site"]) is not bool:
        raise ValueError("Physical venue neutral flag must be explicit Boolean")
    for key in ("physical_stadium_id","nominal_home_stadium_id","roof"):
        _id(venue[key], f"venue {key}")
    if venue["roof"] not in ("outdoors", "open", "closed", "dome", "indoors"):
        raise ValueError("Actual roof status cannot be assumed")
    if venue["physical_neutral_site"] != (
        venue["physical_stadium_id"] != venue["nominal_home_stadium_id"]
    ):
        raise ValueError("Physical location and nominal home venue conflict")
    site = ("NEUTRAL" if venue["physical_neutral_site"] else
            "HOME" if offense == home else "AWAY")
    site_advantage = {"NEUTRAL":0,"HOME":1,"AWAY":-1}[site]
    half_receiver = _alias(toss["second_half_receiver_team"], "pregame receiver")
    if half_receiver not in (home, away):
        raise ValueError("Pregame second-half receiver must be one of playing teams")
    receiver = (
        ("OFFENSE" if half_receiver == offense else "DEFENSE")
        if q == 2 else None
    )
    state = {
        "game_id": "synthetic_" + game_id, "play_id":play_id,
        "season": 2027, "week": week, "quarter": q, "down": 4,
        "offense_team":offense, "home_team":home, "away_team":away,
        "site":site, "site_advantage":site_advantage,
        "game_seconds_remaining":(4-q)*900+current_clock,
        "yardline_100":float(field_distance), "ydstogo":float(yfd),
        "goal_to_go":int(yfd == field_distance),
        "score_differential":float(
            home_points-away_points if offense==home else away_points-home_points
        ),
        "posteam_timeouts_remaining": (
            summary["home" if offense==home else "away"]["remaining_timeouts"]
        ),
        "defteam_timeouts_remaining": (
            summary["away" if offense==home else "home"]["remaining_timeouts"]
        ),
        "roof":venue["roof"], "second_half_receiver":receiver,
        "source_type":"fabricated_only","feature_asof":"pre_snap",
    }
    if set(state) != SNAPSHOT_KEYS:
        raise RuntimeError("Translated Pulse state diverged from frozen input contract")
    validate_synthetic_snapshot(state)
    def _time_for_source(source: dict, kind: str) -> tuple[str, str]:
        observed = _timestamp(source["observed_at_utc"],f"{kind} observed")
        landed = _timestamp(source["received_at_utc"],f"{kind} received")
        if not observed <= landed <= captured:
            raise ValueError(f"{kind} observed/received after capture")
        if captured-observed > timedelta(seconds=FEEDS[kind][2]):
            raise ValueError(f"Stale {kind} evidence cannot be used as pre-snap")
        return source["observed_at_utc"], source["received_at_utc"]
    v_obs,v_rec = _time_for_source(venue, "physical_venue")
    t_obs,t_rec = _time_for_source(toss, "halftime_receiver")
    # Must pass exact disjoint field owners to existing synthetic seal.
    play_state = {k:state[k] for k in STATE_KEYS}
    physical_venue = {
        "home_team":home,"away_team":away,
        **{k:venue[k] for k in VENUE_KEYS-{"home_team","away_team"}},
    }
    receiver_data={"second_half_receiver":receiver}
    wire_digest=digest(message)
    source_events=[
        _event(
            "play_state",play_state,
            created.isoformat().replace("+00:00","Z"),
            envelope["collector_received_at_utc"],
            f"synthetic://play_state/pulse-{pulse_id}-{wire_digest}",
        ),
        _event(
            "physical_venue",physical_venue,v_obs,v_rec,venue["source_ref"],
        ),
        _event(
            "halftime_receiver",receiver_data,t_obs,t_rec,toss["source_ref"],
        ),
    ]
    return {
        "snapshot":state, "source_events":source_events,
        "source_wire_sha256":wire_digest,
        "provider_created_at_utc":created.isoformat().replace("+00:00","Z"),
        "collector_received_at_utc":envelope["collector_received_at_utc"],
        "captured_at_utc":envelope["captured_at_utc"],
        "simulated_snap_at_utc":simulated_snap_at_utc,
        "wall_clock_used_for_snap":False,
        "ready_for_synthetic_forecast_only":True,
        "real_provider_connected":False,
        "real_independent_timestamp_certified":False,
    }



def _safe_exclusion_code(exc: Exception) -> str:
    """Stable aggregate reason code; never log unsanitized vendor contents."""
    detail = str(exc)
    if "BEFORE synthetic snap" in detail:
        return "ARRIVED_AT_OR_AFTER_SNAP"
    if "Stale Pulse" in detail:
        return "STALE_PLAY_STATE"
    if "Stale physical_venue" in detail:
        return "STALE_VENUE_CONTEXT"
    if "Stale halftime_receiver" in detail:
        return "STALE_COIN_TOSS_CONTEXT"
    if "Post-play" in detail or "play details" in detail:
        return "POST_SNAP_PAYLOAD_REJECTED"
    if "players_lining_up" in detail or "event type, version" in detail:
        return "NOT_VERIFIED_PREPLAY_PHASE"
    if any(word in detail for word in (
        "scoreboard", "clocks disagree", "quarters disagree",
        "start situation", "possession", "yardline", "field-side",
    )):
        return "CONTRADICTORY_FOOTBALL_STATE"
    if any(word in detail for word in (
        "venue", "coin toss", "receiver", "roof",
    )):
        return "MISSING_OR_INVALID_CONTEXT"
    return "INVALID_OR_UNTRUSTED_SOURCE"


def audit_fabricated_opportunity_coverage(opportunities: Any) -> dict:
    """Account for every independently supplied FABRICATED fourth-down unit.

    A missing Pulse message still counts as a missing forecast, not a
    silently omitted decision. The row hash chain is a local tamper
    detector, NOT append-only storage or externally timestamped proof.
    """
    if not isinstance(opportunities, list) or not opportunities:
        raise ValueError("Need nonempty independently enumerated synthetic opportunities")
    rows,seen=[],set()
    previous="0"*64
    last_time=None
    for expected in opportunities:
        _exact(expected,OPPORTUNITY_KEYS,"independently listed opportunity")
        game_id=_id(expected["game_id"],"opportunity game id")
        play_id=_id(expected["play_id"],"opportunity play id")
        if (game_id,play_id) in seen:
            raise ValueError("Duplicate independently enumerated fourth down")
        seen.add((game_id,play_id))
        snap=_timestamp(expected["snap_at_utc"],"hypothetical snap")
        if last_time is not None and snap < last_time:
            raise ValueError("Independent opportunity log must be time ordered")
        last_time=snap
        state=None
        reason="MISSING_PULSE_EVENT"
        if expected["candidate"] is not None:
            try:
                state=replay_fabricated_pulse(
                    expected["candidate"],
                    simulated_snap_at_utc=expected["snap_at_utc"],
                )
                if (state["snapshot"]["game_id"] != game_id or
                        state["snapshot"]["play_id"] != play_id):
                    reason="PLAY_IDENTIFIER_MISMATCH"
                    state=None
                else:
                    reason="READY_FOR_SYNTHETIC_FORECAST"
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                # Never expose untrusted provider text in audit output;
                # stable codes make omission mechanisms measurable.
                reason=_safe_exclusion_code(exc)
        row={
            "game_id":game_id, "play_id":play_id,
            "simulated_snap_at_utc":expected["snap_at_utc"],
            "decision_status":reason,
            "wire_sha256":(state["source_wire_sha256"] if state else None),
            "forecast_available":False,
            "real_independent_provenance_certified":False,
        }
        previous=hashlib.sha256(
            (previous+":"+digest(row)).encode("ascii")
        ).hexdigest()
        rows.append({**row,"local_chain_sha256":previous})
    counts=dict(sorted(Counter(
        r["decision_status"] for r in rows
    ).items()))
    candidate_ready=counts.get("READY_FOR_SYNTHETIC_FORECAST",0)
    return {
        "status":"FABRICATED_SOURCE_COVERAGE_NOT_PROSPECTIVE",
        "opportunities":len(rows),
        "synthetic_source_ready":candidate_ready,
        "missing_or_invalid":len(rows)-candidate_ready,
        "readiness_fraction":round(candidate_ready/len(rows),6),
        "reason_counts":counts,
        "rows":rows,
        "local_chain_root_sha256":previous,
        "externally_witnessed_chain_root":False,
        "real_prospective_approval":False,
        "forecast_count":0,
    }
