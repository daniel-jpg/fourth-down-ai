"""Conservative 2026 NFL venue coordinates / unoriented field axes.

Research ONLY. Curated geographic estimates from an external community
Google-Earth reference; neither coordinates nor field direction are
independently surveyed or authenticated. There is NO 2027 venue assumption.

The original community data heading is treated as an *unoriented* axis.
A user must supply attack-end selection for the specific play. NEVER infer
from quarter, field-position yardline, HOME/AWAY, team identity or UI arrows.

The upstream source maps 2026 Buffalo to an old Highmark Stadium.
An independent 2026 OpenStreetMap approximate center supersedes that old
location, but there is NO verified new-field axis; BUF wind projection is
still deliberately blocked. Tennessee's 2026 mapping expires before 2027.
Neutral/relocated games require an explicit actual venue selection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VENUES_FILE = ROOT / "stadium_weather_2026_reference.json"
SUPPORTED_SEASON = 2026
EXPECTED_TEAMS = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
})
ATTACK_ENDS = ("TOWARD_AXIS_BEARING_A", "TOWARD_OPPOSITE_BEARING_B")
PROVENANCE_STATUS = "UNVERIFIED_GOOGLE_EARTH_COMMUNITY_ESTIMATE"


def _only_2026(year: Any) -> None:
    if type(year) is not int or year != SUPPORTED_SEASON:
        raise PermissionError(
            "Venue map is restricted to 2026; 2027 new/relocated stadiums "
            "must be reviewed separately. No protected football outcomes."
        )


def _finite(v: Any, low: float, high: float, label: str) -> float:
    if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f"Invalid {label} for stadium geometry")
    return float(v)


def load_stadium_reference(*, season: Any = SUPPORTED_SEASON) -> dict:
    _only_2026(season)
    if not VENUES_FILE.is_file() or VENUES_FILE.is_symlink():
        raise ValueError("Expected a plain pinned venue reference file")
    obj = json.loads(VENUES_FILE.read_text(encoding="utf-8"))
    if (not isinstance(obj, dict)
            or obj.get("schema_version") != "2026_unofficial_field_axis_lookup_v1"
            or obj.get("season") != 2026
            or not isinstance(obj.get("venues"), dict)
            or set(obj["venues"]) != EXPECTED_TEAMS):
        raise ValueError("2026 NFL venue list is incomplete or from a wrong season")
    provenance = obj.get("source")
    if (not isinstance(provenance, dict)
            or provenance.get("commit") !=
            "0f3876437ab9c798b3482efb712acd54b27cb2b3"
            or provenance.get("blob_sha1") !=
            "12b15041d31b181fe7a2ef06551e046042141158"):
        raise ValueError("Venue source provenance does not match pinned review")
    for team, v in obj["venues"].items():
        if not isinstance(v, dict) or v.get("team") != team:
            raise ValueError("Venue record does not match its selected team")
        if v.get("physical_game_roof_known") is not False:
            raise ValueError("Venue file cannot imply actual game-day roof state")
        if team == "BUF":
            ev = v.get("location_evidence")
            if (
                v.get("georeference_status") !=
                "NEW_2026_VENUE_APPROX_CENTER_DIRECTION_UNVERIFIED"
                or v.get("venue_id") != "BUF_NEW_2026_OSM_W1339149058"
                or v.get("field_axis_bearing_degrees_approx") is not None
                or not isinstance(ev, dict)
                or ev.get("locator") != "OpenStreetMap way/1339149058"
                or ev.get("center_only") is not True
                or ev.get("no_survey_or_field_axis") is not True
            ):
                raise ValueError(
                    "Buffalo 2026 center cannot be paired with a guessed field axis"
                )
            # A coarse geographic consistency bound rejects reuse of the
            # old Buffalo stadium west/east coordinate or a foreign venue.
            lat = _finite(v.get("lat"), 42.772, 42.774, "new Buffalo latitude")
            lon = _finite(v.get("lon"), -78.794, -78.791, "new Buffalo longitude")
            if abs(lat - 42.77304) > .0004 or abs(lon + 78.79219) > .0004:
                raise ValueError("Buffalo 2026 coordinate disagrees with cited map")

        else:
            if v.get("georeference_status") != PROVENANCE_STATUS:
                raise ValueError("Stadium bearing not flagged as approximate")
            _finite(v.get("lat"), -90, 90, "latitude")
            _finite(v.get("lon"), -180, 180, "longitude")
            _finite(
                v.get("field_axis_bearing_degrees_approx"),
                0, 179.999999, "unoriented field axis",
            )
    return obj


def stadium_menu(*, season: Any = SUPPORTED_SEASON) -> dict[str, str]:
    """Show 32 team home-venue locations; Buffalo axis is unresolved."""
    records = load_stadium_reference(season=season)["venues"]
    return {
        team: (
            f"{v['venue_name']} — {team}"
            + (" (center mapped; field bearing unknown)" if team == "BUF" else "")
        )
        for team, v in records.items()
    }


def resolve_stadium_location(*, team: Any, season: Any) -> dict[str, Any]:
    """Return approximate physical VENUE center; never infer offense heading.

    The geographic coordinates are useful for airport station distance
    filtering even when a stadium field bearing is unknown. Location
    alone cannot issue a headwind/crosswind calculation.
    """
    _only_2026(season)
    if type(team) is not str or team not in EXPECTED_TEAMS:
        raise ValueError("Select an explicitly listed 2026 venue")
    rec = load_stadium_reference(season=season)["venues"][team]
    has_axis = rec["field_axis_bearing_degrees_approx"] is not None
    return {
        "season": 2026,
        "team_venue_reference": team,
        "stadium_name": rec["venue_name"],
        "field_lat_approx": rec["lat"],
        "field_lon_approx": rec["lon"],
        "approximate_field_axis_available": has_axis,
        "field_axis_degrees_approx": rec["field_axis_bearing_degrees_approx"],
        "stadium_center_independently_surveyed": False,
        "official_fourth_down_model_unchanged": True,
        "no_2027_game_outcomes_used": True,
        "location_source": (
            "OSM_2026_NEW_BUFFALO_APPROX_CENTER" if team == "BUF"
            else "PINNED_UNVERIFIED_COMMUNITY_GOOGLE_EARTH"
        ),
    }


def resolve_stadium_geometry(
    *, team: Any, season: Any, offense_attack_end: Any,
) -> dict[str, Any]:
    """Choose compass direction from two opposite approximate axis bearings.

    The real end zone is NOT identified by this lookup. End A/B are
    reference labels, not a prediction from score/home/quarter/team.
    The caller must independently check current offense direction AND
    approximate community stadium field axis before using the result.
    """
    _only_2026(season)  # BEFORE access to source file / team validation
    if not isinstance(team, str) or team not in EXPECTED_TEAMS:
        raise ValueError("Select an explicitly listed 2026 venue")
    if offense_attack_end not in ATTACK_ENDS or type(offense_attack_end) is not str:
        raise ValueError("Offense attack end must be explicitly selected")
    record = load_stadium_reference(season=season)["venues"][team]
    if team == "BUF":
        raise ValueError(
            "Buffalo moved to its new Highmark Stadium in 2026; its "
            "approximate coordinates are now mapped, but the NEW field "
            "axis is NOT surveyed or direction-verified. Offense-relative "
            "wind calculation stays BLOCKED until a bearing is audited."
        )
    axis_a = _finite(
        record["field_axis_bearing_degrees_approx"],
        0, 179.999999, "field axis",
    )
    axis_b = (axis_a + 180.0) % 360.0
    bearing = axis_a if offense_attack_end == ATTACK_ENDS[0] else axis_b
    return {
        "status": "APPROXIMATE_2026_VENUE_GEOMETRY_CONTEXT_ONLY",
        "season": 2026,
        "team_venue_reference": team,
        "stadium_id": record["venue_id"],
        "stadium_name": record["venue_name"],
        "field_lat": record["lat"],
        "field_lon": record["lon"],
        "field_axis_bearing_a_approx_deg": axis_a,
        "field_axis_bearing_b_approx_deg": axis_b,
        "offense_attack_end_selected": offense_attack_end,
        "offense_attack_heading_true_deg_approx": bearing,
        "offense_attack_heading_user_confirmed_at_snap": False,
        "stadium_coordinates_independently_verified": False,
        "field_axis_orientation_independently_surveyed": False,
        "actual_stadium_roof_open_verified": False,
        "actual_game_venue_confirmed": False,
        "wind_from_airport_is_on_field_sensor": False,
        "future_2027_venue_approved": False,
        "real_2027_outcomes_accessed": False,
        "can_change_official_fourth_down_recommendation": False,
        "source": {
            "repository": "greerreNFL/stadiums",
            "commit": "0f3876437ab9c798b3482efb712acd54b27cb2b3",
            "blob_sha1": "12b15041d31b181fe7a2ef06551e046042141158",
        },
        "limitations": (
            "Manual community Google-Earth coordinates/axis only; "
            "choose actual stadium for relocated/neutral games. "
            "Offense direction cannot be inferred from NFL quarter, "
            "field position or whether offense is home team. "
            "Verify actual end-zone direction, stadium position and "
            "open roof independently before using airport wind vector."
        ),
    }
