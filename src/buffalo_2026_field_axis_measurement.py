"""Research-only measured Buffalo field-axis geometry from two goal-line points.

Does NOT use stadium bounding polygons, public stadium-center coordinates
as if they were end zones, or cardinal descriptions as 0 degrees.
Requires two source-hashed, user-provided end-zone/goal-line midpoint
coordinates in WGS84 plus claimed uncertainty.

WGS84 meridional/prime-vertical curvature at the midpoint gives precise
local-meter offsets; the max distance is ~120m so higher-order
geodesic terms are negligible relative to field measurement uncertainty.
The output is a CANDIDATE axis with deliberately UNTRUSTED source flags,
NOT a certified surveying product, not a sensor reading, not a model input,
and not even proof either point is on the football field.

No NFL outcomes, API access, weather requests or 2027 games are used.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
import re
from typing import Any, Mapping

from .metar_decision_weather import utc_time
from .nfl_stadium_weather_geometry import resolve_stadium_location

YEAR = 2026
TEAM = "BUF"
CANDIDATE_ONLY = "USER_SUPPLIED_FIELD_GEOMETRY_CANDIDATE_NOT_SURVEY_VERIFIED"
EXPECTED_METERS = {
    "GOAL_LINE_MIDPOINTS_100YDS": 91.44,
    "BACK_END_LINE_MIDPOINTS_120YDS": 109.728,
}
MAX_SOURCE_BYTES = 12 * 1024 * 1024
MAX_POINT_ERROR_M = 12.0
MIN_POINT_ERROR_M = 0.50
SOURCE_IDS = re.compile(r"[A-Za-z0-9_-]{3,64}\Z")
SOURCE_METHODS = {
    "CLAIMED_SURVEY_GNSS",
    "CLAIMED_FIELD_GPS",
    "CLAIMED_GEOREFERENCED_IMAGERY",
}


def _finite(val: Any, low: float, high: float, name: str) -> float:
    if type(val) not in (int, float) or not math.isfinite(val):
        raise ValueError(f"{name} missing or not a finite real number")
    if not low <= val <= high:
        raise ValueError(f"{name} outside admissible range")
    return float(val)


def wgs84_field_vector_m(
    south_lat: float, south_lon: float,
    north_lat: float, north_lon: float,
) -> dict[str, float]:
    """Physical FROM south end zone TO north end zone, true north degrees.

    East offset uses WGS84 N cos(phi); north offset uses WGS84 M. This
    corrects the simple 'longitude degrees == latitude degrees' error.
    For ~100m local fields its truncation error is far below GPS accuracy.
    """
    la1 = _finite(south_lat, -90, 90, "south latitude")
    lo1 = _finite(south_lon, -180, 180, "south longitude")
    la2 = _finite(north_lat, -90, 90, "north latitude")
    lo2 = _finite(north_lon, -180, 180, "north longitude")
    if la2 <= la1:
        raise ValueError("NORTH end must be physically north of SOUTH end")
    if abs(lo2-lo1) >= 180 or abs(la2-la1) >= 1:
        raise ValueError("Not a stadium-scale WGS84 coordinate pair")
    latitude = math.radians((la1+la2)/2)
    a = 6378137.0
    eccentricity_sq = 6.69437999014e-3
    q = 1-eccentricity_sq*math.sin(latitude)**2
    north_radius = a*(1-eccentricity_sq)/q**1.5
    east_radius = a/math.sqrt(q)
    dy = math.radians(la2-la1)*north_radius
    dx = math.radians(lo2-lo1)*east_radius*math.cos(latitude)
    distance = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    return {
        "east_m_from_south_to_north": round(dx, 5),
        "north_m_from_south_to_north": round(dy, 5),
        "distance_m": round(distance, 5),
        "true_azimuth_south_to_north_deg": round(bearing, 5),
        "true_azimuth_north_to_south_deg": round((bearing+180)%360, 5),
    }


def _point(
    side: str, row: Any, source_bytes: bytes,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Each end-zone point needs a source metadata object")
    keys = {
        "latitude", "longitude", "horizontal_uncertainty_m",
        "source_id", "observer_id", "measurement_method",
        "source_url", "collected_at_utc",
        "landmark_description",
    }
    if set(row) != keys:
        raise ValueError("Unexpected field-measurement source fields")
    lat = _finite(row["latitude"], 42.770, 42.776, f"{side} latitude")
    lon = _finite(row["longitude"], -78.798, -78.788, f"{side} longitude")
    uncertainty = _finite(
        row["horizontal_uncertainty_m"],
        MIN_POINT_ERROR_M, MAX_POINT_ERROR_M, f"{side} position uncertainty",
    )
    for name in ("source_id", "observer_id"):
        if type(row[name]) is not str or not SOURCE_IDS.fullmatch(row[name]):
            raise ValueError(f"{side} {name} cannot be blank or malformed")
    if row["measurement_method"] not in SOURCE_METHODS:
        raise ValueError("Unapproved field coordinate provenance method")
    url = row["source_url"]
    if (
        type(url) is not str or not 12 <= len(url) <= 900
        or not url.startswith("https://")
        or "@" in url or "\n" in url or "\r" in url
    ):
        raise ValueError("Field coordinate source URL is not a bounded HTTPS citation")
    observed = utc_time(row["collected_at_utc"])
    if observed.year != 2026:
        raise ValueError("Cannot import a pre-2026 or future field observation")
    desc = row["landmark_description"]
    if (
        type(desc) is not str or not 18 <= len(desc) <= 600
        or side.lower() not in desc.lower()
        or not any(x in desc.lower() for x in ("goal line", "back end line"))
    ):
        raise ValueError(
            "Identify the actual north/south goal-line or back-line MIDPOINT"
        )
    if (
        type(source_bytes) is not bytes
        or not 8 <= len(source_bytes) <= MAX_SOURCE_BYTES
    ):
        raise ValueError("Original coordinate measurement artifact must be bounded bytes")
    return {
        "side": side,
        "latitude": lat,
        "longitude": lon,
        "horizontal_uncertainty_m_user_supplied": uncertainty,
        "source_id": row["source_id"],
        "observer_id": row["observer_id"],
        "measurement_method_user_asserted": row["measurement_method"],
        "source_url_user_asserted": url,
        "collected_at_utc_user_asserted": observed.isoformat().replace("+00:00", "Z"),
        "landmark_description_user_asserted": desc,
        "original_local_artifact_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "original_source_bytes_omitted": True,
        "source_authenticity_verified": False,
        "independent_field_survey_certified": False,
    }


def build_buffalo_field_axis_candidate(
    *,
    season: Any, venue_team: Any, landmark_type: Any,
    point_metadata: Mapping[str, Any], original_artifacts: Mapping[str, bytes],
) -> dict[str, Any]:
    """Calculate untrusted numeric geometry, preserving evidence barriers.

    Even a geometry-consistent pair remains USER-ASSERTED. Coordinate
    errors are taken at face value for indicative uncertainty only.
    """
    if type(season) is not int or season != YEAR:
        raise PermissionError("Buffalo physical-field measurement restricted to 2026")
    if type(venue_team) is not str or venue_team != TEAM:
        raise ValueError("Only the NEW 2026 Buffalo stadium is supported")
    if type(landmark_type) is not str or landmark_type not in EXPECTED_METERS:
        raise ValueError("Choose the same physical field landmark at both ends")
    if (
        not isinstance(point_metadata, Mapping)
        or set(point_metadata) != {"NORTH", "SOUTH"}
        or not isinstance(original_artifacts, Mapping)
    ):
        raise ValueError("Two explicit NORTH and SOUTH measured landmarks required")
    source_ids = [
        point_metadata[side].get("source_id")
        for side in ("NORTH", "SOUTH")
        if isinstance(point_metadata[side], dict)
    ]
    if len(source_ids) != 2 or set(source_ids) != set(original_artifacts):
        raise ValueError("Both point sources must have their exact original artifacts")
    location = resolve_stadium_location(team="BUF", season=2026)
    if location["field_axis_degrees_approx"] is not None:
        raise ValueError("Buffalo axis was already unexpectedly given a bearing")
    points = {
        side: _point(side, point_metadata[side],
                     original_artifacts[point_metadata[side]["source_id"]])
        for side in ("NORTH", "SOUTH")
    }
    north, south = points["NORTH"], points["SOUTH"]
    if (
        north["source_id"] == south["source_id"]
        or north["observer_id"] == south["observer_id"]
        or north["original_local_artifact_sha256"] ==
            south["original_local_artifact_sha256"]
        or north["source_url_user_asserted"] == south["source_url_user_asserted"]
    ):
        raise ValueError("Different independently claimed point/source identities required")
    vec = wgs84_field_vector_m(
        south["latitude"], south["longitude"],
        north["latitude"], north["longitude"],
    )
    # Both points should describe one NFL regulation playing field, not
    # the edges of the oval stadium roof, seating bowl or parking lot.
    distance = vec["distance_m"]
    uncertainty_sum = (
        north["horizontal_uncertainty_m_user_supplied"]
        + south["horizontal_uncertainty_m_user_supplied"]
    )
    expected = EXPECTED_METERS[landmark_type]
    if abs(distance-expected) > max(5.0, uncertainty_sum+1.0):
        raise ValueError(
            "End-zone/goal-line midpoint separation not consistent with "
            "NFL field geometry; likely stadium polygon or wrong landmark"
        )
    if uncertainty_sum >= distance/4:
        raise ValueError("Measured field bearing too uncertain to report")
    middle = (
        (north["latitude"]+south["latitude"])/2,
        (north["longitude"]+south["longitude"])/2,
    )
    from .metar_decision_weather import distance_km
    center_delta_m = distance_km(
        middle,
        (location["field_lat_approx"], location["field_lon_approx"]),
    )*1000
    if center_delta_m > 70:
        raise ValueError(
            "Measured goal-line center falls outside reasonable new Buffalo "
            "playing-field vicinity; perhaps using old Highmark Stadium"
        )
    north_azimuth = vec["true_azimuth_south_to_north_deg"]
    if min(north_azimuth, 360.0-north_azimuth) > 30:
        raise ValueError(
            "Two measured goal-line points not aligned with supported "
            "qualitative NORTH/SOUTH field orientation"
        )
    # Conservative worst-case bound for vector angular perturbation due
    # to the two reported positional errors. This is CONDITIONAL on
    # honest measurements and WGS84 centers, not an independent confidence
    # interval or an authenticated survey.
    angle_bound = math.degrees(
        math.asin(uncertainty_sum/(distance-uncertainty_sum))
    )
    if angle_bound > 12:
        raise ValueError(
            "User-reported coordinate uncertainty too large for useful azimuth"
        )
    return {
        "status": CANDIDATE_ONLY,
        "season": 2026,
        "venue": "NEW_HIGHMARK_STADIUM_BUFFALO",
        "landmark_type_user_asserted": landmark_type,
        "expected_field_separation_m": expected,
        "source_points": points,
        "wgs84_geometry": vec,
        "center_offset_from_2026_osm_stadium_approx_m": round(center_delta_m, 2),
        "position_uncertainties_user_claimed_m": {
            "north": north["horizontal_uncertainty_m_user_supplied"],
            "south": south["horizontal_uncertainty_m_user_supplied"],
        },
        "candidate_northward_azimuth_deg": north_azimuth,
        "candidate_southward_azimuth_deg": vec[
            "true_azimuth_north_to_south_deg"
        ],
        "candidate_worst_case_angle_error_deg_if_errors_true": round(
            angle_bound, 3
        ),
        "independently_surveyed_exact_field_bearing": False,
        "coordinate_provenance_independently_authenticated": False,
        "independent_source_coordinates_crosschecked": False,
        "two_witnesses_independence_verified": False,
        "goal_line_landmarks_verified_on_field": False,
        "attack_end_of_specific_nfl_play_verified": False,
        "snap_level_weather_source_verified": False,
        "may_use_candidate_as_trusted_METAR_direction": False,
        "may_change_official_go_fg_punt_model": False,
        "protected_2026_nfl_outcomes_examined": False,
        "real_2027_outcomes_examined": False,
        "source_contents_in_receipt": False,
        "caveat": (
            "Geometry correct ONLY IF human-entered landmark coordinates, "
            "WGS84 datum, north/south physical labels and stated position "
            "errors are true. Original evidence is locally hashed, but "
            "URLs, source independence, GPS accuracy, field midpoint "
            "placement and any NFL play direction are NOT authenticated. "
            "No numerical wind or win-probability adjustment is authorized."
        ),
    }
