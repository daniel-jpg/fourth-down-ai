# Buffalo 2026 playing-field azimuth: coordinate-calibration research

**Status October 12, 2026: code complete; no authenticated real-field GPS survey has been supplied.** The new Highmark Stadium has a documented approximate NORTH/SOUTH playing-field orientation but no independently surveyed true-north field-axis azimuth in the sources reviewed. The approximate OSM **stadium center** is not a usable field direction, and a polygon fitted to the outer building is not a measured playing-field axis.

Sources for this distinction include the [stadium architect Populous](https://populous.com/highmarkstadium), [independent Associated Press reporting](https://www.washingtonpost.com/sports/nfl/2026/08/07/bills-highmark-stadium/2467e0ae-9291-11f1-9fdc-0a725c989a7b_story.html), the [mapped venue location](https://mapcarta.com/es/W1339149058), and [a separate mapped center crosscheck](https://www.latlong.net/place/new-highmark-stadium-new-york-34252.html). None supplies independently authenticated NORTH and SOUTH goal-line midpoint coordinates.

## What was implemented

The [pure WGS84 field-axis geometry module](../src/buffalo_2026_field_axis_measurement.py) and [one-shot local-file CLI](../scripts/measure_buffalo_field_axis.py) accept original evidence for *both* physical midfield-cross-section landmarks:

- **NORTH** measured goal-line midpoint, and **SOUTH** measured goal-line midpoint, separated by **100 yards = 91.44 meters**, OR two physical BACK END-LINE midpoints, separated by **120 yards = 109.728 meters**. The script refuses to mix reference types.
- Each end must supply WGS84 latitude and longitude, asserted measurement method, explicit claimed horizontal position uncertainty in meters, source ID, observer ID, source URL, date/time in timezone-aware UTC, an accurate physical landmark description, and an original local evidence file (up to 12 MiB). The measurement files stay local; only their SHA256 hashes and human-supplied metadata go in the candidate JSON.
- Geographic checks require the correct 2026 Buffalo location, roughly north–south alignment from actual point geometry, a midpoint near the NEW stadium rather than its retired predecessor, and a measured span compatible with an NFL field. A stadium building polygon, arbitrary crowd/seating coordinate or east–west line cannot sneak into this calculator.
- The algorithm uses WGS84 latitude-dependent northing and easting scales. It calculates a candidate bearing clockwise from TRUE NORTH, from the physical south end to the north end, and the opposite bearing in the reverse direction. The actual survey azimuth remains unknown without real measurements.
- It calculates an **error bound conditional on the claimed input uncertainty**. The conservative angle bound in degrees is asin((error_N + error_S)/(baseline_m - error_N - error_S)), where error_N and error_S are the claimed horizontal positional errors. This is **NOT** an empirically calibrated confidence interval: false GPS errors or wrong landmarks invalidate it. The tool rejects overly imprecise coordinates.
- Mismatched source IDs/files, reused independent observers, reused media bytes, duplicate source citations, missing files, symlinks, unsafe relative paths, unsupported years and incorrect physical landmarks all result in an explicit rejection.

Every candidate is labeled **USER_SUPPLIED_FIELD_GEOMETRY_CANDIDATE_NOT_SURVEY_VERIFIED** and its trust gates permanently mark:
- independent surveyed exact field bearing = FALSE;
- source coordinate authenticity, landmark placement and source independence = UNVERIFIED;
- attacking end of any actual NFL snap = UNVERIFIED;
- permission to use as trusted NOAA wind direction = FALSE;
- permission to alter frozen GO, FG or PUNT model probabilities = FALSE;
- access to protected 2026 and real 2027 outcomes = FALSE.

No value from a candidate is promoted into the pinned 2026 stadium registry, the existing snap-witness module, NOAA METAR simulation or any official decision model.

## Input example (FICTIONAL, not actual Buffalo GPS)

Create a directory named evidence containing two local files named north_gps_raw.txt and south_gps_raw.txt; also create evidence/buffalo_goal_lines.json with this JSON structure:

~~~json
{
  "NORTH": {
    "latitude": 42.773445,
    "longitude": -78.792109,
    "horizontal_uncertainty_m": 2.0,
    "source_id": "northgps01",
    "observer_id": "observernorth01",
    "measurement_method": "CLAIMED_FIELD_GPS",
    "source_url": "https://example.org/north",
    "collected_at_utc": "2026-10-12T18:00:00Z",
    "landmark_description": "north physical goal line midpoint in a fictional GPS fixture",
    "artifact_file": "north_gps_raw.txt"
  },
  "SOUTH": {
    "latitude": 42.772635,
    "longitude": -78.792271,
    "horizontal_uncertainty_m": 2.0,
    "source_id": "southgps01",
    "observer_id": "observersouth01",
    "measurement_method": "CLAIMED_FIELD_GPS",
    "source_url": "https://example.org/south",
    "collected_at_utc": "2026-10-12T18:00:00Z",
    "landmark_description": "south physical goal line midpoint in a fictional GPS fixture",
    "artifact_file": "south_gps_raw.txt"
  }
}
~~~

**The example decimal coordinates above are synthetic and are NOT an authentic field survey.** They are deliberately plausible numbers near the stadium for a reproducible test. Run the tool only when local original source files actually exist:

~~~bash
PYTHONPATH=. python -m scripts.measure_buffalo_field_axis \
  --manifest-json evidence/buffalo_goal_lines.json \
  --landmark-type GOAL_LINE_MIDPOINTS_100YDS \
  --out artifacts/buffalo_untrusted_axis_candidate.json
~~~

The output will include candidate northward and southward true azimuths, measured 2-point separation, the claimed-accuracy-based angular error bound, and a source SHA256 for each local artifact. **It does not prove either observed endpoint is on the stadium playing surface.** It refuses to overwrite an existing evidence record.

## Verification and remaining boundary

- [Unit tests](../tests/test_buffalo_2026_field_axis_measurement.py) exercise true-north WGS84 math, 100-vs-120-yard physical separation, old-venue exclusion, flipped/end-swapped coordinates, east–west alignment, source URL and file hash uniqueness, geographic bounds, measurement uncertainty, symlink/directory traversal, and no 2027 year access.
- The [synthetic CI workflow](../.github/workflows/buffalo_2026_survey_geometry_rehearsal.yml) generates **invented** GPS files, uses the real CLI and verifies it emits only an untrusted JSON record, never video/GPS source file contents or authorized model input.
- The separate [Buffalo north/south snap attack-direction research](buffalo_2026_attack_direction_source_evidence.md) records only user-declared witnesses, never an authenticated NFL play. The two record types cannot be silently conflated.
- Future prerequisites: survey-grade end-zone coordinates and documented datum/accuracy, independently verifiable ground-truth playing-field landmarks, independently verified specific-play attack direction at the decision time, and calibrated on-field weather. Only after an appropriately registered out-of-time action-value experiment can weather features affect production forecasts.

**Conclusion:** The field bearing is now *computable* once defensible source measurements are supplied. The project still does not possess those real surveyed coordinates; Buffalo's production wind-direction adjustment remains disabled.
