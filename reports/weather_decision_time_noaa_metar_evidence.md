# Decision-time weather evidence: NOAA METAR, compass-relative wind and provenance

**October 12, 2026. Research infrastructure only.** The production GO/FG/PUNT model remains frozen. The [2014–2024 weather study](weather_go_punt_2014_2024_findings.md) did not demonstrate a reliable accuracy improvement, and this feature does not promote any weather model.

## Real data source and the main limitation

The [NOAA Aviation Weather Center data API](https://aviationweather.gov/data/api/) exposes recent **airport METAR** observations in JSON, typically reported around once per hour; NOAA describes a historical access window of up to 30 days. The manual collector requests only one explicitly selected station and six recent hours. For older historical research, the [Iowa Environmental Mesonet archive](https://mesonet.agron.iastate.edu/request/download.phtml) provides past ASOS/METAR data, but it warns that the archive has little quality control. A future historical import must be separately audited.

**An airport report is not an on-field NFL observation.** The nearest airport may have different wind, precipitation, sheltering or temperature. A METAR weather token reporting rain or snow at an airport is **not a measured field precipitation rate**; the absence of such a token is unknown, not verified dry.

## Implemented code

- [src/metar_decision_weather.py](../src/metar_decision_weather.py) parses bounded raw METAR reports, checks decoded wind against raw wind, verifies UTC observed/NOAA-received times, binds reported weather to original source-byte hashes, checks airport-to-field distance and computes true-bearing headwind and crosswind.
- [scripts/capture_awc_decision_weather.py](../scripts/capture_awc_decision_weather.py) explicitly performs a **single HTTPS GET** to NOAA if invoked with the live flag, or accepts an offline JSON fixture. It exclusively creates a research receipt and cannot overwrite an existing one. No background schedule, provider subscription or NFL data download is used.
- [tests/test_metar_decision_weather.py](../tests/test_metar_decision_weather.py) tests mislabeled/missing/variable wind groups, rain and snow present-weather words, bad time zones, future/stale NOAA reports, wrong station, distance mismatches, conflicting original bytes, exact signed wind-vector arithmetic and unsigned receipt tampering.
- [CI workflow](../.github/workflows/metar_weather_evidence.yml) runs offline only on fabricated weather reports and explicitly verifies that no official prediction or authenticated on-field evidence is claimed.
- [Streamlit app](../app.py) now offers an optional **Inspect timestamped airport METAR evidence** checkbox after manual weather entry. Paste the JSON receipt to inspect the airport, observation time, distance, computed head/crosswind and airport precipitation label. **The pasted receipt never changes either the official strategy or the separate experimental FG what-if.**

## Fail-closed controls

1. **Station and source:** The requested four-character ICAO station must match both the NOAA decoded ID and raw METAR header. Original response bytes are SHA-256 hashed and must match the decoded report list. Raw report hashes are checked again for display.
2. **Time:** The observation AND NOAA provider receipt must be no later than the claimed decision time, and the observation no more than **90 minutes old**. Source retrieval timestamp is stored independently. A download after the decision is explicitly labeled **retrospective context only**. This does not prove that a live model possessed the weather before the snap.
3. **Location:** An airport station farther than **30 km** from the supplied field coordinates is rejected. Stadium coordinates, offense heading and decision timestamp are user-supplied and **not independently authenticated**.
4. **Compass direction:** METAR gives wind direction **FROM true north, in knots**. For a supplied offense attack bearing, positive signed headwind opposes the offense/kick; negative means tailwind. Positive signed crosswind is wind arriving from the offense's right. One knot is 1.150779448 mph. Variable wind direction produces null directional projections, not a guessed crosswind.
5. **Precipitation:** Raw present-weather tokens such as -RA and SN are **airport report categories only**. Remark-only precipitation text is not automatically a present-weather code. Neither rainfall nor snowfall **at the field per hour** is inferred.
6. **Governance:** Result fields always mark source-receipt authenticity, independent on-field measurement, independent live weather witness, official weather model promotion and real 2027 NFL outcomes as **false or unproven**.

## How to use the manual collector

Supply an authentic recent decision timestamp, independently confirmed stadium location and the **actual offense attack bearing clockwise from true north**. The following example date, coordinates and field heading are *illustrative*, not authenticated NFL snap coordinates or heading:

    PYTHONPATH=. python -m scripts.capture_awc_decision_weather \
      --live-noaa-awc --station KSEA \
      --decision-utc 2026-10-12T18:35:00Z \
      --field-lat 47.595 --field-lon -122.332 \
      --offense-heading-true-deg 90 --roof outdoors \
      --out artifacts/airport_weather_research_only.json

Since NOAA's recent-observations endpoint is time-limited, this command may reject the example date or return no weather evidence. Replace it with the actual **recent** decision and authoritative source context. After it creates a receipt, paste the saved JSON into the app's optional viewer.

Run local regression tests with:

    PYTHONPATH=. python -m pytest -q tests/test_metar_decision_weather.py

The CI workflow runs the full CLI with a deliberately **fictional** METAR and verifies that the receipt cannot claim real station authentication, on-field rain intensity or a validated weather-aware prediction.

## Remaining research before changing official recommendations

- Independently survey each stadium's field-to-true-north end-zone orientation and which end zone the actual offense attacks, including quarter-by-quarter switches. Field position and home/away do not determine the compass bearing.
- Obtain a signed, calibrated **on-field** weather sensor/event feed, including wind vector, gust and quantitative precipitation near each snap. NOAA airport METAR may serve as a crosscheck, not substitute for a field sensor.
- Build the historical source-to-game crosswalk with original retrieval hashes and temporal availability. For past observations sourced from an archive, say **retrospective**, not independently witnessed before snap.
- Pre-register out-of-time GO/FG/PUNT decision-value calibration and missingness audits, with the current fixed 2026 and 2027 evaluation gates respected. Do not tune the official frozen engine against those protected outcomes.

**Result:** Actual timestamped airport evidence can now be captured and audited; its wind can be correctly decomposed into offense-relative directions when field bearing is supplied. No unvalidated meteorological input influences production forecasts.
