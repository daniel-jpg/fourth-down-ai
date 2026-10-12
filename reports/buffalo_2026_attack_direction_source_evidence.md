# Buffalo 2026: documented north–south field and per-snap direction evidence

**October 12, 2026 — research-only, with no changes to the frozen model.**

## Confirmed qualitative field orientation

We found two stronger and independent sources for the 2026 replacement Highmark Stadium:

- [Populous, the stadium architect](https://populous.com/highmarkstadium), explicitly describes its north and south end zones.
- [Associated Press, August 7, 2026 (Washington Post)](https://www.washingtonpost.com/sports/nfl/2026/08/07/bills-highmark-stadium/2467e0ae-9291-11f1-9fdc-0a725c989a7b_story.html), reports the new stadium's north–south field orientation, contrasting it with the old east–west alignment.
- [WBEN/Audacy August 2026 on-site reporting](https://www.audacy.com/wcmf/sports/buffalo-bills/bills-open-highmark-stadium-for-first-event) provides further on-site north/south end-zone context.

These reports establish a **rough north–south axis**, NOT a surveyed 0°/180° azimuth, angular uncertainty, orientation of the offense at a fourth down or field wind vector. The [pinned 2026 location registry](../src/stadium_weather_2026_reference.json) now records qualitative N/S evidence and these source URLs while keeping the **numerical Buffalo field axis NULL**. Neither the METAR integration nor the official fourth-down model is allowed to derive a precise headwind from a verbal compass description.

## Specific-play observations: source-hashed receipts

The [2026 per-snap evidence module](../src/nfl_2026_snap_direction_evidence.py) and [local CLI](../scripts/capture_buffalo_snap_direction_claim.py) let a user create an **unsigned research receipt** for a *particular* asserted Buffalo 2026 snap direction, without accessing football outcomes or downloading game plays.

Requirements: 2026 game date in Eastern Time, declared BUF venue and opponent, offense, fictional or actual explicitly supplied play ID, regulation quarter, game clock, UTC decision time, and 1–2 independently attributed local source files. A source record needs an ID, witness ID, source category (video frame, on-site camera, observer note), observation and source-availability timestamps, a NORTH or SOUTH end-zone claim, geographic landmark description, HTTPS citation, and relative path to a local source file.

The CLI ensures original files are bounded, local and not symlinked or directory-traversed; computes SHA-256 digests; rejects two conflicting end-zone assertions and reused witness IDs, source URLs or source file bytes; requires the observation no later than and within 150 seconds of the asserted decision; excludes post-play results/actions; and creates an immutable-name receipt with hashes and metadata only, **never the photo/video contents**.

A source obtained **after** the decision is labeled RETROSPECTIVE. Two agreeing user-supplied witnesses remain **self-reported, not authenticated**: timestamps, play identity, cited video provenance, stadium side and independent source attribution have NOT been proven. No precision or reliability is implied by exact-looking UTC strings or hashes.

### Example CLI (fictional values)

Create a directory evidence/ with one or two original local evidence files and a JSON manifest in that same directory. Each object has fields source_id, witness_id, source_type, artifact_file, artifact_observed_at_utc, artifact_available_at_utc, attack_end_claim, location_landmark_observed, source_url.

    PYTHONPATH=. python -m scripts.capture_buffalo_snap_direction_claim \
      --season 2026 --venue-team BUF \
      --opponent-team NE --offense-team BUF \
      --game-date-et 2026-10-12 --play-key fictionalplay459 \
      --quarter 4 --game-clock 04:59 \
      --decision-utc 2026-10-12T20:30:00Z \
      --witness-manifest-json evidence/witness_metadata.json \
      --out artifacts/buffalo_snap_direction_research.json

These are NOT an authentic NFL matchup, play, score or weather observation. Replace all identifying fields when recording an actual independently observed play. The original image/video/note remains local and is never uploaded through the CLI.

In the [Streamlit weather venue UI](../app.py), selecting the 2026 Buffalo stadium now shows the architect/AP-supported north–south orientation and a **research-only north/south end-zone claim**. An optional viewer can inspect a pasted research receipt and flag internal inconsistencies. **The receipt viewer cannot verify source authenticity because the source bytes remain local.**

## Why this does not yet yield a headwind adjustment

- The exact physical true-north field-axis bearing is still unmeasured.
- For any real play, the actual attack end and game-location witness evidence has not been authenticated in the project.
- A nearby airport METAR is not a calibrated stadium wind sensor. Wind may swirl at field level in a partially sheltered bowl.
- Even correctly signed wind physics does not establish empirically validated FG/GO/PUNT success or win-probability adjustments.
- The prospective/retrospective source-availability distinction and reserved 2026/2027 holdout studies remain protected.

Every evidence receipt permanently sets independent direction certification, numeric wind approval, trusted live forecast provenance, protected season evaluation and official forecast changes to **false**.

## Verification

- [Source and provenance unit tests](../tests/test_nfl_2026_snap_direction_evidence.py) exercise documented N/S architecture; missing exact bearing; timestamps, source-hash uniqueness, late evidence, witness conflict, malformed files, post-play label exclusion and fail-closed display.
- [Streamlit AppTest integration](../tests/test_streamlit_app_integration.py) checks north/south selection and receipt review cannot enter the frozen engine's state or alter its recommendation.
- [Dedicated read-only synthetic CI](../.github/workflows/2026_buffalo_snap_direction_receipt.yml) uses fictional evidence only, hashes local files and publishes the JSON receipt without source media or real NFL play results.
- Frozen GO/FG/PUNT artifacts, official action ranking and protected 2026 once-only studies remain untouched.

**Result:** Buffalo's qualitative north/south field is supported by strong external evidence, and we can reproducibly audit user-supplied play-direction claims without inventing exact compass bearings or validated weather effects.
