# Stadium-aware airport wind context: 2026 NFL venue research registry

**October 12, 2026.** This feature supplies approximate 2026 stadium geography for a separate weather research tool. It does not change the official GO/FG/PUNT model, trained weights, or any protected 2026/2027 study.

## Source and coverage

Our curated [2026 venue registry](../src/stadium_weather_2026_reference.json) includes **32 team venue suggestions and 32 approximate stadium-center coordinates**, with **31 approximate field-axis pairs** (Buffalo has an approximate center but no verified axis). The independent [greerreNFL/Stadiums](https://github.com/greerreNFL/stadiums) community dataset supplies the geographical facts, originally manually assembled using Google Earth. We pin external commit 0f3876437ab9c798b3482efb712acd54b27cb2b3 and its team_stadiums.csv blob 12b15041d31b181fe7a2ef06551e046042141158.

**These coordinates and bearings are approximate community observations, NOT independently surveyed true-north field directions.** The original stadium heading is treated strictly as an **unoriented field axis modulo 180 degrees**. Selecting one of two opposite compass ends A/B must be done separately for the specific offensive play. Home/away, quarter, score, yardline and the on-screen left/right attacking arrow never determine the actual physical bearing.

The registry does not import the upstream roof_type into game state. Retractable roof conditions must be entered independently for the actual game. The registry is a team-based menu only: international games and alternative/neutral/relocated sites must select the actual venue, not blindly use the scheduled home team's usual stadium.

## Known exception controls

**Buffalo:** The upstream dataset still points to the prior Highmark Stadium (BUF00). The Bills [opened the replacement in June 2026](https://www.buffalobills.com/news/impressed-by-every-aspect-of-it-bills-officially-open-highmark-stadium-with-ribbon-cutting-ceremony) and [began using it in 2026](https://www.buffalobills.com/news/buffalo-bills-to-host-detroit-lions-in-2026-home-opener-at-highmark-stadium). The new stadium center is now approximately **42.77304, -78.79219** per [OSM way 1339149058](https://mapcarta.com/es/W1339149058), crosschecked against [a separate coordinate reference](https://www.latlong.net/place/new-highmark-stadium-new-york-34252.html) (42.772999, -78.792305). Its center is about 0.43 km west of the retired venue, but its FIELD AXIS remains **null and unverified**. The app displays the replacement coordinates and refuses to calculate headwinds from an invented axis. Map geometry is not a surveyed football pitch. © OpenStreetMap contributors under ODbL ([attribution](https://www.openstreetmap.org/copyright)).

**Tennessee:** The existing Nissan Stadium is the Titans' **2026** venue, and the [new enclosed Nashville stadium opens in 2027](https://www.tennesseetitans.com/new-stadium/info). The entire registry rejects every season other than 2026, so it cannot accidentally reuse the old coordinates in the new stadium.

**Shared facilities:** LA Chargers/Rams and New York Giants/Jets use identical physical venue IDs. Las Vegas is indexed as **LV**, not legacy OAK.

## New components

- [src/nfl_stadium_weather_geometry.py](../src/nfl_stadium_weather_geometry.py): 2026-only venue source validation, provenance pin, approximate field geometry and separately selected compass attack bearing. It permanently disclaims independent survey, current attack-end authentication, actual game/roof verification and official weather authority.
- [scripts/capture_nfl_venue_awc_context.py](../scripts/capture_nfl_venue_awc_context.py): explicit venue team, exact 2026 season, attack-end A/B, airport ICAO, decision UTC, open roof, and one-shot NOAA response or offline fixture. It composes the existing timestamp/station/distance-gated METAR code, not a new forecast model.
- [app.py](../app.py): under Weather observations (optional), first enable manual weather and then select Show 2026 stadium compass lookup (research only). Choose the actual venue and the offense's physical attacking end. Approximate coordinates, both compass bearings and a CLI research command are displayed. For Buffalo, the app displays the new center but refuses any unverified wind bearing.
- [tests/test_nfl_stadium_weather_geometry.py](../tests/test_nfl_stadium_weather_geometry.py), [headless Streamlit integration tests](../tests/test_streamlit_app_integration.py) and [read-only GitHub CI](../.github/workflows/weather_2026_stadium_geometry.yml): check 32/31 source coverage, blocked Buffalo field bearing, two opposite ends, shared stadiums, rejected future years, no hidden engine inputs, and a complete **fabricated** 2026 Seattle/Lumen Field to KBFI airport METAR integration.

## All nine official 2026 NFL international fixture venues

The [NFL Football Operations international schedule](https://operations.nfl.com/programs-initiatives/international-growth/nfl-international-games) identifies **nine games in eight international stadiums**. The [pinned exact-match schedule](../src/2026_nfl_international_venue_schedule.json) and [source guard](../src/nfl_2026_international_venue_guard.py) do not inspect real NFL outcomes or pretend to know game-day field attack direction.

| NFL schedule date in ET | Teams | Actual scheduled venue |
| --- | --- | --- |
| Sept. 10 | SF–LAR | Melbourne Cricket Ground, Australia |
| Sept. 27 | BAL–DAL | Maracanã Stadium, Brazil |
| Oct. 4 | IND–WAS | Tottenham Hotspur Stadium, London |
| Oct. 11 | PHI–JAX | Tottenham Hotspur Stadium, London |
| Oct. 18 | HOU–JAX | Wembley Stadium, London |
| Oct. 25 | PIT–NO | Stade de France, Paris |
| Nov. 8 | CIN–ATL | Bernabéu Stadium, Madrid |
| Nov. 15 | NE–DET | FC Bayern Munich Arena, Munich |
| Nov. 22 | MIN–SF | Estadio Banorte, Mexico City |

A new **2026 venue context** option in the Streamlit weather panel shows the official international fixture and correct country/stadium without inventing coordinates or compass field axes. It blocks confusing the team's domestic home field with the overseas event.

The one-shot live NOAA stadium collector now requires an **NFL schedule date in ET** and **opponent team** before NOAA HTTP traffic. When the exact date and team-pair match any of the nine known international fixtures, the collector refuses a default domestic stadium lookup. A nonmatch does **not** verify that the game happened at a domestic home venue, or eliminate other neutral/international/relocated events. The published schedule is a venue source, never a witnessed pre-snap event, a climate sensor or protected 2026 football outcome.

## Example: Seattle 2026 nearby airport (not a verified NFL decision)

After independently checking the 2026 game's real venue, time, physical offense attacking end, roof and nearby airport report availability, the research wrapper can be run manually. For a hypothetical Seattle offense attacking the approximate field axis end A at 0 degrees true:

    PYTHONPATH=. python -m scripts.capture_nfl_venue_awc_context \
      --venue-team SEA --season 2026 --attack-end A \
      --game-date-et YYYY-MM-DD --opponent-team OTHER_TEAM \
      --station KBFI --decision-utc YYYY-MM-DDTHH:MM:SSZ \
      --roof outdoors --live-noaa-awc \
      --out artifacts/research_venue_weather.json

You must replace the placeholder with a real offset-aware UTC decision time. The command can legitimately refuse an observation if NOAA has no report within 90 minutes, published no report by the time of the decision, or the selected airport is over 30 km away.

The output is an **unsigned airport weather research receipt** with user-supplied geometry. Even if real NOAA HTTPS returns a valid airport METAR, the report is NOT a signed on-field weather sensor, verified direction at a snap, or evidence the official model should adjust any probability. Downloading the report after the decision is labeled retrospective context only.

## Remaining work

1. Independently review new 2026 Buffalo stadium coordinates/field orientation; cross-check all 31 community bearings and their true-north convention.
2. Build verified actual game-to-venue and per-snap attack-direction evidence, including changes of possession, ends and overseas/neutral games; never infer these from user-facing quarter or home/away fields.
3. Obtain reliably time-stamped on-field wind direction, gusts, rain/snow intensity and source calibration; airport METAR is only a secondary environmental crosscheck.
4. Validate any weather-enhanced GO/FG/PUNT model using separate future-approved out-of-time samples; retain protected 2026 Steps 66/83 and 2027 once-only study boundaries.

**Result:** The app can now display 32 approximate 2026 venue centers, but only 31 community-estimated field axes. Buffalo center-mapping was updated without falsely claiming a known field direction. Scheduled international games cannot silently use domestic home-stadium geometry. True attack direction and actual venue must still be explicitly verified. There is no weather-aware production recommendation.
