# Fourth-Down AI: Manual weather inputs and experimental FG sensitivity

**October 12, 2026 — Research-only weather comparison.** The official frozen decision model and protected 2026/2027 evaluation gates are unchanged.

## New inputs
Under Game site / Stadium / roof, the app has **Weather observations (optional)**. For outdoor or open-roof games, enter observed **temperature in Fahrenheit (-25 to 115)**, **wind speed in mph (0 to 40)**, and conditions (unknown, clear/dry, cloudy, rain, snow, mixed precipitation, fog). These are **manual** readings, not an automatic weather API. Closed roofs and domes are excluded. Wind *direction* is unavailable and is not modeled.

**Rain, snow and fog are descriptive context only.** We do not assign invented precipitation probability penalties. Missing weather remains the default and does not alter the official result.

## Evidence, including the negative validation finding

The original [nflfastR weather schema](https://nflfastr.com/reference/fast_scraper.html) contains stadium-level temperature and wind, for outdoor or open-roof stadiums, sourced from Pro-Football-Reference. These are **game-level** observations, not guaranteed exact at the fourth-down snap.

Earlier [field-goal ablations](../src/19_train_field_goal_enhanced.py) considered weather, but the [frozen deployed FG model](../src/42_build_decision_engine.py) uses distance, roof and kicker inputs, not temperature/wind. We left that model unchanged.

A new [historical-only training and ablation module](../src/weather_fg_candidate_development.py), run in [CI #38288634919](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38288634919), tested clean outdoor/open-roof field-goal attempts with 2014–22 training and 2023–24 validation; **no 2025/2026/2027 outcomes** were used. The [pinned research coefficients](../data/fg_weather_2014_2024_research_only.json) show:

| Check | Observed |
| --- | ---: |
| Training attempts | 5,705 |
| Validation attempts | 1,186 |
| Baseline validation Brier | 0.102068 |
| Weather candidate validation Brier | 0.102114 |
| Baseline minus weather Brier | -0.000046 (weather slightly worse) |
| Game-cluster 95% CI for improvement | [-0.000552, +0.000510] |
| Base / weather validation log loss | 0.327716 / 0.327328 |
| Fitted FG clean-make log-odds effect, +10 mph wind | -0.187754 |
| Fitted FG clean-make log-odds effect, +20°F | -0.025996 |

**The weather model failed to improve validation Brier.** The small temperature coefficient is not a defensible causal or general predictive effect. These coefficients therefore remain **research-only**, not a promoted policy model.

## What the separate what-if does

The [isolated weather FG module](../src/weather_fg_sensitivity.py) applies the historical candidate's bounded clean-make *log-odds* sensitivity against **65°F / 5 mph**, capped to +/-0.55 log-odds. It adjusts only the FG make/miss/blocked split, preserves broken-play probability and sum-to-one normalization, and runs the existing original action simulator in a per-user private Python function namespace. Frozen weights, Q4 1,200-trial native kickoff inputs, GO and PUNT forecasts, and the official recommendation stay unchanged.

The UI shows **Weather-adjusted field goal comparison (experimental)** below the unchanged official result, including original vs what-if FG make rate, the scenario FG expected WP and its modeled leader. It states prominently that the candidate failed held-out Brier and is **not validated coaching advice**.

Limitations: This is not a full weather-aware policy. There is **no** modeled rain/snow/fog effect, wind direction, effect on pass, run, punt or return outcomes, instant stadium weather measurement, or future-season causal validation. Any Monte Carlo differences in non-FG rows should not be attributed to weather. No external provider, subscription, vendor email or protected holdout access was needed.

## Tests and future work

- [Research-only candidate workflow](../.github/workflows/weather_fg_historical_candidate.yml) fits and validates the historical coefficients without modifying production models.
- [Weather-sensitive Streamlit workflow](../.github/workflows/weather_fg_streamlit_research.yml) checks the optional UI, cache invalidation, roof controls, real-engine isolation, FG distribution normalization and production gating.
- [Unit tests](../tests/test_weather_fg_sensitivity.py) and [UI integration tests](../tests/test_streamlit_app_integration.py) verify that weather can be recorded and a bounded what-if run without mutating the official recommendation.
- To integrate weather into official decisions later, separately preregister and validate per-play weather, wind direction, precipitation and weather effects on GO/PUNT as well as FGs. Do not tune against protected 2026/2027 evaluations.

**Bottom line:** Weather conditions are now accepted as optional decision context, and temperature/wind can influence a labeled FG research what-if; no weather-induced official recommendation is claimed or silently substituted.
