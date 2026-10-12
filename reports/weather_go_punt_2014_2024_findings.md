# GO and PUNT weather ablation (2014–2024 historical development)

**Verified October 12, 2026.** [Dedicated GitHub Actions study #38290647808](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38290647808) passed. The exact machine-readable validation metrics, cohort counts, game-cluster intervals and source provenance are pinned in [weather_go_punt_2014_2024_metrics.json](weather_go_punt_2014_2024_metrics.json).

**Bottom line:** The weather models are NOT promoted to the official fourth-down AI. Temperature/wind features worsened historical GO conversion Brier score and had a very small, statistically inconclusive gross-punt-distance MAE improvement. Adding conservative rain/snow text features did not change this conclusion. No trusted **headwind/tailwind/crosswind** direction measurement exists in the archived input columns.

## Methods and provenance

- [GO/PUNT development script](../src/weather_go_punt_development.py) reads only the existing local fourth-downs labeled parquet. Historical seasons are filtered to **2014–2024 before collected play outcomes are materialized**. Train = 2014–2022; validation = 2023–2024. No uncompleted 2026 or future 2027 games are downloaded, and the reserved once-only 2026 evaluations are never called.
- Only outdoor and retractable-open games with complete reasonable stadium-level Fahrenheit temperature and wind speed are used. **Both baseline and candidate models have identical train/validation rows.** GO conditions on actual recorded normal GO-run/GO-pass actions. Target: whether the actual play converted. The observed choice to run/pass was made by coaches and cannot establish the alternative action's causal outcome.
- PUNT sample consists of recorded ordinary-execution punts with a physical positive **same-play GROSS kick distance** (a directly present historical nflverse field). Target: gross punt yards, **not** net punt yards, ensuing opponent field position or change in win probability. This restriction avoids inventing next-state labels from an earlier audit table that lacks the reconstructed fields.
- GO uses the same logistic-regression family for baseline and candidates; PUNT uses the same gradient-boosted regression family. The baseline includes game situation, field position, timeouts, quarter, roof/site and actual run/pass indicator for GO. Candidate 1 adds continuous temperature and wind. Candidate 2 also adds conservative indicators of rain/snow/dry words and an unspecified-weather flag. These are **research comparators** and are **not** the actual frozen GO/PUNT models.
- Candidate error improvements are baseline error minus candidate error. Positive is better. Intervals are 350-bootstrap-draw **game-cluster** 95% intervals on 2023–24 validation games. None of the weather improvements is securely above zero.

## Out-of-time validation results

| Recorded historical action / metric | Baseline | Temp + wind | Temp + wind + precipitation text |
| --- | ---: | ---: | ---: |
| GO conversion Brier (lower is better) | 0.216460 | 0.216981 | 0.217123 |
| GO conversion log loss (lower is better) | 0.620889 | 0.621978 | 0.622199 |
| PUNT gross kick distance MAE in yards (lower is better) | 6.37037 | 6.33963 | 6.33869 |
| PUNT gross kick distance RMSE in yards | 8.22959 | 8.17506 | 8.16722 |

Weather candidate changes and uncertainty:

| Metric: baseline minus candidate | Temperature + wind | Add precipitation text |
| --- | ---: | ---: |
| GO Brier improvement | **-0.000522** | **-0.000664** |
| GO 95% game-cluster interval | [-0.001451, +0.000278] | [-0.001676, +0.000294] |
| PUNT MAE improvement, yards | **+0.0307 yd** | **+0.0317 yd** |
| PUNT 95% game-cluster interval, yards | [-0.0210, +0.0756] | [-0.0205, +0.0802] |

**GO:** 4,891 original outdoor normal GO attempts, 4,467 weather-complete cases after excluding 424 with missing/out-of-range weather or incomplete preplay features; train **3,531**, validation **936** (324 unique validation games). The weather-enhanced models were slightly worse and intervals include no improvement.

**PUNT:** 18,908 original outdoor normal punts, 17,371 with complete weather and valid gross punt target after 1,537 exclusions; train **14,635**, validation **2,736** (338 validation games). The ~0.03-yard MAE change is negligible compared with ~6.37-yard baseline MAE and its uncertainty interval includes negative improvement. It is not evidence that adding weather will improve the official punt win-probability estimates.

## Precipitation and wind-direction limits

The [text parser](../src/weather_pbp_signal.py) only assigns a *description-mentions-rain* or *description-mentions-snow/ice* category when weather words explicitly appear. It does not count "chance of rain", "possible snow", or missing/overcast text as confirmed weather. Unknown conditions are not imputed dry.

Among the **validation weather-complete** plays, descriptions contained rain words for **34 GO plays and 219 punts**, snow/ice words for **23 GO plays and 48 punts**; another **564 GO plays and 1,541 punts** were textually unspecified. A stadium-level game description is **not proof of rain, snow or wind direction at the snap**; intensity and the effect of wet footing on different football actions are not identified here.

The archived input schema returned **only the field named wind** among wind/direction candidate columns. This records wind *speed*, not orientation relative to the kicking or offensive direction at a specific fourth down. The [Streamlit weather UI](../app.py) now lets the user enter **Headwind / Tailwind / Crosswind / Variable / Unknown**, but it stores that input as **descriptive context only**. No directional weather coefficient exists and no directional win-probability adjustment is applied.

## Controls and reproducibility

- [Focused read-only CI workflow](../.github/workflows/weather_go_punt_historical.yml) runs unit/negative tests, then the 2014–24 comparison and future-season denial checks, without saving fitted models.
- [Unit tests](../tests/test_weather_pbp_signal.py) cover uncertain precipitation text and no inferred direction; [model tests](../tests/test_weather_go_punt_development.py) check identical-cohort error comparisons, missingness, fixed chronological split, proper metric definitions and refusal of any 2025+ input.
- No production artifact, frozen decision engine, model weights, Streamlit headline recommendation or protected Step 66/83 code was altered by the GO/PUNT study.
- The study is an **observational retrospective prediction exercise**, not an independent before-snap weather study, policy counterfactual evaluation, proof of causal weather effect or justification to change betting/coaching decisions.

### Next requirements before any official weather promotion

A separately preregistered and independently time-stamped **snap-specific** weather dataset is needed, including wind vector measured relative to the play's direction (not just wind speed), trustworthy observed precipitation and intensity, stadium roof context, official play-action labels, return/turnover outcomes and kicker/punter context. Then evaluate the **full fourth-down action-value model**, its calibration, and decision stability with truly out-of-time data before changing production.

**No paid provider, vendor outreach or second PR is needed for this completed historical audit.**
