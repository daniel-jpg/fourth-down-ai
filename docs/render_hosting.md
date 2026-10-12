# Fourth-Down AI — Render Free hosting

**Live hostname:** https://fourth-down-ai-nfl.onrender.com/

Render service: [fourth-down-ai-nfl](https://dashboard.render.com/web/srv-db6lj3ij9qps73ctut8g).
It is a separate **Free** Docker deployment from GitHub `main` with
automatic deployments disabled, running exactly the unmodified frozen
decision engine. The repo's [`render.yaml`](../render.yaml) reflects
this name for any future configuration review; a Blueprint edit does not
rename an existing service or automatically trigger a deployment.

## Limits and verification status

The Free tier has **512 MB RAM** and a small CPU allocation. The
[complete public Chromium end-to-end smoke #38315221787](https://github.com/daniel-jpg/fourth-down-ai/actions/runs/38315221787)
**PASSED** on this exact hostname: source fingerprint, weather UI,
Q4 HOME calculation (21.71 s), 32°F/25 mph Snow/Rain/wind-direction
comparisons, invalid METAR receipt rejection, international venue
context and a closed-roof rerun. No frozen weights were modified.
The weather sensitivity remains **research-only**: historical held-out
Brier did not improve, so weather does not change the official action
recommendation. A passing browser smoke is **not** a claim of long-term
uptime, model calibration or coaching effectiveness.

The server can sleep after 15 minutes idle. There is a **750-hour shared
monthly free-service allowance across the workspace**. Running several
free instances concurrently will exhaust the allowance faster. If a
payment method is on file, build/bandwidth overages can carry charges;
the user requested no card and no paid compute. Do not change the plan
or add a card for this project.

## Verify a public deployment

1. Open [the exact public URL](https://fourth-down-ai-nfl.onrender.com/)
   and wait for a cold start if necessary.
2. Use [GitHub Actions — Verify public Render Q4 and weather in Chromium](https://github.com/daniel-jpg/fourth-down-ai/actions/workflows/render_public_browser_smoke.yml)
   with the `public_base_url` input `https://fourth-down-ai-nfl.onrender.com`.
3. Confirm source fingerprint, HOME Q4 recommendation, corrected-kickoff
   disclosure, weather-invariance checks, invalid METAR receipt rejection,
   international venue and closed-roof scenarios all pass. Check the
   GitHub Actions screenshots if they do not.
4. Monitor [Render logs](https://dashboard.render.com/web/srv-db6lj3ij9qps73ctut8g/logs),
   memory/CPU metrics, and GitHub status; a 200 health check alone
   cannot prove that a fourth-down calculation works.

## Workspace cleanup verified

On October 12, 2026, a fresh Render workspace service listing returned
**only** `fourth-down-ai-nfl` (service ID
`srv-db6lj3ij9qps73ctut8g`). The two older Fourth-Down AI web
services (`fourth-down-ai-old` and the extra `fourth-down-ai`) are
no longer listed. There is no remaining duplicate-service deletion to
perform. This does **not** delete or change the GitHub repository.

For any future cleanup, delete a Render service only after confirming
its exact service ID and destination. The deployed app currently uses
`https://fourth-down-ai-nfl.onrender.com/`.


## Updating the public app

Keep `autoDeployTrigger: "off"`. After a code change, first pass the
full GitHub regression suite and Docker/browser QA, then manually
deploy the approved commit on the **fourth-down-ai-nfl** Render service.
Rerun the public URL smoke against the deployed code. Do not rerun
protected 2026 Weeks 6–18 or 2027 prospective study gates prematurely.

Legacy [Streamlit Community Cloud mirror](https://fourth-down-ai.streamlit.app/)
is separate and has shown intermittent startup issues.

Provider docs: [Free tiers](https://render.com/docs/free) ·
[Docker](https://render.com/docs/docker) ·
[Health checks](https://render.com/docs/health-checks).
