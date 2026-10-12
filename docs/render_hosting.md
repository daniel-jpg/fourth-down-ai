# Fourth-Down AI — Render Free hosting

**Live hostname:** https://fourth-down-ai-nfl.onrender.com/

Render service: [fourth-down-ai-nfl](https://dashboard.render.com/web/srv-db6lj3ij9qps73ctut8g).
It is a separate **Free** Docker deployment from GitHub `main` with
automatic deployments disabled, running exactly the unmodified frozen
decision engine. The repo's [`render.yaml`](../render.yaml) reflects
this name for any future configuration review; a Blueprint edit does not
rename an existing service or automatically trigger a deployment.

## Limits and verification status

The Free tier has **512 MB RAM** and a small CPU allocation. Q4 HOME
and Snow/Rain/wind-direction what-if calculations have passed in a real
browser on this hostname. The complete METAR receipt, international venue
and closed-roof end-to-end smoke is still pending. The weather adjustment
is research-only; the held-out Brier score did not improve, so weather
does not affect the official action recommendation.

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

## Delete duplicate Free services after this new URL is validated

Two older Render Free Fourth-Down AI services were present when the
new exact-name instance was created. To avoid three services sharing
the monthly free-instance allowance, open each old service's
**Settings → Delete Service** after the new browser smoke passes:

- [fourth-down-ai-old](https://dashboard.render.com/web/srv-db6l6f7f3r2c73cqiq10)
  — old URL `fourth-down-ai-verified.onrender.com`.
- [fourth-down-ai](https://dashboard.render.com/web/srv-db6lhvjrjlhs73fb4bk0)
  — old URL `fourth-down-ai-1la8.onrender.com`.

Keep **fourth-down-ai-nfl** (service ID `srv-db6lj3ij9qps73ctut8g`).
Deleting a Render web service does not delete its GitHub repository.
This action is not available through the current Render connector,
so the owner must confirm the deletions in the dashboard.

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
