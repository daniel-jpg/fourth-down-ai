# Publish the verified Fourth-Down AI Docker app on Render

These steps deploy the **same frozen-model container** that passed the local and Docker-hosted Q4 HOME and weather browser tests. The original [Streamlit Community Cloud app](https://fourth-down-ai.streamlit.app/) is unchanged. This page does not claim a second public URL already exists.

## Before creating a billable service

Render's currently published compute plans include 512 MB free, 512 MB Starter, and 2 GB Standard. We verified the app inside a **1 GiB** limited container, not 512 MB. The included [Render Blueprint](../render.yaml) therefore requests one `1c-2g` (1 CPU / 2 GB) instance in Oregon, which Render lists at **$25/month** on a $0 Hobby workspace. Review billing and network charges before authorizing this ongoing expense. Committing deployment files alone creates no Render service.

[Official Render pricing](https://render.com/pricing) · [Compute specifications](https://render.com/docs/compute-plans) · [Blueprint reference](https://render.com/docs/blueprint-spec)

## Create the service (account owner action)

1. Connect Render through the ChatGPT Render app suggestion, or sign in at the [Render Dashboard](https://dashboard.render.com/).
2. Open [Deploy this GitHub repository to Render](https://render.com/deploy?repo=https://github.com/daniel-jpg/fourth-down-ai/tree/main). Authorize the repository connection as needed and import the root `render.yaml` Blueprint.
3. Confirm **repository** `daniel-jpg/fourth-down-ai`, **branch** `main`, **service** `fourth-down-ai-verified`, **runtime** Docker, **plan** `1c-2g` ($25/month), **region** Oregon, and **health path** `/_stcore/health`.
4. Confirm that **automatic deploys are disabled** and there is no database or disk. Only proceed with service creation after approving any displayed billable charges.
5. Allow Render to build the existing repository `Dockerfile`. Copy the **actual** assigned HTTPS `*.onrender.com` URL. Do not assume it equals the requested service name.

Render's HTTP health checks use `/_stcore/health`. The Docker startup command listens on the host-provided `PORT`; it does not require secrets or a database.

## Verify the real public URL before announcing it

1. Confirm that the exact Render URL plus `/_stcore/health` responds with HTTP 200. That check only establishes server availability, not correct predictions.
2. Open GitHub [Actions: Verify public Render Q4 and weather in Chromium](https://github.com/daniel-jpg/fourth-down-ai/actions/workflows/render_public_browser_smoke.yml).
3. Select **Run workflow** on `main`, paste the actual assigned `https://YOUR-SERVICE.onrender.com` URL as `public_base_url`, and start the test.
4. Require a green workflow with all three checks: the **hidden source fingerprint equals tested GitHub main**, the Q4 HOME browser simulation produces Recommendation / WP / corrected-kickoff caption, and the weather browser tests include research isolation, invalid METAR rejection, venue context, and covered-roof behavior.
5. Inspect the retained GitHub Actions browser screenshots and Render logs if any stage fails. A green health check alone is not a passed app test.

The smoke workflow rejects localhost, HTTP, non-Render domains, and URLs with extra paths. It does not call live NOAA, modify models or open the 2026/2027 protected evaluation gates.

## Updating the service safely

The Blueprint deliberately sets `autoDeployTrigger` to `off`. After new GitHub model/UI changes, first require the repository's full tests to pass, then **manually deploy the approved commit** through Render and repeat the public-browser check against the exact revision. If Render serves a stale commit, the fingerprint test fails on purpose.

Leave the current Streamlit demo link in place until the new public app proves stable and the user explicitly chooses to advertise or switch to it. All weather what-if probabilities remain experimental research rather than validated coaching advice.

Render background documentation: [Docker builds](https://render.com/docs/docker) · [HTTP health checks](https://render.com/docs/health-checks) · [Manual versus automatic deploys](https://render.com/docs/deploys).
