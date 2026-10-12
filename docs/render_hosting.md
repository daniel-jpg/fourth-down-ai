# Fourth-Down AI Render Free deployment and verification

The Render Free web service is now deployed with the **same frozen-model
container** that passed Docker browser tests. A real public browser
successfully calculated Q4 HOME and the 32°F / 25 mph snow what-if.
The complete weather-interaction suite has not yet passed, so the
[original Streamlit link](https://fourth-down-ai.streamlit.app/) remains
the README's primary demo until the service has its final name and URL.

## Free-only trial: no paid instance authorized

The owner declined paid hosting. The included [Render Blueprint](../render.yaml) now explicitly requests **plan `free`: 0.1 CPU and 512 MB RAM**. This is a free-compute experiment, **not a guaranteed $0 bill or an already validated hosting tier**. Our passing Docker browser test used 1 GiB of memory; the free instance has only half of that allocation and may fail under memory or CPU pressure.

Free Render services sleep after 15 idle minutes and can take about a minute to wake. Each workspace shares **750 free running-instance hours per calendar month**, including the existing `chips-wordguesser-cs169` free service. Exceeding them suspends all free services for the month. **Outbound bandwidth and build minutes can incur overage charges if a payment method is on file**. To avoid surprises, inspect the [Render billing and usage page](https://dashboard.render.com/billing) and spend limits before a prolonged public launch; do not upgrade the compute plan or add a payment method for this experiment. GitHub configuration alone creates no service.

If the free instance fails, stop the experiment and evaluate another genuinely free hosting option. Do **not** silently upgrade to a paid instance.

[Official Render pricing](https://render.com/pricing) · [Compute specifications](https://render.com/docs/compute-plans) · [Blueprint reference](https://render.com/docs/blueprint-spec)

## Create a free test service

Render is connected to ChatGPT already. The assistant can create the new free web service through the Render connector when authorized. Alternatively, open [Deploy this GitHub repository to Render](https://render.com/deploy?repo=https://github.com/daniel-jpg/fourth-down-ai/tree/main) and import the root `render.yaml`.

An existing Free service has now deployed successfully under the original temporary name `fourth-down-ai-verified`. **Rename that existing service rather than creating another one:** in [Render service settings](https://dashboard.render.com/web/srv-db6l6f7f3r2c73cqiq10), find its name in **Settings / General**, select **Edit**, and change it to `fourth-down-ai` (or `fourth-down-ai-nfl` if that name is unavailable). Save the change and inspect the *actual* URL Render assigns. This repository's `render.yaml` now uses the preferred name for future infrastructure-as-code, but editing the YAML alone does not rename the running service. The Render connector currently has no rename action.

The 512 MB trial may fail even though the 1 GiB Docker smoke passed. Only use the actual HTTPS URL provided by Render, not an assumed service name.

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
