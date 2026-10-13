# Optional container hosting; the Community Cloud app is unchanged.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    HOME=/home/appuser

WORKDIR /app

# Shared OpenMP library for shipped XGBoost/scikit-learn runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

# Only app/runtime paths. The .dockerignore explicitly blocks all
# prospective/research/evaluation inputs and non-production models.
COPY --chown=appuser:appuser app.py ./
# Include ONLY the public Streamlit display configuration (no secrets).
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser assets/ ./assets/

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8501') + '/_stcore/health', timeout=3).read()"

# Shell exec forwards TERM to Streamlit; PORT is set above and overridable.
CMD ["sh", "-c", "exec python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false"]
