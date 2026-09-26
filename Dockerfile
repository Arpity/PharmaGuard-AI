# syntax=docker/dockerfile:1
# PharmaGuard AI - Streamlit application image.
# Multi-stage: dependencies are built in a throw-away stage; the runtime image has no compiler, no pip cache,
# runs as a non-root user and contains NO secrets (configuration is injected at run time via environment variables).

ARG PYTHON_VERSION=3.12

# ---- build stage: create a virtualenv with the runtime dependencies --------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# requirements.lock pins every (transitive) dependency, so the image is reproducible and matches what CI tested.
COPY requirements.txt requirements.lock ./
RUN pip install -c requirements.lock -r requirements.txt

# ---- runtime stage ---------------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG APP_VERSION=dev
LABEL org.opencontainers.image.title="PharmaGuard AI" \
      org.opencontainers.image.description="Pharmaceutical batch-quality analytics and investigation assistant (Streamlit)" \
      org.opencontainers.image.version="${APP_VERSION}"

# Non-root user with a fixed uid so mounted volumes can be chowned predictably.
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --home-dir /home/app --create-home --shell /usr/sbin/nologin app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Application code and the (synthetic) data the app reads. See .dockerignore for what is deliberately left out
# (.env, tests, git history, local databases and logs).
COPY --chown=app:app app/ app/
COPY --chown=app:app config/ config/
COPY --chown=app:app src/ src/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app knowledge/ knowledge/
COPY --chown=app:app .streamlit/ .streamlit/
COPY --chown=app:app data/raw/ data/raw/
COPY --chown=app:app data/processed/ data/processed/
COPY --chown=app:app data/evaluation/ data/evaluation/
COPY --chown=app:app reports/ reports/
COPY --chown=app:app requirements.txt requirements.lock .env.example .gitignore README.md ./

# Writable runtime state (review + trace databases, logs). Mount volumes here to persist it.
RUN mkdir -p data/app logs && chown -R app:app data/app logs
VOLUME ["/app/data/app", "/app/logs"]

USER app
EXPOSE 8501

# Streamlit's built-in liveness endpoint (no curl in the slim image, so use Python).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('STREAMLIT_SERVER_PORT','8501'), timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py"]
