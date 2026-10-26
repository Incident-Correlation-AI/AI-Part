# syntax=docker/dockerfile:1.7
#
# Path C image for the dual-deployment audit.
#
# The image is deliberately boring: it installs the project into a virtualenv,
# runs the same `python -m deploy.service` entrypoint used by Path L, runs as a
# non-root user and contains no model artifact.  Artifacts are mounted
# read-only at runtime so the exact file hash can be verified from /readyz.
#
# Build from the repository root:
#   docker build -f deploy/Dockerfile -t lad-anomaly-detector:audit .

FROM python:3.13-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what is needed to build the wheel and install runtime + test deps.
COPY pyproject.toml README.md ./
COPY src ./src
COPY deploy ./deploy
COPY tests ./tests
COPY configs ./configs

RUN pip install --upgrade pip setuptools wheel \
 && pip install ".[dev]"

FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PATH="/opt/venv/bin:$PATH" \
    LAD_HOST=0.0.0.0 \
    LAD_PORT=8080 \
    LAD_DEPLOYMENT_MODE=container \
    LAD_MODEL_PATH=/app/artifacts/network/model/model.joblib \
    LAD_MANIFEST_PATH=/app/artifacts/network/model/manifest.json \
    LAD_NETWORK_CONFIG=/app/configs/network.yaml \
    LAD_CORRELATION_CONFIG=/app/configs/correlation.yaml \
    LAD_LOG_LEVEL=INFO

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY deploy ./deploy
COPY tests ./tests

RUN mkdir -p /app/artifacts \
 && chown -R app:app /app

USER app
EXPOSE 8080

# The healthcheck uses the same readiness endpoint that Compose probes.  It
# returns 503 until the mounted model artifact is loaded.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=3).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "deploy.service"]
