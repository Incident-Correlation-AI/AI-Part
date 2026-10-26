# Path C — containerized / cloud-parity deployment

This directory is the deployment binding used by the dual-deployment audit. It
is intentionally small and contains **no new detection logic**: the container
runs the same `python -m deploy.service` command as Path L, importing the real
parser, window builder, Isolation Forest detector, evidence correlator and
optional explanation adapter.

> **Audit adapter, not a production architecture.** The repository had no HTTP
> API or Docker files before this audit. `deploy/service.py` is a bounded
> transport around the existing library so both deployment paths can be
> exercised end to end. It is not a claim that this is the production API.

## Contents

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage image, non-root user, `HEALTHCHECK`, entrypoint. |
| `docker-compose.yml` | Detector + optional local mock LLM, read-only artifact mount, healthcheck, restart policy. |
| `.env.example` | Checked-in environment contract (no secrets). |
| `service.py` | HTTP binding around the real ML/correlation path. |
| `mock_llm.py` | Deterministic local OpenAI-compatible provider boundary. |
| `local_telemetry_lab.py` | Controlled local web app + SQLite + collector for Layer B. |
| `README.md` | This document. |

## Build

From the repository root:

```bash
docker build -f deploy/Dockerfile -t lad-anomaly-detector:audit .
```

The build context is the repository root, but `.dockerignore` excludes `.git`,
datasets, model artifacts, reports, ZIPs, prompt archives and `.env` files. The
model is **not** baked into the image.

## Run (Compose)

```bash
cd deploy
cp .env.example .env          # optional; defaults work without it
docker compose up --build -d
docker compose ps
curl -fsS http://127.0.0.1:8080/readyz | python -m json.tool
```

The stack is healthy when `detector` reports `healthy`. `/readyz` returns `503`
until the mounted model artifact loads; it never reports ready without a model.

## Run (plain Docker)

```bash
docker run --rm -p 8080:8080 \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --security-opt no-new-privileges:true --cap-drop ALL \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  lad-anomaly-detector:audit
```

## Required environment

All variables have safe defaults; see `.env.example`. The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `LAD_HOST` / `LAD_PORT` | `0.0.0.0` / `8080` | Bind address / container port. |
| `LAD_DEPLOYMENT_MODE` | `container` in Compose | Reported in every response for parity evidence. |
| `LAD_MODEL_PATH` | `/app/artifacts/network/model/model.joblib` | Read-only model artifact. |
| `LAD_MANIFEST_PATH` | `/app/artifacts/network/model/manifest.json` | Manifest cross-checked by `load_network_detector`. |
| `LAD_NETWORK_CONFIG` | `/app/configs/network.yaml` | Rules / window / model contract. |
| `LAD_CORRELATION_CONFIG` | `/app/configs/correlation.yaml` | Correlation bounds and LLM switches. |
| `LAD_MAX_*` | see `.env.example` | Hard request/buffer/response bounds. |
| `LAD_LLM_PROVIDER` | `mock` | `none`, `mock` or `http`. |
| `LAD_LLM_ENDPOINT` | `http://mock-llm:8081/v1/chat/completions` | Used only when provider is `http`. |
| `LAD_LLM_API_KEY_ENV` | `LAD_LLM_API_KEY` | **Name** of the env var holding the key; the value is never serialized. |

## Artifact mount and identity

`docker-compose.yml` mounts `../artifacts` read-only at `/app/artifacts`.
`GET /readyz` and every inference response include:

* `artifact_sha256` — SHA-256 of the exact `model.joblib` loaded;
* `manifest_sha256` — SHA-256 of the manifest;
* model name, code revision, artifact version, threshold, score direction and
  feature schema version from the loaded contract.

Compare those values with Path L (`GET /v1/model`) to prove the container loaded
the same artifact. A missing or mis-mounted artifact makes `/readyz` return
`503`; inference returns `503 model_unavailable` and **never** a scored result.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness only; does not touch the model. |
| `GET` | `/readyz` | Readiness + artifact hashes. |
| `GET` | `/v1/model` | Full model/feature/contract metadata. |
| `POST` | `/v1/events` | Parse and buffer canonical LAD envelopes. |
| `POST` | `/v1/infer` | Parse + window + score (ML interface only). |
| `POST` | `/v1/analyze` | Score + evidence correlation + optional explanation. |

Bodies accept either a bare JSON array of LAD events or:

```json
{
  "events": [ ... canonical LAD envelopes ... ],
  "use_buffer": true,
  "explain": false
}
```

`use_buffer` defaults to `true`, so events sent earlier to `/v1/events` are
included. Set it to `false` for a one-shot inference over only this request.

Example (after ingesting events):

```bash
curl -fsS -X POST http://127.0.0.1:8080/v1/analyze \
  -H 'Content-Type: application/json' \
  -d '{"use_buffer": true, "explain": false}' | python -m json.tool
```

## Local vs mocked vs real LLM

* `LAD_LLM_PROVIDER=none` — no explanation adapter. Correlation and incident
  drafts still run; the ML decision is unchanged.
* `LAD_LLM_PROVIDER=mock` — in-process deterministic provider. It is **mocked**
  and labelled as such in `explanation.provider` and `correlation.llm`.
* `LAD_LLM_PROVIDER=http` — the real `HttpExplanationProvider` sends only the
  allowlisted `EvidenceBundle.to_prompt_dict()` payload. The endpoint can be:
  * the local `mock-llm` Compose profile (still a mock boundary);
  * an explicitly authorized external provider with `LAD_LLM_API_KEY` exported
    in the shell. No external provider is contacted by default. An external
    run is not claimed unless it was actually performed.

The optional `mock-llm` service is enabled with:

```bash
cd deploy
LAD_LLM_PROVIDER=http LAD_CORRELATION_EXPLAIN=true \
  docker compose --profile mock-llm up --build -d
```

## Healthcheck

The image and Compose use:

```text
GET http://127.0.0.1:8080/readyz  -> 200 only when the model is loaded
```

`start_period` is 25 s to tolerate cold model loading. `/healthz` is suitable
for liveness probes where a model-less process should not be killed.

## Run the regression suite inside the container

The runtime image includes the `dev` extra so the deployment-relevant suite can
run without network access:

```bash
docker compose run --rm --entrypoint python detector -m pytest -q
# or, without Compose:
docker run --rm --entrypoint python \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  lad-anomaly-detector:audit -m pytest -q
```

Container-specific tests live in `tests/test_deploy_service.py` and run against
the same HTTP binding in both deployments.

## Secret scan before build/push

`gitleaks`/`trufflehog` are not required; a documented deterministic scan is
provided:

```bash
python scripts/secret_scan.py --staged
python scripts/secret_scan.py --paths deploy src scripts tests docs .dockerignore
```

The scan fails on credential-shaped strings, private-key blocks, forbidden
paths (`.env`, datasets, model binaries, ZIPs) and high-entropy tokens. It never
prints the matched value.

## Failure modes to verify

| Failure | Expected behaviour |
|---|---|
| Model artifact missing / wrong schema | `/readyz` 503; inference 503 `model_unavailable`; no scored result. |
| Malformed event | HTTP 200 with explicit rejection/quality record; valid events still scored. |
| Empty request with empty buffer | HTTP 400 `no_events`. |
| Oversized body / event count / window count | HTTP 413 with the limit; no partial verdict. |
| LLM provider timeout/error | Correlation succeeds; explanation is ungrounded/fallback; ML decision preserved. |
| Read-only root / dropped capabilities | Service still runs; only `/tmp` is writable. |
| Cold start | `/healthz` up immediately; `/readyz` becomes 200 after model load. |

## Known blockers on the audit host

Docker execution requires access to a daemon. On the audit host the client is
present but `/var/run/docker.sock` is root-only and the user is not in the
`docker` group, and no passwordless `sudo`, Podman, nerdctl or rootless Docker
is available. In that environment `docker build` and `docker compose up` cannot
be executed; the files are still statically validated and the exact blocker is
recorded in the audit report. Remediation: add the user to the `docker` group
(re-login) or provide an accessible daemon / rootless runtime.
