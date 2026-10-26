# ML + LLM End-to-End Test Plan — Dual Deployment (Local + Containerized)

Status of this document: **executed plan, written after reconnaissance and before
the final evidence pass.** It records what was verified, what was built, what was
run, and what remains blocked. The companion report
(`docs/ml_llm_e2e_test_report.md`) contains the observed results.

- Audit date: **2026-10-26** (America/Chicago)
- Repository: `Incident-Correlation-AI/AI-Part`
- Governing prompt: `opencode_prompt.md` (included verbatim in
  `docs/opencode_dual_deployment_audit_bundle.md`)
- Local deployment status: **PASS WITH LIMITATIONS**
- Container/cloud deployment status: **BLOCKED** (Docker daemon inaccessible)
- Overall status: **BLOCKED** (one required execution path could not run)

---

## 1. Scope and honesty rules

This plan covers two real execution paths for the same anomaly-detection
library:

| Path | Meaning in this repository |
|---|---|
| **Path L — local binding** | `python -m deploy.service` on the host, loading the local artifact from `artifacts/network/model/`. |
| **Path C — containerized/cloud parity** | The same entrypoint inside the image defined by `deploy/Dockerfile`, orchestrated by `deploy/docker-compose.yml`, with env-var configuration and a read-only artifact mount. |

Hard rules applied throughout:

1. No mock replaces the main ML path. The parser, window builder, Isolation
   Forest, threshold decision, evidence adapter and correlation pipeline are the
   real library implementations in both paths.
2. A mock provider is used **only** at the LLM transport boundary. It is labelled
   `mock` in every response.
3. No claim is made that developer-authored synthetic data is independent or
   real-world validation.
4. Labels, scenario names, test IDs and future events are evaluator-only. They
   never enter features, inference requests or provider prompts.
5. Every result in the report was produced by a command that was actually run on
   this host. Blockers are recorded verbatim.

---

## 2. Verified repository state before the audit

### 2.1 Git

- Working branch at start: `main`
- Remote: `git@github.com:Incident-Correlation-AI/AI-Part.git`
- Prior audit commit: `6d48bfc` (already pushed)
- Pre-existing untracked work: prompt archives (`opencode_prompt.md`,
  `ml_anomaly_detector_milestone_*`), `trained_model_data_from_kaggle/`.
  These are **not** part of the source change and are not staged.

### 2.2 Baseline tests

| Item | Value |
|---|---|
| Command | `python -m pytest -q` |
| Result | **411 passed, 27 warnings** |
| Python | 3.13.14 |
| Warnings | scikit-learn `max_samples` notices in existing fixtures |

The baseline was recorded before adding deployment code. No existing test was
modified to make the baseline green.

### 2.3 Existing deployment surface

Before this audit the repository had **no** `Dockerfile`, Compose file,
`deploy/` directory, HTTP API, or CI deployment file. It is a Python library plus
CLI scripts. The audit therefore had to add a minimal, bounded HTTP binding and a
container area so the prompt's two paths could be exercised at all. This is
documented as an **audit/deployment adapter**, not a claim that it is the
project's production API.

### 2.4 Model artifact and contract

| Field | Value |
|---|---|
| Model file | `artifacts/network/model/model.joblib` |
| Model SHA-256 | `a0185fb925302984127d98d07ebee2473c0c106bfba8ccb9355497d3ac38a570` |
| Manifest | `artifacts/network/model/manifest.json` |
| Manifest SHA-256 | `f88677cd8ca30331edaefaf6eaf9267f1cdf4dda5788fdeb3211315d10e31b8b` |
| Contract version | `1.1.0` |
| Feature schema | `network-features-v1` |
| Model | `sklearn.ensemble.IsolationForest` |
| Threshold | `-0.5235479383749508` |
| Score direction | `lower_is_more_anomalous` |
| Decision rule | `is_anomaly = raw_score <= threshold` |
| Model code revision | `747457f` |

Artifacts are local build outputs and are gitignored. They are mounted read-only
in Path C and are never baked into the image.

---

## 3. Verified architecture and component boundaries

Data flow exercised end to end:

```text
canonical LAD event envelope
  -> deploy.service._sanitize_event        (ingress allowlist, label-key stripping)
  -> anomaly_detector.network.lad_adapter.parse_lad_events
       (schema validation, explicit rejections and quality records)
  -> anomaly_detector.network.features.build_network_windows
       (network-features-v1, epoch-aligned 1-minute windows, causal rolling features)
  -> anomaly_detector.network.inference.NetworkDetector.score_windows
       (Isolation Forest + transparent rule baseline + severity)
  -> deploy.service._build_evidence        (window -> supporting event ids)
  -> anomaly_detector.correlation.EvidenceCorrelationPipeline
       (eligible scored anomalies only, incidents, redaction)
  -> anomaly_detector.network.llm explanation provider (optional)
       (allowlisted EvidenceBundle -> provider -> grounding validation)
```

| Stage | Module | Verified property |
|---|---|---|
| Ingress | `deploy/service.py` | Drops non-allowlisted top-level and attribute keys; counts them without logging values. |
| Parsing | `network/lad_adapter.py` | Invalid individual events become rejections/quality records; the batch does not crash. |
| Features | `network/features.py` | `network-features-v1`; rolling features use only prior same-host/service windows; labels are not inputs. |
| Scoring | `network/inference.py` | Loads artifact + manifest, validates feature order/schema, returns explicit statuses. |
| Contract | `inference/contract.py` | `1.1.0`; statuses `scored`, `insufficient_data`, `invalid_input`, `model_unavailable`, `inference_error`. |
| Evidence | `correlation/evidence.py` | Only eligible scored anomalies with supporting event IDs enter correlation. |
| Correlation | `correlation/pipeline.py` | Explicit counts for candidates, incidents, rejected, duplicates, unscored, orphan/unused evidence. |
| LLM | `network/llm.py` | `HttpExplanationProvider`; grounding validation rejects unknown citations, forbidden tokens, PII and unsupported numbers. |
| Redaction | `correlation` + `deploy/service.py` | Secret/PII diagnostics bounded and redacted. |

### 3.1 HTTP binding added for the audit

`deploy/service.py` adds only transport, bounded buffering, request validation,
redaction and artifact identity. It does not reimplement scoring, thresholding,
routing or correlation.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness; never touches the model. |
| `GET` | `/readyz` | Readiness + artifact/manifest hashes; `503` without a model. |
| `GET` | `/v1/model` | Full model/feature/contract metadata. |
| `POST` | `/v1/events` | Parse and buffer canonical LAD envelopes. |
| `POST` | `/v1/infer` | Parse + window + score (ML interface only). |
| `POST` | `/v1/analyze` | Score + evidence correlation + optional explanation. |

Request bodies accept a bare JSON array or:

```json
{"events": [ ... ], "use_buffer": true, "explain": false}
```

`use_buffer=true` includes events previously posted to `/v1/events`. Every
request is bounded by `LAD_MAX_REQUEST_BYTES`, `LAD_MAX_EVENTS_PER_REQUEST` and
`LAD_MAX_WINDOWS_PER_REQUEST`; exceeding a bound is a hard `413`, never a
partial verdict.

### 3.2 Result schema (actual, observed)

A scored result contains:

```json
{
  "contract_version": "1.1.0",
  "status": "scored",
  "reason_code": "ok",
  "window_id": "host|service|2026-10-26T00:00:00+00:00",
  "window_start": "...",
  "window_end": "...",
  "host_id": "...",
  "service_name": "...",
  "score": {"raw_score": -0.4367, "score_direction": "lower_is_more_anomalous", "is_probability": false},
  "decision": {"is_anomaly": false, "threshold": -0.5235479383749508},
  "model": {"name": "sklearn.ensemble.IsolationForest", "version": "747457f", "artifact_version": "unknown", "feature_schema_version": "network-features-v1", "threshold_config_version": "configs/network.yaml@747457f:-0.523547938375"},
  "data_quality": {"imputed_features": ["..."], "ignored_features": []},
  "extra": {"network": {"severity": "none", "decision_source": "isolation_forest", "baseline": {"is_anomaly": false, "risk_score": 0, "rule_hits": [], "severity": "none"}, "evidence_features": {"...": 0.0}, "telemetry_quality": {"...": 0}}},
  "duration_ms": 71.89
}
```

Non-`scored` results carry the same identity fields but `is_anomaly=null` and an
explicit `reason_code`. They are never counted as normal.

---

## 4. Path L — local startup

Verified command:

```bash
cd "/path/to/ml-anomaly-detector"
python -m deploy.service --host 127.0.0.1 --port 18081 --deployment-mode local
```

Environment overrides (all optional, all in `deploy/.env.example`):
`LAD_MODEL_PATH`, `LAD_MANIFEST_PATH`, `LAD_NETWORK_CONFIG`,
`LAD_CORRELATION_CONFIG`, `LAD_HOST`, `LAD_PORT`, `LAD_DEPLOYMENT_MODE`,
`LAD_MAX_*`, `LAD_LLM_PROVIDER`, `LAD_LLM_ENDPOINT`, `LAD_LLM_MODEL`,
`LAD_LLM_API_KEY_ENV`, `LAD_LLM_TIMEOUT_SECONDS`.

Readiness gate:

```bash
curl -fsS http://127.0.0.1:18081/readyz
```

Observed: HTTP 200, `status=ready`, artifact hash matching the table in §2.4.

---

## 5. Path C — containerized deployment area

Added files and intent:

| File | Purpose |
|---|---|
| `deploy/Dockerfile` | Multi-stage build on `python:3.13-slim-bookworm`; non-root UID/GID 10001; `HEALTHCHECK` on `/readyz`; entrypoint `python -m deploy.service` (same as Path L). |
| `deploy/docker-compose.yml` | Detector service + optional `mock-llm` profile; env-var configuration; read-only artifact mount `../artifacts:/app/artifacts:ro`; healthchecks; restart policy; `read_only`, `cap_drop: ALL`, `no-new-privileges`, PID/memory/CPU bounds. |
| `deploy/.env.example` | Checked-in environment contract with no secret values. |
| `.dockerignore` | Excludes `.git`, `.env`, datasets, model binaries, ZIPs, reports, prompt archives and local run output. |
| `deploy/README.md` | Build/run/healthcheck/artifact/LLM/test/secret-scan/failure-mode documentation. |
| `deploy/mock_llm.py` | Deterministic OpenAI-compatible provider boundary; labelled mock; derived only from the supplied evidence bundle. |

### 5.1 Build and run (as designed)

```bash
docker build -f deploy/Dockerfile -t lad-anomaly-detector:audit .
cd deploy
cp .env.example .env          # optional; defaults work
docker compose up --build -d
curl -fsS http://127.0.0.1:8080/readyz
```

The image contains **no** model artifact. The model is mounted read-only and its
hash is reported by `/readyz` and every inference response. A missing artifact
makes readiness `503` and inference `503 model_unavailable`; there is no silent
fallback.

### 5.2 Environment contract (container-relevant)

| Variable | Default | Effect on verdicts |
|---|---|---|
| `LAD_DEPLOYMENT_MODE` | `container` in Compose | Reported only; does not change scoring. |
| `LAD_MODEL_PATH` / `LAD_MANIFEST_PATH` | `/app/artifacts/...` | Must point at the mounted artifact; mismatch fails closed. |
| `LAD_NETWORK_CONFIG` | `/app/configs/network.yaml` | Rules, rolling windows, minimum coverage, thresholds. |
| `LAD_CORRELATION_CONFIG` | `/app/configs/correlation.yaml` | Correlation bounds and LLM switches. |
| `LAD_MAX_*` | see `.env.example` | Hard request/buffer/response bounds. |
| `LAD_LLM_PROVIDER` | `mock` | `none` / `mock` / `http`. Never changes the ML decision. |
| `LAD_LLM_ENDPOINT` | `http://mock-llm:8081/v1/chat/completions` | Used only for `http`. |
| `LAD_LLM_API_KEY_ENV` | `LAD_LLM_API_KEY` | *Name* of the env var holding the key; the value is never serialized. |

The same env contract was applied to a host process to test configuration parity
(`LAD_DEPLOYMENT_MODE=container`, container paths mapped to host paths). This is
**not** container execution and is reported separately.

### 5.3 Compose stack

- `detector`: healthcheck on `/readyz`, `restart: unless-stopped`, read-only
  root, writable `/tmp` tmpfs, dropped capabilities, PID/memory/CPU limits.
- `mock-llm` (profile `mock-llm`): deterministic provider on port 8081, same
  hardening. Enabled only when testing the `http` provider boundary.

---

## 6. Validation layers

### Layer A — developer-authored synthetic regression

- Source: deterministic simulator in `network/simulator.py`.
- Command: `scripts/stream_network_telemetry.py` streams one whole run
  (`run-02`, 7,216 events) into `/v1/events`, then `/v1/analyze`.
- Ground truth: simulator labels written only to `labels.csv`.
- Purpose: regression, schema, contract and HTTP-path behaviour.
- Independence: **none claimed.** The same project authored the generator, the
  model and this harness.

### Layer B — actual telemetry from controlled local services

- Source: `deploy/local_telemetry_lab.py` starts a real local HTTP application
  and a real SQLite database, drives ordinary and faulty traffic through them,
  and collects the application's own structured telemetry.
- Faults are recorded, not inferred: 5xx burst, SQLite lock/timeout, auth-failure
  burst, benign high traffic, benign latency shift, restart, malformed/duplicate/
  out-of-order telemetry, recovery.
- Ground truth: a separate evaluator schedule written to `labels.csv`; never sent
  to the detector.
- Time: accelerated logical clock (20×), documented in the run manifest.
- Purpose: exercise the real ingestion/inference path with telemetry produced by
  actual local services rather than a pre-written feature table.
- Independence: **controlled local services, developer-authored fault schedule.**
  It is stronger than Layer A but is still not independent real-world data.

### Layer C — independent or pre-existing dataset

- Repository search found only:
  - `tests/fixtures/hdfs_sample/` — a small synthetic Loghub-shaped fixture;
  - `trained_model_data_from_kaggle/run_20261026T191658Z.zip` — a 40 KB archive of
    **this project's own** run artifacts (predictions, metrics, model), not an
    independent dataset.
- No public/independent dataset is present. No download was attempted because the
  prompt requires asking before transmitting/downloading and no credentials or
  license decision were available.
- Status: **NOT EXECUTED.** No Layer C claim is made.

### Layer D — blind held-out evaluation

- `scripts/run_blind_holdout.py` generates a new simulator dataset with
  `seed=20261026` (three runs, 90 minutes each), which was **not** used to fit the
  model or select the threshold.
- The deployed artifact is loaded unchanged. No training, tuning or threshold
  selection occurs in this script.
- Labels stay in memory/evaluator tables; only feature windows are scored.
- Purpose: measure generalisation to a new draw of the generator.
- Limitation: same generator family, so it is **blind to the seed, not
  independent of the simulator**.

### Layer E — unscripted variations and borderline cases

Partially covered by Layer B (rate changes, latency shifts, restarts, mixed
services, missing fields, duplicates, out-of-order events, recovery) and by the
simulator's malformed/duplicate/benign scenarios. Not covered: genuinely
unscripted production traffic, time-of-day effects, real dependency failures,
and human-reviewed ambiguous cases. Status: **PARTIAL**.

---

## 7. Independence and leakage controls

| Control | Implementation | Verification |
|---|---|---|
| Label stripping | `_sanitize_event` allowlists envelope and attribute keys; label/answer keys counted as `evaluation_key` and dropped. | `tests/test_deploy_service.py::test_ingest_infer_and_label_key_stripping`; Layer B leakage attempt. |
| No labels in features | `network-features-v1` column set contains only operational measurements. | Existing leakage tests + `test_prompt_payload_contains_no_evaluation_metadata`. |
| No labels in prompts | `EvidenceBundle.to_prompt_dict()` exposes only window/model/decision/rule/measurement/quality fields. | Captured provider payload assertion. |
| Future events | Windows are half-open and rolling features use only prior windows. | Existing feature tests; `configs/network.yaml` contract. |
| Unscored ≠ normal | Non-`scored` results are excluded from supervised metrics and reported separately. | Evaluator implementation + Layer B three `insufficient_data` windows. |
| Threshold frozen | Layer D loads the existing artifact; no threshold API. | `run_blind_holdout.py` source + artifact hash in output. |
| Secret/PII scan | `scripts/secret_scan.py` deterministic scan. | Run before push; 0 findings on intended files. |

---

## 8. Cross-deployment parity methodology

1. Record the artifact and manifest SHA-256 in both paths.
2. Compare `/v1/model` contract fields (threshold, score direction, feature
   schema, decision rule, model revision).
3. Send the **same** event stream to both endpoints and compare window counts,
   per-window statuses/scores, correlation counts and explanation state.
4. Apply the container env contract to a host process to detect env-var drift.
5. Enumerate deployment-only differences: entrypoint, paths, mounts, healthcheck,
   DNS/egress, resource limits, provider endpoint.
6. Treat any silent mock/replacement as a defect.

The container half of step 3 could not be executed because the Docker daemon is
inaccessible. The env-contract host run was executed and is reported as
**parity-adjacent evidence, not container parity**.

---

## 9. Test matrix and commands

### 9.1 Regression suite

```bash
python -m pytest -q
python -m pytest tests/test_deploy_service.py -q
```

The deployment suite covers: health/readiness/model identity; ingest/infer;
label stripping; empty body; wrong content type; invalid JSON; size limits;
unknown routes; malformed events; duplicates; out-of-order events; unsafe event
IDs; correlation with evidence; real HTTP mock provider; prompt payload purity;
model-unavailable fail-closed; redaction; deployment-mode reporting.

### 9.2 Path L end-to-end

```bash
python -m deploy.service --host 127.0.0.1 --port 18081 --deployment-mode local &
python scripts/stream_network_telemetry.py \
  --url http://127.0.0.1:18081 --run-index 2 --explain \
  --out /tmp/opencode/dual-e2e/layer-a
python scripts/evaluate_http_run.py \
  --labels /tmp/opencode/dual-e2e/layer-a/labels.csv \
  --response /tmp/opencode/dual-e2e/layer-a/detector_response.json \
  --out /tmp/opencode/dual-e2e/layer-a/evaluation.json
```

### 9.3 Layer B

```bash
python deploy/local_telemetry_lab.py \
  --detector-url http://127.0.0.1:18081 \
  --out /tmp/opencode/dual-e2e/layer-b
python scripts/evaluate_http_run.py \
  --labels /tmp/opencode/dual-e2e/layer-b/labels.csv \
  --response /tmp/opencode/dual-e2e/layer-b/detector_response.json \
  --out /tmp/opencode/dual-e2e/layer-b/evaluation.json
```

### 9.4 Layer D

```bash
python scripts/run_blind_holdout.py --seed 20261026 \
  --out /tmp/opencode/dual-e2e/layer-d
```

### 9.5 Path C

```bash
docker build -f deploy/Dockerfile -t lad-anomaly-detector:audit .
cd deploy && docker compose up --build -d
```

Attempted; blocked at the daemon socket. See the report for the exact output.

### 9.6 Env-contract parity (host)

```bash
set -a; . deploy/.env.example; set +a
export LAD_DEPLOYMENT_MODE=container LAD_PORT=18082 \
  LAD_MODEL_PATH="$PWD/artifacts/network/model/model.joblib" \
  LAD_MANIFEST_PATH="$PWD/artifacts/network/model/manifest.json" \
  LAD_NETWORK_CONFIG="$PWD/configs/network.yaml" \
  LAD_CORRELATION_CONFIG="$PWD/configs/correlation.yaml"
python -m deploy.service &
```

Then repeat the Layer A stream against `:18082` and compare.

### 9.7 Secret scan

```bash
python scripts/secret_scan.py --staged
python scripts/secret_scan.py --paths deploy src scripts tests docs .dockerignore
```

---

## 10. Expected outcomes and acceptance criteria

| Check | Expected |
|---|---|
| Path L readiness | 200 only with a loadable artifact; artifact hash reported. |
| Path C readiness | Same, inside the container, with the mounted artifact. |
| Valid stream | Windows scored, statuses explicit, no label values in responses. |
| Malformed stream | 200 with rejection/quality records; valid events still scored. |
| Empty/malformed request | 400 with stable error code. |
| Oversized request | 413 with the limit; no partial verdict. |
| Missing model | 503 `model_unavailable`; no scored result. |
| Correlation | Only scored anomalies become candidates; evidence IDs trace to real events. |
| LLM grounding | Unknown citations, forbidden tokens, PII or unsupported numbers are rejected. |
| Parity | Same artifact hash and same per-window decisions on the same stream. |

---

## 11. Blockers and known limitations

1. **Docker execution blocked.** The client exists, but
   `/var/run/docker.sock` is `root:docker` and the audit user is not in the
   `docker` group; passwordless `sudo` is unavailable; no rootless Docker,
   Podman, nerdctl or buildah is installed. Exact evidence is in the report and
   `/tmp/opencode/dual-e2e/docker_blocker.log`.
2. **No Compose plugin.** Neither `docker compose` nor `docker-compose` is
   installed. Compose files are statically validated as YAML only.
3. **No real LLM provider.** No credentials and no authorized endpoint. The real
   `HttpExplanationProvider` transport is exercised against a local deterministic
   provider; real-provider behaviour is **untested**.
4. **No independent dataset.** Layer C is not executed.
5. **Layer D is seed-blind, not generator-independent.** It uses a new simulator
   seed and the frozen artifact, but the same generator family.
6. **Layer B uses accelerated logical time and a developer-authored fault
   schedule.** High-latency scenarios can reduce the number of completed requests
   per logical minute; this produced three `insufficient_data` windows.
7. **The HTTP binding is new in this audit.** It is a bounded adapter around the
   existing library, not a production-hardened service. It has no authentication,
   persistence, rate limiting or multi-tenant isolation, and the buffer is
   in-memory and bounded.

---

## 12. Deliverables

- `deploy/` — service, Dockerfile, Compose, env template, mock provider, local lab,
  README.
- `scripts/stream_network_telemetry.py` — Layer A HTTP stream.
- `scripts/evaluate_http_run.py` — evaluator-only join and metrics.
- `scripts/run_blind_holdout.py` — Layer D.
- `scripts/secret_scan.py` — deterministic pre-push scan.
- `tests/test_deploy_service.py` — HTTP/contract/safety regressions.
- `docs/ml_llm_e2e_test_plan.md` — this plan.
- `docs/ml_llm_e2e_test_report.md` — observed results and statuses.
- `docs/opencode_dual_deployment_audit_bundle.md` — prompt + plan + report + PR
  summary + Git record.
