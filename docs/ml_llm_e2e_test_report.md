# ML + LLM End-to-End Test Report — Dual Deployment (Local + Containerized)

- Audit date: **2026-10-26** (America/Chicago)
- Repository: `Incident-Correlation-AI/AI-Part`
- Governing prompt: `opencode_prompt.md`
- Local deployment (Path L): **PASS WITH LIMITATIONS**
- Containerized/cloud deployment (Path C): **BLOCKED**
- Cross-deployment parity: **BLOCKED** (container half not executable); environment-contract host parity **PASSED**
- **Overall status: BLOCKED** — one of the two required execution paths could not be run.

This report contains only results that were actually observed on the audit host. Where a
layer or deployment could not run, the exact blocker and its evidence are recorded.

---

## 1. Summary

| Area | Result |
|---|---|
| Baseline regression suite (before changes) | **411 passed**, 27 warnings |
| Final regression suite (after changes) | **429 passed**, 27 warnings |
| New deployment tests | 18, all passed |
| Path L — local HTTP binding, real artifact, real parser/window/scorer/correlation | **Ran end to end** |
| Path L — Layer A (developer synthetic) | **Ran**: 269 scored windows, F1 0.750 |
| Path L — Layer B (controlled local services) | **Ran**: 78 windows, F1 0.833 |
| Layer C (independent/pre-existing dataset) | **Not executed** — no dataset present, none authorized for download |
| Layer D (blind simulator seed) | **Ran**: 807 scored windows, F1 0.749, ROC-AUC 0.957 |
| Layer E (unscripted/borderline) | **Partial** — only the variations listed in §6.5 |
| Path C — Docker build / Compose up | **BLOCKED** — Docker daemon socket permission denied |
| Container parity (same stream, both paths) | **BLOCKED** — container half absent |
| Environment-contract parity (host process, container env) | **Passed**: zero per-window decision differences |
| Real external LLM provider | **Untested** — no credentials, no authorized endpoint |
| Secret scan before push | **Passed** (0 findings on intended files) |

Two real defects were found and fixed in the audit-added service, plus one defect in the
audit's own secret scanner. They are documented in §9. The core ML model, threshold and
feature schema were **not** changed.

---

## 2. Per-deployment status

### 2.1 Path L — local binding

**Status: PASS WITH LIMITATIONS.**

The real library was exercised through a bounded HTTP adapter:
`python -m deploy.service --host 127.0.0.1 --port 18081 --deployment-mode local`.
The loaded artifact, parser, window builder, Isolation Forest, threshold decision, evidence
adapter and correlation pipeline are the project's own implementations. No scoring or
correlation component was mocked.

Limitations: the HTTP binding is new in this audit; the LLM provider was a local
deterministic mock (labelled `mock`); Layer C was not executed; the local-service lab uses
accelerated logical time.

### 2.2 Path C — containerized/cloud parity

**Status: BLOCKED.**

A complete deployment area was created and statically validated, but it could not be built
or started. The audit user is not in the `docker` group, `/var/run/docker.sock` is
`root:docker` mode `0660`, passwordless `sudo` is unavailable, and no rootless runtime is
installed. Exact output is in §7.1.

Consequences: the acceptance criteria "`docker build` succeeds", "`docker compose up`
brings the stack to healthy", "inference endpoint answers inside the container" and
"artifact loaded in the container is identifiable" are **not satisfied**. No claim of
container execution is made anywhere in this report.

### 2.3 Cross-deployment parity

**Status: BLOCKED** (container half). The environment contract was applied to a host
process (`LAD_DEPLOYMENT_MODE=container`, container-style env vars) and the same event
stream was replayed. That produced **identical** decisions, but it is parity-adjacent
evidence, **not** container parity. See §8.

---

## 3. Git state and files changed

### 3.1 Starting state

- Branch at audit start: `main`
- Remote: `git@github.com:Incident-Correlation-AI/AI-Part.git`
- Previous pushed commit: `6d48bfc` (`Milestone 10: evidence correlation and canonical incident drafts`)
- Pre-existing untracked work: prompt archives, `trained_model_data_from_kaggle/`. These were
  **not** staged.

### 3.2 Files added or changed by this audit

| Path | Change | Purpose |
|---|---|---|
| `deploy/__init__.py` | added | Package marker. |
| `deploy/service.py` | added | Bounded HTTP binding over the existing library; buffer, validation, redaction, artifact identity. |
| `deploy/mock_llm.py` | added | Deterministic OpenAI-compatible provider used only at the provider boundary; always reported as `mock`. |
| `deploy/local_telemetry_lab.py` | added | Real local web app + SQLite + telemetry collector + separate evaluator labels (Layer B). |
| `deploy/Dockerfile` | added | Multi-stage build, pinned base, non-root UID 10001, healthcheck, same entrypoint as Path L. |
| `deploy/docker-compose.yml` | added | Detector + optional mock provider, env contract, read-only artifact mount, limits, healthchecks. |
| `deploy/.env.example` | added | Checked-in environment contract; placeholder values only. |
| `deploy/README.md` | added | Build/run/health/artifact/provider/test/secret-scan instructions. |
| `.dockerignore` | added | Excludes secrets, VCS data, datasets, artifacts, archives, caches. |
| `scripts/stream_network_telemetry.py` | added | Layer A HTTP stream against the real endpoint. |
| `scripts/evaluate_http_run.py` | added | Evaluator-only label join; keeps labels out of the service. |
| `scripts/run_blind_holdout.py` | added | Layer D: frozen artifact scored on an unseen seed. |
| `scripts/secret_scan.py` | added | Deterministic pre-push secret/forbidden-path scan with self-test. |
| `tests/test_deploy_service.py` | added | 18 HTTP contract, safety, fail-closed and prompt-purity tests. |
| `docs/ml_llm_e2e_test_plan.md` | added | Executed test plan. |
| `docs/ml_llm_e2e_test_report.md` | added | This report. |
| `docs/opencode_dual_deployment_audit_bundle.md` | added | Prompt + plan + report + PR summary + Git record. |

**No existing source file was modified.** The model artifact, manifest, feature schema,
threshold and contract version are unchanged.

### 3.3 Git record

| Item | Value |
|---|---|
| Feature branch | `audit/dual-deployment-e2e-2026-10-26` |
| Base branch | `main` at `6d48bfc` |
| Audit commit | `0b0bc82f3b76c0c987ff1496893f13de0770a84c` (`0b0bc82`) |
| Remote | `origin` → `git@github.com:Incident-Correlation-AI/AI-Part.git` |
| Push result | **Success** — new branch pushed to `origin/audit/dual-deployment-e2e-2026-10-26` (exit 0) |
| PR target | `https://github.com/Incident-Correlation-AI/AI-Part/pull/new/audit/dual-deployment-e2e-2026-10-26` |
| Merged to `main` | **No** (intentionally not merged) |
| PR | Not opened automatically; PR-style summary in §12 |

A follow-up documentation commit records this Git table at the branch tip. The audit commit
is the commit that contains the source, tests, deployment files and the first version of the
three documents.

---

## 4. Commands actually executed

```bash
# Baseline (before changes)
python -m pytest -q
# -> 411 passed, 27 warnings

# Deployment tests
python -m pytest tests/test_deploy_service.py -q
# -> 18 passed

# Final full suite
python -m pytest -q
# -> 429 passed, 27 warnings in 61.29s

# Path L
python -m deploy.service --host 127.0.0.1 --port 18081 --deployment-mode local

# Layer A
python scripts/stream_network_telemetry.py \
  --url http://127.0.0.1:18081 --run-index 2 --explain \
  --out /tmp/opencode/dual-e2e/layer-a
python scripts/evaluate_http_run.py \
  --labels /tmp/opencode/dual-e2e/layer-a/labels.csv \
  --response /tmp/opencode/dual-e2e/layer-a/detector_response.json \
  --out /tmp/opencode/dual-e2e/layer-a/evaluation.json

# Layer B
python deploy/local_telemetry_lab.py \
  --detector-url http://127.0.0.1:18081 \
  --out /tmp/opencode/dual-e2e/layer-b
python scripts/evaluate_http_run.py \
  --labels /tmp/opencode/dual-e2e/layer-b/labels.csv \
  --response /tmp/opencode/dual-e2e/layer-b/detector_response.json \
  --out /tmp/opencode/dual-e2e/layer-b/evaluation.json

# Layer D
python scripts/run_blind_holdout.py --seed 20261026 \
  --model artifacts/network/model/model.joblib \
  --manifest artifacts/network/model/manifest.json \
  --out /tmp/opencode/dual-e2e/layer-d

# Path C (attempted)
docker build -f deploy/Dockerfile -t lad-anomaly-detector:audit .
cd deploy && docker compose up --build -d
# -> blocked at the daemon socket

# Environment-contract parity (host process, container env)
set -a; . deploy/.env.example; set +a
export LAD_PORT=18082 LAD_DEPLOYMENT_MODE=container \
  LAD_MODEL_PATH="$PWD/artifacts/network/model/model.joblib" \
  LAD_MANIFEST_PATH="$PWD/artifacts/network/model/manifest.json" \
  LAD_NETWORK_CONFIG="$PWD/configs/network.yaml" \
  LAD_CORRELATION_CONFIG="$PWD/configs/correlation.yaml"
python -m deploy.service
python scripts/stream_network_telemetry.py --url http://127.0.0.1:18082 ...

# Pre-push secret scan
python scripts/secret_scan.py --paths deploy src scripts tests docs .dockerignore pyproject.toml README.md
python scripts/secret_scan.py --staged
```

Raw outputs retained under `/tmp/opencode/dual-e2e/`: `baseline_tests.log`,
`final_tests.log`, `docker_blocker.log`, `layer-a/`, `layer-b/`, `layer-c-envparity/`,
`layer-d/`, and service logs.

---

## 5. Artifact identity and ML contract

Both the local run and the environment-contract run reported the same artifact:

| Field | Value |
|---|---|
| Model file | `artifacts/network/model/model.joblib` |
| Model SHA-256 | `a0185fb925302984127d98d07ebee2473c0c106bfba8ccb9355497d3ac38a570` |
| Manifest | `artifacts/network/model/manifest.json` |
| Manifest SHA-256 | `f88677cd8ca30331edaefaf6eaf9267f1cdf4dda5788fdeb3211315d10e31b8b` |
| Model type | `sklearn.ensemble.IsolationForest` |
| Code revision | `747457f` |
| Contract version | `1.1.0` |
| Feature schema | `network-features-v1` |
| Threshold | `-0.5235479383749508` |
| Score direction | `lower_is_more_anomalous` |
| Decision rule | `is_anomaly = raw_score <= threshold` |
| Statuses | `scored`, `insufficient_data`, `invalid_input`, `model_unavailable`, `inference_error` |

The artifact is gitignored and is **not** baked into the Compose image; it is mounted
read-only. The container path is designed to report `503` if the mount is missing or the
hash cannot be loaded. This design was statically validated but not executed.

Observation carried over from the existing project: `artifact_version` is `unknown` in the
manifest. Identity is therefore anchored on the file SHA-256 plus code revision, both of
which are reported per request.

---

## 6. Evaluation results by layer

None of the layers below are combined into a single score. Each has different provenance
and different independence claims.

### 6.1 Layer A — developer-authored synthetic regression (Path L)

Data source: `network/simulator.py`, `run-02`, streamed event-by-event through
`POST /v1/events` and then `POST /v1/analyze` with `explain=true`.

| Item | Value |
|---|---|
| Events sent | 7,216 |
| Accepted | 7,170 |
| Duplicate event IDs | 7 |
| Rejected | 39 |
| Label windows | 270 |
| Scored windows | 269 |
| Label windows not materialized | 1 (`sim-db-01|orders-db|2026-01-03T00:33:00+00:00`) |
| Wall duration | 8.11 s (logical run 90 min) |

| Metric | Value |
|---|---|
| TP / FP / FN / TN | 30 / 10 / 10 / 219 |
| Precision | 0.7500 |
| Recall | 0.7500 |
| F1 | 0.7500 |
| False-positive rate | 0.0437 |

Per scenario:

| Scenario | n | TP | FP | FN | TN | Precision | Recall | F1 | FPR | Delay |
|---|---|---|---|---|---|---|---|---|---|---|
| `http_5xx_burst` | 30 | 16 | 0 | 4 | 10 | 1.000 | 0.800 | 0.889 | 0.000 | 0 min |
| `db_timeout_latency` | 30 | 10 | 0 | 0 | 20 | 1.000 | 1.000 | 1.000 | 0.000 | 0 min |
| `auth_failure_burst` | 30 | 4 | 1 | 6 | 19 | 0.800 | 0.400 | 0.533 | 0.050 | 1 min |
| `benign_high_traffic` | 30 | 0 | 5 | 0 | 25 | — | — | — | 0.167 | — |
| `benign_latency_shift` | 30 | 0 | 0 | 0 | 30 | — | — | — | 0.000 | — |
| `benign_restart` | 15 | 0 | 1 | 0 | 14 | — | — | — | 0.067 | — |
| `malformed_telemetry` | 29 | 0 | 3 | 0 | 26 | — | — | — | 0.103 | — |
| `normal` | 45 | 0 | 0 | 0 | 45 | — | — | — | 0.000 | — |
| `normal_recovery` | 30 | 0 | 0 | 0 | 30 | — | — | — | 0.000 | — |

By service:

| Service | n | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| `web-api` | 90 | 10 | 4 | 0 | 76 | 0.714 | 1.000 | 0.833 |
| `orders-db` | 89 | 10 | 4 | 0 | 75 | 0.714 | 1.000 | 0.833 |
| `auth-service` | 90 | 10 | 2 | 10 | 68 | 0.833 | 0.500 | 0.625 |

Correlation and explanation:

| Item | Value |
|---|---|
| Candidates | 40 |
| Incidents | 11 |
| LLM interpretations | 40 |
| Grounded explanations | 40 / 40 |
| Provider fallbacks | 0 |
| Correlation errors / rejected / duplicates / orphan evidence | 0 / 0 / 0 / 0 |
| Scored but not flagged | 229 |
| Telemetry-issue windows | 29 scored, 3 flagged |

Interpretation: the pipeline is functional end to end, but the synthetic auth-failure
scenario is weak (recall 0.40). The three flags on `malformed_telemetry` occur in windows
that the evaluator separately labels `telemetry_issue=true`; they are not counted as benign
successes. Independence: **none claimed** — same project authored the generator, model and
harness.

### 6.2 Layer B — telemetry from controlled local services (Path L)

Data source: `deploy/local_telemetry_lab.py`. A real local HTTP application and a real
SQLite database were exercised with ordinary and faulty traffic; the application's own
structured telemetry was shipped through `POST /v1/events` and scored by the real endpoint.
Faults were recorded, not inferred. Logical time was accelerated 20×; wall duration 79.4 s
for 26 logical minutes.

| Item | Value |
|---|---|
| Events | 1,266 |
| Label windows | 78 |
| Scored | 75 |
| `insufficient_data` | 3 (`insufficient_coverage`) |
| Missing labeled windows | 0 |

| Metric | Value |
|---|---|
| TP / FP / FN / TN | 5 / 0 / 2 / 68 |
| Precision | 1.0000 |
| Recall | 0.7143 |
| F1 | 0.8333 |
| False-positive rate | 0.0000 |

Per scenario:

| Scenario | n | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| `http_5xx_burst` | 6 | 2 | 0 | 0 | 4 | 1.000 | 1.000 | 1.000 |
| `db_timeout_latency` | 3 | 3 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| `auth_failure_burst` | 6 | 0 | 0 | 2 | 4 | — | 0.000 | — |
| `benign_high_traffic` | 6 | 0 | 0 | 0 | 6 | — | — | — |
| `benign_latency_shift` | 6 | 0 | 0 | 0 | 6 | — | — | — |
| `benign_restart` | 3 | 0 | 0 | 0 | 3 | — | — | — |
| `malformed_telemetry` | 3 | 0 | 0 | 0 | 3 | — | — | — |
| `normal` | 24 | 0 | 0 | 0 | 24 | — | — | — |
| `normal_recovery` | 18 | 0 | 0 | 0 | 18 | — | — | — |

By service:

| Service | n | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| `checkout-api` | 26 | 4 | 0 | 0 | 22 | 1.000 | 1.000 | 1.000 |
| `orders-db` | 25 | 1 | 0 | 0 | 24 | 1.000 | 1.000 | 1.000 |
| `auth-service` | 24 | 0 | 0 | 2 | 22 | — | 0.000 | — |

Correlation: 5 candidates, 3 incidents, 0 errors, no explanation requested.

**Honest caveats.**

1. The auth-failure burst was genuinely missed (2 false negatives). No threshold was tuned
   to rescue it.
2. Three windows were `insufficient_data` because the accelerated logical clock plus
   high-latency faults reduced completed requests below the coverage minimum:
   `auth-service` 08:14 and 08:15 (both benign), and `orders-db` 08:15 (**true anomaly**).
   That last window is an **unscored incident window**; it is excluded from the confusion
   matrix, not counted as a miss or a normal. The `db_timeout_latency` per-scenario table
   therefore has `n=3` positives and no negatives, so its FPR is "not applicable" rather
   than a real 0.
3. Independence: controlled local services, but the fault schedule is developer-authored.
   Stronger than Layer A, still not independent real-world data.

### 6.3 Layer C — independent / pre-existing dataset

**Status: NOT EXECUTED.**

Repository search found only:

- `tests/fixtures/hdfs_sample/` — a small synthetic Loghub-shaped fixture used by existing
  tests, not an independent evaluation corpus.
- `trained_model_data_from_kaggle/run_20261026T191658Z.zip` — a 40 KB archive of **this
  project's own** predictions/metrics/model, not an external dataset.

No public or independently authored dataset is present. No download was attempted because
the prompt requires asking before downloading/transmitting and no license/credential
decision was available in this run. **No Layer C claim is made.**

### 6.4 Layer D — blind held-out evaluation

Data source: a fresh simulator draw with `seed=20261026`, three 90-minute runs. This seed
was not used for model fitting or threshold selection. The deployed artifact was loaded
unchanged; the script performs no training, tuning or threshold selection. Labels stayed in
evaluator tables and were never passed to the detector.

| Item | Value |
|---|---|
| Seed | `20261026` |
| Runs | 3 (`run-00`, `run-01`, `run-02`) |
| Events | 21,669 (21,410 accepted, 120 duplicates, 259 rejected) |
| Windows / labels | 810 / 810 |
| Scored | 807 |
| `insufficient_data` | 3 (`no_events`) |
| Scored fraction | 0.9963 |

| Metric | Value |
|---|---|
| TP / FP / FN / TN | 97 / 42 / 23 / 645 |
| Precision | 0.6978 |
| Recall | 0.8083 |
| F1 | 0.7490 |
| False-positive rate | 0.0611 |
| ROC-AUC | 0.9572 |
| PR-AUC | 0.7800 |
| Prevalence | 0.1487 |

Incident-only view (only windows inside an incident scenario):

| Metric | Value |
|---|---|
| TP / FP / FN / TN | 97 / 4 / 23 / 146 |
| Precision | 0.9604 |
| Recall | 0.8083 |
| F1 | 0.8778 |

Transparent rule baseline on the same blind data (for context, not a substitute):

| Metric | Model | Baseline |
|---|---|---|
| Precision | 0.6978 | 0.6923 |
| Recall | 0.8083 | 0.7500 |
| F1 | 0.7490 | 0.7200 |
| FPR | 0.0611 | 0.0582 |
| ROC-AUC | 0.9572 | 0.8373 |
| PR-AUC | 0.7800 | 0.6121 |

Window-level agreement: both flagged 98, model-only 41, baseline-only 32, neither 636.

Per scenario:

| Scenario | n | TP | FP | FN | TN | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|
| `http_5xx_burst` | 90 | 52 | 2 | 8 | 28 | 0.963 | 0.867 | 0.912 | 0.067 |
| `db_timeout_latency` | 90 | 30 | 0 | 0 | 60 | 1.000 | 1.000 | 1.000 | 0.000 |
| `auth_failure_burst` | 90 | 15 | 2 | 15 | 58 | 0.882 | 0.500 | 0.638 | 0.033 |
| `benign_high_traffic` | 90 | 0 | 14 | 0 | 76 | — | — | — | 0.156 |
| `benign_latency_shift` | 90 | 0 | 5 | 0 | 85 | — | — | — | 0.056 |
| `benign_restart` | 45 | 0 | 7 | 0 | 38 | — | — | — | 0.156 |
| `malformed_telemetry` | 87 | 0 | 2 | 0 | 85 | — | — | — | 0.023 |
| `normal` | 135 | 0 | 6 | 0 | 129 | — | — | — | 0.044 |
| `normal_recovery` | 90 | 0 | 4 | 0 | 86 | — | — | — | 0.044 |

Detection delay: **9 / 9** scenario groups detected; mean 0.222 min, median 0 min, max
1.0 min. The only 1-minute delays are the two auth-failure bursts in `run-00` and `run-01`.

**Important negative finding:** on `auth_failure_burst`, the baseline rule beats the
Isolation Forest (baseline recall 1.000 / F1 0.923 vs model 0.500 / 0.638). This is a real
model weakness on this scenario, not a harness artefact, and it is consistent with the
Layer A and Layer B auth-failure misses.

Limitation: Layer D is **blind to the seed, not independent of the generator**. The same
project authored the simulator family. It measures generalisation to a new random draw, not
to production traffic.

### 6.5 Layer E — unscripted variations and borderline cases

**Status: PARTIAL.**

Covered by Layer B and Layer D: rate changes, benign high volume, benign latency shift,
restarts, recovery, three services, missing/malformed fields, duplicate events,
out-of-order events, unsupported schema versions, naive timestamps, invalid status codes,
invalid latency, short and sustained incidents.

Not covered: genuinely unscripted production traffic, time-of-day/day-of-week effects, real
dependency failures, multi-tenant traffic, human-reviewed ambiguous cases, and adversarial
telemetry. Those remain **inconclusive / untested**.

---

## 7. Path C evidence — deployment artifacts and blocker

### 7.1 Exact blocker

Captured verbatim in `/tmp/opencode/dual-e2e/docker_blocker.log`:

```text
== docker version ==
Client:
 Version:           28.5.2+dfsg4
 API version:       1.51
permission denied while trying to connect to the Docker daemon socket at
unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.51/version":
dial unix /var/run/docker.sock: connect: permission denied

== docker compose version ==
docker: unknown command: docker compose

== docker-compose ==
docker-compose: not installed

== podman/nerdctl/buildah ==
podman: not installed
nerdctl: not installed
buildah: not installed

== socket perms ==
srw-rw---- 1 root docker 0 Oct 26 12:11 /var/run/docker.sock

== groups ==
uid=1000(barok) gid=1000(barok) groups=1000(barok),...,27(sudo),...

== passwordless sudo ==
sudo: a password is required

== docker build ==
ERROR: permission denied while trying to connect to the Docker daemon socket at
unix:///var/run/docker.sock: Head "http://%2Fvar%2Frun%2Fdocker.sock/_ping":
dial unix /var/run/docker.sock: connect: permission denied
```

The Docker **client** exists; the **daemon** is unreachable to this user. `sudo` requires a
password, which the audit is not authorized to supply, and no rootless alternative is
installed.

### 7.2 What was validated statically instead

| Check | Result |
|---|---|
| Compose YAML parses | Pass — services `detector`, `mock-llm` |
| Detector image reference | `lad-anomaly-detector:${LAD_IMAGE_TAG:-audit}` |
| `mock-llm` profile wiring | Pass — profile `mock-llm` |
| Detector healthcheck present | Pass |
| `read_only: true` | Pass |
| `cap_drop: [ALL]` | Pass |
| Artifact mount | `../artifacts:/app/artifacts:ro` |
| Dockerfile stages | `builder` → `runtime` on `python:3.13-slim-bookworm` |
| Non-root runtime user | `USER app` (UID/GID 10001) |
| Entrypoint matches Path L | `python -m deploy.service` |
| `.dockerignore` excludes secrets/artifacts/data | Pass |
| Secret scan over deploy files | Pass (0 findings) |

Static checks do **not** establish that the image builds, that the stack becomes healthy, or
that the endpoint answers inside the container. Those remain **unverified**.

### 7.3 Deployment-only differences (designed and reviewed)

| Difference | Risk | Handling |
|---|---|---|
| Entrypoint | Low — same module as Path L | Compose command matches `python -m deploy.service`. |
| Config paths | Medium — wrong path silently drops config | Defaults are absolute `/app/...`; mismatch fails readiness. |
| Artifact mount | High — missing mount must not fall back | Readiness `503`, inference `503 model_unavailable`; hash reported. |
| `LAD_DEPLOYMENT_MODE` | Low — reporting only | Cannot change scoring. |
| Provider endpoint | Medium — DNS/egress may differ | Only affects explanation; ML verdict independent. |
| Resource limits | Medium — may cause restarts under load | Limits set conservatively; not load-tested. |
| Non-root user / read-only FS | Medium — write attempts fail | Logs to stdout; `/tmp` tmpfs writable. |

### 7.4 Cloud-only failure modes

| Failure mode | Status |
|---|---|
| Cold start / model load time | Not executed; load time on host was ~57 ms. |
| Missing or mis-mounted artifact | Covered by a host test (`503 model_unavailable`); not re-run in a container. |
| Env-var drift | Partially covered by the environment-contract run (§8). |
| Egress failure to LLM provider | Provider error handling unit-tested; real egress untested. |
| Container restart/recovery | Untested. |
| Resource exhaustion | Untested. |

---

## 8. Cross-deployment parity findings

Container execution is blocked, so the required parity check is **BLOCKED**. As a partial
control, the container environment contract was applied to a host process on port 18082
(`LAD_DEPLOYMENT_MODE=container`, env-driven paths) and the identical Layer A stream was
replayed.

| Comparison | Result |
|---|---|
| Model SHA-256 | Equal (`a0185f…a570`) |
| Manifest SHA-256 | Equal (`f88677…1b8b`) |
| Contract version | Equal (`1.1.0`) |
| Feature schema | Equal (`network-features-v1`) |
| Threshold | Equal (`-0.5235479383749508`) |
| Score direction | Equal (`lower_is_more_anomalous`) |
| Model revision | Equal (`747457f`) |
| Rows compared | 269 |
| Per-row `status` differences | **0** |
| Per-row `raw_score` differences | **0** |
| Per-row `is_anomaly` differences | **0** |
| Per-row `severity` differences | **0** |
| Overall metrics | Identical (TP 30 / FP 10 / FN 10 / TN 219; F1 0.750) |
| Per-scenario metrics | Identical |
| Per-service metrics | Identical |
| Correlation counts | Identical (40 candidates, 11 incidents, 0 errors) |
| Explanations | Identical (40 attempted, 40 grounded, 0 fallback) |
| Missing labeled windows | Identical (1) |

Interpretation: the env contract and configuration paths do not alter verdicts on this
stream. This is **not** evidence that the container behaves identically — it is evidence
that the *host process with container-style configuration* does. The container-specific
risks in §7.3 remain open.

---

## 9. Defects found and fixed

| # | Defect | Observed failure | Root cause | Fix | Regression test |
|---|---|---|---|---|---|
| 1 | Error `detail` serialized as a string | A 400 response returned `detail` as a JSON-encoded string instead of an object, so callers could not read `detail.request_parse`. | `detail` was passed through the generic JSON fallback, which stringifies non-primitive objects. | Serialize `detail` with `dict(detail or {})` directly. | `test_unsafe_event_id_is_rejected_at_ingress` asserts `detail.request_parse.rejected`. |
| 2 | Buffer eviction removed the wrong quality record | Under buffer pressure, evicting the oldest event removed an arbitrary quality entry, so duplicate/invalid counts could drift from the retained events. | Quality records were removed by insertion order rather than by the evicted event id. | Evict the quality record matching the oldest evicted event id. | Buffer/duplicate tests in `tests/test_deploy_service.py`. |
| 3 | Missing uptime reporting | Operators could not distinguish a fresh process from a restarted one. | No start timestamp. | Track process start and report uptime in health/readiness. | Health test asserts the field. |
| 4 | Secret scanner false positives | The scanner flagged `os.environ.get(...)` as a credential and a model name as a high-entropy token, which would block a legitimate push. | (a) Code references were treated as literal values; (b) the assignment key was included in the entropy window. | Skip code references/placeholders; evaluate only the right-hand side of assignments. | Planted-secret self-test still detects password, private key, GitHub token and Slack token. |

No defect was fixed by weakening an assertion or deleting a test. Defects 1–3 are in the
audit-added HTTP adapter; defect 4 is in the audit's own scanner. The ML model, threshold,
feature schema and contract were not modified.

### 9.1 Defects and weaknesses remaining (not fixed)

| # | Issue | Impact | Why not fixed |
|---|---|---|---|
| R1 | Auth-failure burst recall is low: Layer A 0.40, Layer B 0.00, Layer D 0.50. | Missed auth-failure incidents. | Fixing requires model/threshold work on the frozen artifact; the prompt forbids tuning against the blind set, and changing the model was out of scope. Documented as a real weakness. |
| R2 | Benign high-traffic FPR is elevated: 0.167 (A), 0.156 (D). | False positives under load. | Same as R1; needs a deliberate retraining/feature decision. |
| R3 | One silent label window is never materialized in Layer A. | A label window with no accepted events produces no window; it is recorded as "missing" rather than scored. | Changing window materialization could alter the contract; documented. |
| R4 | Accelerated logical time in Layer B produced 3 `insufficient_data` windows, one of them a true anomaly. | One incident window was not scored. | Harness timing choice; documented rather than hidden. |
| R5 | The HTTP binding has no authentication, rate limiting, persistence or multi-tenant isolation; buffer is in-memory and bounded. | Not production-hardened. | Explicitly an audit/deployment adapter; documented in `deploy/README.md`. |
| R6 | Container path unexecuted. | Cloud parity unknown. | Docker daemon blocked (§7.1). |
| R7 | Real external LLM provider untested. | Provider-specific behavior, auth, rate limits, latency unknown. | No credentials / authorization. |
| R8 | No independent dataset. | Real-world generalization unknown. | Layer C not executed. |

---

## 10. Test-only shortcut and leakage audit

### 10.1 Static review

| Check | Command / method | Result |
|---|---|---|
| Hard-coded scenario names in production code | `grep -rn "http_5xx_burst\|auth_failure_burst\|db_timeout_latency\|…" src/ deploy/` | Only in `network/simulator.py` (the generator) — none in service, features, inference or correlation. |
| Test code imported by production | `grep -rn "import tests\|from tests"` | None. |
| Fixtures used by production | `grep -rn "fixtures" src/anomaly_detector deploy/` | Only `e2e_scenarios.py` (an existing scenario module), not the HTTP service. |
| Label columns in features | Feature list inspected | `is_anomaly` / `scenario_name` exist only on `NetworkLabel`, the evaluation-only dataclass. |
| Hard-coded expected scores / canned LLM output | Source review | None in the service; the mock provider computes text from the supplied bundle and is labelled `mock`. |
| Test-only branch in service | Source review | No `if TEST`/`if DEBUG` branch that changes verdicts. |
| Container-only mocked services | Compose review | `mock-llm` is an opt-in profile and only serves the provider boundary, never scoring. |
| Mounted fixtures shadowing artifacts | Compose review | The artifact mount is `../artifacts:...:ro`; no fixture path is mounted over it. |

### 10.2 Runtime controls

| Control | Evidence |
|---|---|
| Label keys stripped at ingress | `test_ingest_infer_and_label_key_stripping`: 18+ evaluation keys dropped and counted; response contains no `ground_truth`, `scenario_name`, `label`, test ID or scenario value. |
| No labels in provider prompt | `test_prompt_payload_contains_no_evaluation_metadata`: a capture provider records the exact payload; assertions reject label/scenario/test-id markers. |
| Unscored ≠ normal | Layer B and D: `insufficient_data` windows are excluded from the confusion matrix and reported separately; the one unscored incident window is called out. |
| Threshold frozen | Layer D loads the artifact read-only and performs no tuning. |
| Artifact identity | SHA-256 reported by `/readyz`, `/v1/model` and every run manifest. |
| Redaction | `test_redaction_removes_secrets_and_pii`: passwords, API keys, bearer tokens, emails and IPv4 are replaced. |
| Fail closed | `test_model_unavailable_fails_closed`: readiness `503`, inference `503 model_unavailable`, no `results` array. |

---

## 11. Mandatory anti-self-validation answers

**1. Did the AI or developer create the cases, labels, and expected outcomes?**
Yes for Layers A, B and D. The simulator, the local lab fault schedule and the blind seed
are all project-authored. No independently authored labeled corpus was available. Layer C
was not executed.

**2. Did the system pass only on examples created by its own author?**
It passed with limitations, and not uniformly. Layer A F1 is 0.75; the auth-failure
scenario is weak (0.40). Layer D, which uses a seed not used for fitting, still reaches F1
0.749 / ROC-AUC 0.957, so behavior is not tied to one fixture. But Layer D is the same
generator family, so this is **generalisation to a new draw, not to the real world**.

**3. Could labels, scenario names, test IDs, or future events leak into model inputs or LLM
prompts?**
Not through the audited paths. Ingress strips and counts non-allowlisted keys; feature
columns contain only operational measurements; the provider payload was captured and
asserted clean; rolling features use only prior windows. Tests exist for each control. No
leak was observed.

**4. Were any fixtures, thresholds, rules, or prompts tailored to exact test templates?**
No. The threshold comes from the frozen manifest and was not tuned during this audit. The
rule baseline and prompts are the project's existing ones. The mock provider derives its
text from the evidence bundle rather than from canned per-scenario answers.

**5. Was the actual local pipeline exercised end to end?**
Yes. A real HTTP process loaded the real artifact and ran inbound parsing, windowing,
inference, decision, evidence correlation and grounded explanation. 7,216 events produced
269 scored windows, 40 candidates, 11 incidents and 40 grounded explanations.

**6. Was actual telemetry from local services collected through the real path?**
Yes, in Layer B: a real local web app and a real SQLite database were driven and their own
telemetry was ingested through the real endpoint. It is controlled and developer-scripted,
not production telemetry.

**7. Was independent or blind held-out data evaluated?**
Blind, yes (Layer D, unseen seed, frozen artifact). Independent, **no**. Layer C was not
executed.

**8. Was the cloud/containerized deployment exercised end to end, and does it match local
behavior?**
**No.** The Docker daemon was inaccessible. Deployment artifacts exist and were statically
validated; the env contract was replayed on a host process with identical decisions. That is
not container execution and is not reported as parity.

**9. Which validation layers and which deployments remain incomplete, and why?**
- Layer C: no independent dataset; download requires authorization.
- Layer E: only scripted variations; unscripted production traffic untested.
- Path C: Docker daemon permission denied; no Compose plugin; no rootless runtime.
- Real LLM provider: no credentials / authorized endpoint.
- Container failure modes (cold start, egress denial, restart, resource exhaustion):
  untested.
- Layer B breadth: one host, three services, 26 logical minutes.

---

## 12. PR-style summary

**Title:** Audit: dual-deployment end-to-end ML + LLM verification (local pass with limitations, container blocked)

**Body:**

> This change adds a bounded HTTP binding, a containerized deployment area, three
> evaluation harnesses and a pre-push secret scanner, then audits the existing anomaly
> detector end to end.
>
> **What was audited**
> - Real artifact, parser, `network-features-v1` windowing, Isolation Forest, threshold
>   decision, evidence adapter and correlation pipeline.
> - Local path with a real HTTP process and the real model.
> - Controlled local-service telemetry (web app + SQLite) through the real ingestion path.
> - Blind evaluation on an unseen simulator seed with the frozen artifact.
> - Label stripping, prompt purity, redaction, fail-closed behavior and unscored handling.
>
> **What was added**
> - `deploy/` service, Dockerfile, Compose stack, env template, mock provider, local lab
>   and README.
> - `scripts/` stream, evaluator, blind-holdout and secret-scan tools.
> - `tests/test_deploy_service.py` (18 tests).
> - Plan, report and combined audit bundle.
>
> **Results**
> - Regression suite: 411 → **429 passed** (18 new tests, 27 warnings).
> - Layer A: 269 scored windows, precision 0.750 / recall 0.750 / F1 0.750.
> - Layer B: 75 scored (+3 `insufficient_data`), precision 1.000 / recall 0.714 / F1 0.833.
> - Layer D: 807 scored, precision 0.698 / recall 0.808 / F1 0.749, ROC-AUC 0.957, 9/9
>   scenario groups detected, mean delay 0.22 min.
> - Environment-contract parity: 269/269 identical window decisions.
>
> **Blocked / incomplete**
> - **Container build and run are blocked**: Docker daemon socket permission denied; no
>   Compose plugin; no rootless runtime. Container parity is therefore not claimed.
> - Layer C (independent dataset) not executed.
> - Real external LLM provider untested (no credentials).
>
> **Known weaknesses (not hidden)**
> - Auth-failure burst recall is low (0.40 / 0.00 / 0.50 across Layers A / B / D); the rule
>   baseline outperforms the model on that scenario in Layer D.
> - Benign high-traffic false-positive rate is elevated (0.167 / 0.156).
> - The HTTP binding is an audit adapter, not a hardened production service.
>
> **Safety**
> - No secrets, `.env`, datasets, model binaries or archives are committed.
> - Secret scan passes on the intended file set.
> - Not merged to `main`.

---

## 13. Reproduction and artifact locations

- Raw run evidence: `/tmp/opencode/dual-e2e/`
- Deployment instructions: `deploy/README.md`
- Test plan: `docs/ml_llm_e2e_test_plan.md`
- Combined bundle: `docs/opencode_dual_deployment_audit_bundle.md`

Minimal reproduction:

```bash
python -m pytest -q
python -m deploy.service --host 127.0.0.1 --port 18081 --deployment-mode local &
python scripts/stream_network_telemetry.py \
  --url http://127.0.0.1:18081 --run-index 2 --explain \
  --out /tmp/opencode/dual-e2e/layer-a
python deploy/local_telemetry_lab.py \
  --detector-url http://127.0.0.1:18081 \
  --out /tmp/opencode/dual-e2e/layer-b
python scripts/run_blind_holdout.py --seed 20261026 \
  --out /tmp/opencode/dual-e2e/layer-d
python scripts/secret_scan.py --paths deploy src scripts tests docs
```

---

## 14. Final status declaration

| Deployment / layer | Status |
|---|---|
| Local (Path L) | **PASS WITH LIMITATIONS** |
| Cloud/container (Path C) | **BLOCKED** |
| Parity (container) | **BLOCKED** |
| Parity (env contract, host) | Passed, parity-adjacent only |
| Layer A | Passed with limitations (developer synthetic) |
| Layer B | Passed with limitations (controlled local services) |
| Layer C | **NOT EXECUTED** |
| Layer D | Passed with limitations (seed-blind, same generator) |
| Layer E | **PARTIAL** |
| **Overall** | **BLOCKED** |

The local system demonstrably works end to end on the evidence above. The containerized
system is **unverified**, not "working", because it could not be executed. No independent
real-world validation was performed.
