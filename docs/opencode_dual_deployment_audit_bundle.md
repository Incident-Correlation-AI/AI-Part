# Dual-Deployment (Local + Cloud) Real End-to-End ML + LLM Verification — Audit Bundle

This single file is the downloadable deliverable requested by `opencode_prompt.md`.
It contains, in order:

1. The governing prompt, verbatim (Section 1).
2. The completed `docs/ml_llm_e2e_test_plan.md` (Section 2).
3. The completed `docs/ml_llm_e2e_test_report.md` (Section 3).
4. The PR-style summary (Section 4).
5. The GitHub branch / commit / push record and delivery note (Section 5).

**Status at a glance**

| Deployment / layer | Status |
|---|---|
| Local (Path L) | PASS WITH LIMITATIONS |
| Cloud/container (Path C) | BLOCKED |
| Container parity | BLOCKED |
| Env-contract host parity | PASSED (parity-adjacent only) |
| Layer A (developer synthetic) | PASSED WITH LIMITATIONS |
| Layer B (controlled local services) | PASSED WITH LIMITATIONS |
| Layer C (independent dataset) | NOT EXECUTED |
| Layer D (blind seed) | PASSED WITH LIMITATIONS |
| Layer E (unscripted) | PARTIAL |
| **Overall** | **BLOCKED** |

No secrets, credentials, private telemetry, model binaries, datasets or archives are
included in this bundle.

---

# Section 1 — Governing prompt (verbatim)

# OpenCode Master Prompt: Dual-Deployment (Local + Cloud) Real End-to-End ML + LLM Verification

## Mission

Act as a senior ML engineer, backend engineer, DevOps/SRE engineer, and independent software test engineer. Audit, execute, validate, and improve the existing Automated Log Anomaly Detector pipeline across **two real execution paths**:

- **Path L — Local backend binding**: the project run directly on the host (e.g. `uvicorn`, `node`, `python -m`, `make run`) with a local model artifact and local LLM provider config.
- **Path C — Cloud/containerized deployment**: the project built and run via Docker (and/or Docker Compose / the project's cloud manifest) in a local container runtime, mirroring how it would run in the cloud (env-var config, mounted model artifacts, healthchecks, egress to an external LLM provider or a mocked provider boundary).

The system includes:

- A baseline ML anomaly detector and an Isolation Forest-based detector.
- A real telemetry ingestion and normalization pipeline.
- An ML inference interface.
- An evidence adapter that passes eligible scored findings downstream.
- An existing evidence-correlation LLM and its orchestration.
- Defined integration contracts and downstream handling.
- A containerization layer (Dockerfile / Compose / deploy manifests) — to be created or extended as part of this work.

The goal is to establish whether the **actual built system works in both deployments**, identify defects, fix them where appropriate, rerun the relevant tests, document the evidence, **push the work to GitHub**, and provide a downloadable `.md` report bundle.

**Do not merely create your own normal and anomalous examples, test against those same examples, and declare the system correct.** That is circular validation. Developer-created synthetic tests are useful for regression testing, but they are not independent proof of correctness.

## Non-negotiable rules

1. Read the architecture documents, ML inference contract, handoff documentation, source code, configuration, existing tests, and any existing Docker/Compose/cloud files before changing anything.
2. Inspect the Git working tree and current branch. Do not overwrite or discard unrelated work.
3. Preserve the documented interfaces unless a genuine defect requires a change. Document any contract changes and update affected tests and documentation.
4. Test the actual application through **both** real execution paths where feasible:
   - Local backend binding.
   - Containerized/cloud-parity deployment.
   Do not substitute mocked components for the main end-to-end run.
5. Use mocks only where appropriate for isolated unit tests, unavailable external services, or deterministic tests. Clearly label which components were mocked **and in which deployment** they were mocked.
6. Never claim that a test passed unless you actually ran it and observed the result — in the specific deployment you claim.
7. Never claim independent or real-world validation based solely on AI-generated synthetic examples.
8. Keep ground-truth labels, scenario names, test IDs, and future events out of model inputs, features, inference requests, and LLM prompts.
9. Do not hard-code test scenarios, labels, expected outputs, or canned responses merely to make the evaluation pass.
10. Use safe local environments and synthetic test data. Do not attack public systems, third-party services, or production infrastructure.
11. Do not expose secrets, credentials, private telemetry, or sensitive information in logs, reports, images, or git history.
12. **GitHub push is authorized for this task** — but only after the privacy/secret scan in Rule 11 passes and only to the project's own repository/branch. Never force-push over others' work. Never commit `.env`, credentials, private telemetry, model weights above the repo's size policy, or large datasets. Push a branch and open a PR-style summary; do not merge to `main` unless explicitly told.
13. If a test cannot run because a dependency, dataset, credential, service, or container runtime is unavailable, document the blocker rather than pretending it passed.

---

## Critical anti-self-validation audit

Before accepting any results, investigate whether the AI or developer has accidentally built an evaluation that only proves the implementation works on its own examples — **and whether "works locally" is being quietly substituted for "works in cloud."**

Inspect:

- Synthetic data generators and scenario templates.
- Training data, fixtures, test labels, and evaluation datasets.
- Feature engineering, normalization, thresholds, and scoring logic.
- Test assertions, evaluation scripts, and expected outcomes.
- LLM prompts, mock responses, canned outputs, and test-only configuration.
- Whether labels or scenario metadata can leak into inference.
- **Deployment-only differences**: env vars, mounted model artifacts, container entrypoints, healthcheck timing, DNS/egress, and whether any behavior differs between Path L and Path C.

Answer explicitly:

1. Who or what created each dataset and its labels?
2. Were the cases created by the same AI or developer who implemented the detector?
3. Were the thresholds, features, prompts, or rules tuned against those exact cases?
4. Can the detector or LLM see the answer key, scenario names, test IDs, or future events?
5. Does any code path detect a test scenario by name or use a hard-coded response?
6. Were actual application components exercised, or were they replaced with mocks — and in which deployment?
7. What evidence exists that the system generalizes beyond its own fixtures?
8. **Does the cloud/containerized run exercise the same code paths as the local run, or does it silently bypass, mock, or replace them?**
9. **Are any cloud-only env vars, mounts, or provider configs causing the container to behave differently from local in a way that affects verdicts?**

Record findings in `docs/ml_llm_e2e_test_report.md`.

### Required validation layers

**Layer A — Developer-authored synthetic regression tests**
Test known edge cases, reproducibility, contract behavior, and regressions. Keep results separate from independent validation. Run in **both** Path L and Path C where the test is deployment-relevant.

**Layer B — Actual telemetry from controlled local services**
Run real local services (local web app, test database), exercise ordinary usage, errors, latency, restarts, auth failures, and recovery. Collect telemetry through the project's actual collectors and ingestion pipeline. Do this at least once via Path L and once via Path C (container talking to the same local services via the documented network).

**Layer C — Independent or pre-existing datasets**
Search the repo and available project resources for independently authored or previously existing log datasets. Record provenance, license, schema mapping, missing fields, preprocessing, and limitations. If none is available locally, identify a candidate and ask before downloading/transmitting. Do not claim this layer passed if it was not executed.

**Layer D — Blind held-out evaluation**
Reserve cases or time periods not used to develop features, tune thresholds, or debug the detector. Keep labels in an evaluator-only file. Do not expose them to model inputs, feature generation, LLM prompts, or the application under test in **either** deployment. If a genuinely independent blind evaluation cannot be established, say so explicitly.

**Layer E — Unscripted variations and borderline cases**
Vary event rates, ordering, baseline duration, time of day, service mix, latency distributions, missing fields, duplicates, severity, and recovery. Include benign high-volume traffic and benign distribution shifts. Ground outcomes in independent labels, recorded controlled faults, or documented human-reviewed judgments. Mark ambiguous cases inconclusive.

---

## Phase 1 — Discover the actual implementation

1. Inspect repository structure, Git state, and existing Docker/Compose/cloud files.
2. Read all relevant architecture and integration documentation.
3. Locate:
   - Telemetry collectors and ingestion endpoints.
   - Normalization and feature extraction.
   - Baseline model and Isolation Forest implementation.
   - Model artifacts and training code.
   - ML inference endpoint and response schema.
   - Scored and unscored result handling.
   - Evidence adapter and downstream correlation path.
   - Existing LLM orchestration and provider boundary.
   - Redaction and privacy controls.
   - Existing tests, fixtures, and simulation scripts.
   - **Dockerfile(s), Compose file(s), deployment manifests, entrypoints, healthchecks, env templates, and CI files.**
4. Identify actual startup commands **for Path L and Path C separately**, dependencies, configuration, and verified endpoint routes.
5. Determine whether model training and inference use compatible feature schemas — **and whether the container loads the same artifact version as the local run.**
6. Identify gaps between documented behavior and implementation, and between local and cloud behavior.
7. Audit test data for circular validation and label leakage before relying on existing results.

Do not assume a route, config setting, model artifact, feature, or container target exists until you verify it.

Create `docs/ml_llm_e2e_test_plan.md` with the verified architecture, both startup paths, routes, schemas, data provenance, independence controls, test plan, deployment-parity plan, and blockers.

## Phase 2 — Establish a baseline

Run existing tests **locally first**, before changes. Record commands, passes/failures, skipped tests, dependency/environment failures, known defects, and pre-existing Git changes. Do not silently modify tests to make the baseline green.

If a container build already exists, build it and record whether it currently builds and starts — but treat that as informational until Phase 3.5.

## Phase 3 — Build or improve the bounded telemetry simulator

The simulator is one validation layer, not the whole evaluation. Generate time-ordered telemetry gradually at a bounded pace so the app processes events as they arrive (not a static CSV). Include normal traffic, 5xx spikes, DB timeouts/latency, auth-failure bursts, benign high traffic, benign distribution shifts, malformed/duplicate telemetry, restart/recovery, short and sustained anomalies, and borderline cases.

Requirements:
1. Use the real ingestion/normalization path where feasible.
2. Keep labels/test IDs outside model inputs.
3. Record ground truth separately for the evaluator.
4. Don't reuse the same templates/seeds for development and final evaluation.
5. Reserve unseen variations for blind evaluation.
6. Clearly mark generated data as synthetic.
7. Never present simulator results as independent proof of real-world performance.
8. Run simulator-driven checks in **both** deployments where feasible.

## Phase 3.5 — Prepare the cloud/containerized deployment (Docker place)

Create or extend a deployment area (e.g. `deploy/` or `docker/`) so the cloud-based deployment can be built, run, and tested locally with cloud parity. Deliver at minimum:

- `Dockerfile` (multi-stage if appropriate) with a pinned base image, non-root user, and a clear entrypoint matching the documented Path C startup.
- `docker-compose.yml` (or the project's equivalent) wiring the app, any local backing services (e.g. DB), env vars from a **checked-in `.env.example`** (never real secrets), volumes for model artifacts and logs, healthchecks, and restart policy.
- `deploy/README.md` with: build command, run command, required env vars, artifact mount paths, exposed ports, healthcheck endpoint, how to point at a local vs real LLM provider, and how to run the test suite **inside** the container.
- `.dockerignore` to keep secrets, datasets, and `.git` out of the image.
- A **secret-scan step** (e.g. `gitleaks`/`trufflehog` or a documented manual check) in the build/run instructions.

Acceptance for this phase:
- `docker build` succeeds.
- `docker compose up` brings the stack to healthy (or documented alternative).
- The ML inference endpoint responds inside the container.
- The evidence adapter + LLM orchestration path is reachable inside the container (with a mocked provider **or** an explicitly authorized real provider).
- The model artifact actually loaded in the container is identifiable (version/hash recorded).

Record build/run commands and outputs in the plan and report.

## Phase 4 — Validate features and model behavior

Verify features derive from the documented schema; timestamps/windows are consistent; missing/malformed fields handled explicitly; no leakage of labels/scenario names/test IDs/future info; training and inference feature order/types match; the model artifact actually loaded is the one evaluated — **verify this in both Path L and Path C, and record artifact hashes**; scores/thresholds/statuses follow contract; benign variation doesn't auto-become anomaly; measure both missed anomalies and false positives; unscored/malformed/failed results stay distinguishable from valid scored results.

Do not tune thresholds on the blind evaluation set.

## Phase 5 — Test the real ML inference interface (local)

Start the actual local services. Send valid telemetry through the actual inference endpoint. Test: valid events, multiple events over time, missing optional fields, missing required fields, invalid types, empty requests, malformed payloads, duplicates, timestamp edge cases, service restart and model-load failures, unscored outcomes, explicit error handling. Verify response structure against the ML inference contract. Record actual status codes, response fields, scores, errors, logs (redacted). Do not assume an endpoint is correct because it responds successfully.

## Phase 6 — Validate the actual LLM integration (local)

Trace a valid scored finding from ML inference through the evidence adapter into the existing evidence-correlation LLM orchestration. Verify: only eligible `scored` results enter correlation; unscored/failed handled explicitly and don't masquerade as valid findings; evidence fields map correctly; timestamps/ordering correct; evidence redacted before any external provider; LLM receives needed evidence without hidden labels/answer keys; explanations grounded in supplied evidence; LLM doesn't invent missing telemetry or unsupported root causes; provider errors/timeouts/malformed responses/unavailable credentials handled; output conforms to schema. Use a mocked provider for deterministic unit tests if needed, but also test the actual orchestration/adapter, and distinguish mocked-provider from real-provider runs. Do not send private telemetry to an external model without explicit authorization. If real-provider testing is unavailable, report as untested.

## Phase 6.5 — Validate the cloud/containerized deployment and cross-deployment parity

Using the Phase 3.5 setup:

1. Build the image and start the stack (`docker compose up` or equivalent).
2. Confirm the container reaches healthy state and the ML inference endpoint answers **inside** the container.
3. Re-run the Phase 5 inference interface tests against the containerized endpoint. Record status codes, schemas, scores, errors.
4. Re-run the Phase 6 LLM-orchestration trace against the containerized path (mocked provider acceptable; real provider only with explicit authorization). Record redaction behavior and output schema.
5. Run the developer-authored regression suite **inside** the container where the suite is deployment-relevant.
6. **Parity check**: compare Path L vs Path C on the same inputs. Record:
   - Identical vs divergent outputs/scores/statuses.
   - Model artifact hash equality.
   - Env-var and config differences that could change verdicts.
   - Any silent mock/replacement in the container path.
7. Test cloud-only failure modes: cold start (model load time), missing/mis-mounted artifact, env-var drift, egress failure to LLM provider, container restart/recovery, resource limits.
8. Confirm that failure cases in the container cannot silently become valid scored findings or trigger misleading downstream conclusions.

Never report a mocked-provider container run as proof that a real external service behaves the same way.

## Phase 7 — Evaluate detection quality independently

Produce **separate** results for:
A. Developer-authored synthetic regression cases.
B. Actual telemetry from controlled local services.
C. Independent or blind held-out datasets (if available).

Never combine these into one aggregate score that hides the absence of independent validation. **Additionally**, report results per deployment (Path L / Path C) so cloud-only regressions are visible.

For datasets with trustworthy labels, compute precision, recall, F1, false-positive rate, confusion matrix, and detection delay where measurable. Include sample counts and explain each metric. For unlabeled data, report descriptive findings and limitations; do not manufacture labels.

Report results by scenario, data source, and deployment. Identify false positives, missed anomalies, and inconclusive cases. If actual local-service, independent-data, or cloud-deployment testing is blocked, document the exact reason and downgrade the overall status accordingly.

## Phase 8 — Failure and robustness testing

Test realistic failures in **both** deployments where feasible: ML endpoint unavailable; model artifact missing/incompatible; invalid/partial telemetry; out-of-order timestamps; duplicate delivery; LLM provider timeout; rate limiting/failure; malformed LLM output; missing evidence fields; redaction failures; unexpectedly large event volumes within safe local limits; restart/recovery; **container-specific**: cold-start timeout, mis-mounted artifact, egress denial, resource exhaustion.

Confirm failure cases cannot silently become valid scored findings or trigger misleading downstream conclusions. Never report mocked failure behavior as proof that a real external service behaves the same way.

## Phase 9 — Check for test-only shortcuts

Search for hard-coded scenario names, exact test-payload matching, fixed expected scores, hard-coded labels/outcomes, test-only branches enabled during the claimed real run, canned LLM responses mistaken for real provider output, fixtures used unintentionally by production code, label/future-event leakage, thresholds tuned against the final evaluation set — **and cloud-specific shortcuts** (e.g. container entrypoint that runs a canned demo, env var that flips to "test mode" in the image, mounted fixtures that shadow real artifacts, Compose overrides that silently mock services).

For any suspicious code, determine whether it is legitimate application behavior or an evaluation shortcut. Fix genuine defects without removing useful tests.

## Phase 10 — Fix defects and rerun

Fix reproducible defects in scope. For every fix: explain the observed failure; identify root cause; make the smallest appropriate change; add a regression test; rerun relevant tests; rerun the affected end-to-end path **in both deployments where relevant**; record the final observed result. Preserve unrelated work. Do not weaken assertions, delete failing tests, or alter expected outcomes just to produce a passing report. If a defect cannot be fixed safely, document it and its impact.

## Phase 11 — GitHub push

After Phases 1–10 and after the privacy/secret scan passes:

1. Create a feature branch (e.g. `audit/dual-deployment-e2e-<date>`) from the current working branch.
2. Stage only the intended changes (source fixes, tests, `deploy/` files, docs, `.env.example`). Do **not** stage `.env`, real credentials, private telemetry, large model weights beyond repo policy, or large datasets. Update `.gitignore` / `.dockerignore` as needed.
3. Run the secret scan (`gitleaks`/`trufflehog` or documented manual check) over staged changes; abort and remediate if anything is found.
4. Commit with a clear message summarizing the audit, fixes, and deployment setup.
5. Push the branch to the project's own remote. Do **not** force-push. Do **not** merge to `main`/protected branches unless explicitly instructed.
6. Prepare a PR-style summary (title + body) in the report: what was audited, what was fixed, what deployment was added, which validation layers passed/blocked, and which remain incomplete.
7. Record the branch name, commit SHA, and push result in the report.

If no remote is configured or credentials are unavailable, stop before any network operation, record the blocker, and still produce the local commit + downloadable `.md`.

## Phase 12 — Final documentation

Create or update:

### `docs/ml_llm_e2e_test_plan.md`
Include: verified architecture and component boundaries; **Path L and Path C startup instructions (including `deploy/README.md` cross-links)**; real endpoint routes and schemas; **Docker/Compose build+run+healthcheck+artifact-mount details and the `.env.example` contract**; test scenarios and commands (per deployment); dataset sources and provenance; ground-truth and blind-evaluation methodology; label-leakage and circular-validation controls; **cross-deployment parity methodology**; expected outcomes and known limitations.

### `docs/ml_llm_e2e_test_report.md`
Include: summary and overall status; **per-deployment status (Local / Cloud / Parity)**; Git state and files changed; **branch, commit SHA, push result**; commands actually executed; baseline and final test results; **build/run/health evidence for the container**; end-to-end evidence showing the real path was exercised in both deployments; separate metrics for synthetic / local-service / independent-held-out data, **broken down by deployment**; sample counts and evaluation limitations; false positives, missed anomalies, inconclusive cases; LLM integration, redaction, and contract results (both deployments); failure and recovery results (both deployments); **cross-deployment parity findings**; defects fixed and remaining; test-only shortcut and data-leakage audit (including cloud-specific shortcuts); reproduction commands; explicit blockers and untested areas.

### Downloadable prompt bundle
Also produce a single downloadable `.md` containing: this prompt, the plan, and the report, so it can be fetched as one file (see "Downloadable .md deliverable" below).

### Mandatory anti-self-validation report
Answer explicitly:
1. Did the AI or developer create the cases, labels, and expected outcomes?
2. Did the system pass only on examples created by its own author?
3. Could labels, scenario names, test IDs, or future events leak into model inputs or LLM prompts?
4. Were any fixtures, thresholds, rules, or prompts tailored to exact test templates?
5. Was the actual local pipeline exercised end to end?
6. Was actual telemetry from local services collected through the real path?
7. Was independent or blind held-out data evaluated?
8. **Was the cloud/containerized deployment exercised end to end, and does it match local behavior?**
9. Which validation layers and which deployments remain incomplete, and why?

Use one of these statuses **for each of Local and Cloud, plus an overall status**:

- **PASS WITH EVIDENCE:** Required execution paths and independent validation criteria actually satisfied for that deployment.
- **PASS WITH LIMITATIONS:** Tested paths passed; meaningful validation layers remain incomplete.
- **BLOCKED:** Required execution or validation could not be completed.
- **FAIL:** Reproducible defects or critical requirements not met.

Do not award a full pass solely because AI-generated cases passed. Do not award a full cloud pass solely because the local run passed.

---

## Downloadable .md deliverable

At the end of the run, produce a single Markdown file (suggested path `docs/opencode_dual_deployment_audit_bundle.md`) containing:

1. This prompt (verbatim, as the header).
2. The completed `docs/ml_llm_e2e_test_plan.md`.
3. The completed `docs/ml_llm_e2e_test_report.md`.
4. The PR-style summary and the GitHub branch/commit/push record.

Then **give the user a direct download path/URL** to that file (e.g. the raw GitHub URL on the pushed branch, or a local absolute path if the push was blocked). State clearly which one was delivered and why.

---

## Final execution instruction

Begin with repository reconnaissance, the anti-self-validation audit, and an inventory of existing Docker/Compose/cloud files. Then:

1. Run the actual built system through **Path L (local binding)**.
2. Prepare and run **Path C (Docker/cloud-parity)** using the Phase 3.5 deliverable.
3. Use controlled local-service telemetry where feasible and independent or blind held-out data where available — **in both deployments**.
4. Perform the cross-deployment parity check.
5. Fix reproducible defects, rerun, and document.
6. Push the work to GitHub per Phase 11 (after secret scan).
7. Produce the downloadable `.md` bundle and hand the user a direct download link/path.

Do not merely create your own "good" and "bad" cases and prove the system passes them. Keep labels separate, investigate leakage and test-only shortcuts, test unseen variations, exercise the actual ML endpoint and existing LLM orchestration in **both** deployments, fix reproducible defects, and rerun the tests.

Report results separately by data source **and by deployment**. **Never claim real-world, independent, or cloud-parity validation unless it was actually performed.** If a validation layer is blocked, state why and mark it accordingly.

Do not stop at code review, unit tests, an endpoint health check, or a green `docker build`. The objective is to determine what works, what fails, and what remains unverified — in local and cloud — based on evidence rather than assumptions.

---

# Section 2 — Test plan (`docs/ml_llm_e2e_test_plan.md`)

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

---

# Section 3 — Test report (`docs/ml_llm_e2e_test_report.md`)

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
| Audit commit | `AUDIT_COMMIT_SHA` |
| Remote | `origin` → `git@github.com:Incident-Correlation-AI/AI-Part.git` |
| Push result | `PUSH_RESULT` |
| Merged to `main` | **No** (intentionally not merged) |
| PR | Not opened automatically; PR-style summary in §12 |

The commit SHA and push result are filled in by the audit runner after the secret scan and
push. If the push was blocked, the local commit and this bundle remain available.

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

---

# Section 4 — PR-style summary

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

---

# Section 5 — GitHub record and delivery note

| Item | Value |
|---|---|
| Repository | `Incident-Correlation-AI/AI-Part` |
| Remote | `origin` → `git@github.com:Incident-Correlation-AI/AI-Part.git` |
| Base branch | `main` at `6d48bfc` |
| Feature branch | `audit/dual-deployment-e2e-2026-10-26` |
| Audit commit | `AUDIT_COMMIT_SHA` |
| Push result | `PUSH_RESULT` |
| Merged to `main` | No |

## Delivery note

This bundle is delivered two ways:

1. **Repository path:** `docs/opencode_dual_deployment_audit_bundle.md`
2. **Raw GitHub URL (feature branch):**
   `https://raw.githubusercontent.com/Incident-Correlation-AI/AI-Part/audit/dual-deployment-e2e-2026-10-26/docs/opencode_dual_deployment_audit_bundle.md`

The GitHub URL resolves only if the branch push succeeded. If the push was blocked, the
local absolute path above is the authoritative copy. The push result field records the
actual outcome.

## What remains unverified

- Container build, startup, in-container inference and in-container explanation path.
- Cross-deployment parity against a real container.
- Independent / real-world dataset generalization.
- A real external LLM provider.
- Unscripted production traffic and the container-specific failure modes listed in the
  report.

These are documented as blockers, not as passes.
