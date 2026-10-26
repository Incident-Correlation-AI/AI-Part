# Network / Service-Log Anomaly Detection — Reconnaissance

**Status:** complete for the accessible repositories · **Date:** 2026-10-26
**Scope:** what existed before this extension, what the canonical contracts are,
and why the network detector was added as an additive component rather than a
replacement.

This document records the reconnaissance that preceded implementation. It is
deliberately explicit about what could **not** be inspected, so that no
unverified upstream behaviour is presented as fact.

---

## 1. Questions the reconnaissance had to answer

1. What is the canonical event envelope, and who owns it?
2. What does the existing ML component already expose, and which parts are
   frozen contracts?
3. Is there an existing upstream ML / LLM component that this work must extend
   rather than duplicate?
4. Which network/service incidents are in scope, and what would count as a
   false positive?
5. Where can labels live so that they cannot leak into model features?

---

## 2. Current repository — what already existed

The repository at reconnaissance time was a standalone ML component on branch
`main` at commit `747457f` (Milestone 8), with **no Git remote configured**.

Relevant existing documents:

| File | What it establishes |
|---|---|
| `docs/ml_inference_contract.md` | The versioned request/result boundary (`1.0.0`, `features-v1`). |
| `docs/ML_HANDOFF.md` | The downstream caller's quick start and obligations. |
| `docs/INTEGRATION_BOUNDARY.md` | The forwarding rule and ownership split. |
| `docs/READINESS_AUDIT.md` | The M7 verdict: **controlled demo only**. |
| `docs/DATA_READINESS.md`, `docs/BASELINE.md`, `docs/MODEL_COMPARISON.md` | Data, baseline and comparison evidence. |

Relevant existing code:

| Module | Contract-relevant behaviour |
|---|---|
| `src/anomaly_detector/schemas.py` | `StructuredEvent`, `FeatureWindow`, `WindowMetadata`, `EventType`. |
| `src/anomaly_detector/preprocessing.py` | Epoch-aligned, half-open `[start, end)` windows per host/service. |
| `src/anomaly_detector/features.py` | The frozen 13-column `features-v1` allowlist. |
| `src/anomaly_detector/inference/contract.py` | `InferenceResult`, `InferenceStatus`, `ReasonCode`, `CONTRACT_VERSION`, `route_result()`. |
| `src/anomaly_detector/baseline_model.py` | `IsolationForest`, `score_samples`, lower = more anomalous. |
| `src/anomaly_detector/integration_harness.py` | Scored-only forwarding and explicit unscored handling. |

Two properties were treated as non-negotiable:

* `features-v1` and its 13 features must not change. Existing artifacts and
  downstream callers depend on the exact allowlist and order.
* The result object must remain `InferenceResult`. A second result schema would
  fork the integration surface.

---

## 3. Upstream repository access — blocker

The specified upstream repository was:

```
https://github.com/kalmak889900/automated-log-anomaly-detector
```

It could not be accessed from this environment:

| Attempt | Result |
|---|---|
| `git clone https://github.com/kalmak889900/automated-log-anomaly-detector` | Failed: `fatal: could not read Username for 'https://github.com'` |
| GitHub REST API lookup | `404` |
| `gh repo view kalmak889900/automated-log-anomaly-detector` | `Could not resolve to a Repository.` |

`gh auth status` showed an authenticated account (`Barok-y`, SSH protocol) with
`repo` scope; the failure is therefore repository visibility/identity, not a
missing token. The token value was not read or recorded.

**Consequence:** the exact upstream ML and LLM code could not be inspected. This
extension therefore does **not** claim to match it. The blocker is carried
forward in §8 instead of being silently replaced by a different repository.

### Update — M10 (2026-10-26)

The blocker above was resolved through SSH access after M9. The repository was
cloned read-only at `b155a05c00741cc5f129610d2d90af0d18ca48ed` and fully
inspected. Key result: **it contains no LLM**. It provides a rule-based
`correlator.py` (`correlate_anomalies`), MAD detection and upstream incident
construction. M10 vendors only `correlator.py` and maps its groups into the
canonical LAD incident shape; the optional explanation layer is this
repository's existing `network.llm` adapter. See
[`ml_llm_integration_reconnaissance.md`](ml_llm_integration_reconnaissance.md)
for the full findings and the premise mismatch.

---

## 4. Accessible LAD repositories — inspected

The public organisation `Incident-Correlation-AI` was accessible. Two
repositories were cloned and inspected:

| Repository | Revision | Role |
|---|---|---|
| `Incident-Correlation-AI/LAD` | `bfad05c` | Backend/API and canonical contracts. |
| `Incident-Correlation-AI/lad-collector` | `0372eac` | Host collectors that emit canonical events. |

### 4.1 Canonical event envelope

`LAD/contracts/schemas/event.schema.json` is the source of truth for ingestion.

* `schema_version` constant: `1.0`
* Required fields: `schema_version`, `event_id`, `event_ts`, `observed_ts`,
  `host`, `source`, `category`, `type`, `severity`, `message`
* Optional fields: `service`, `attributes`, `depends_on`, `evidence_ref`,
  `request_id`, `template`, `trace_id`

The ML component does not invent a third event format. The network adapter maps
this envelope into the existing `StructuredEvent` and records every rejection
and partial-quality problem. See
[`ml_network_features.md`](ml_network_features.md) §4.

### 4.2 Ingestion endpoint

`LAD/services/api/routers/ingestion.py` exposes:

```
POST /api/v1/events/batch
```

The collector authenticates and submits batches. A `202` means the batch was
**queued**, not analysed. ML windowing and scoring remain a separate stage; this
extension does not change ingestion, storage or the API.

### 4.3 Incident contract

`LAD/contracts/schemas/incident.schema.json` requires, among other fields:

* `incident_id`, `schema_version`, `status`, `severity`
* `affected_entities`, `anomaly_score`, `detector`
* `evidence` with **`supporting_event_ids`** and
  **`contradicting_event_ids`** (structurally separate)
* `hypotheses` as a separate array
* `diagnostic_checks` whose recommendations carry `executes: false`
* `created_at`, `updated_at`, `audit`

This is a downstream incident representation. The ML component produces a
scored window plus evidence; it does not populate an incident, choose
hypotheses or execute checks. See
[`ml_network_integration.md`](ml_network_integration.md) §8.

### 4.4 LLM / explanation component

No LLM or explanation module was found in `LAD`, `lad-collector` or the current
ML repository. Searches for `llm`, `openai`, `anthropic`, `bedrock` and `gpt`
returned only dashboard dependency noise and, after implementation, the new
adapter itself.

**Consequence:** there was no upstream LLM integration to extend or verify. The
explanation adapter is a narrow, new, optional boundary. It is disabled by
default, makes no network call unless explicitly configured, and is documented
as an upstream unknown rather than presented as an existing integration.

---

## 5. Gap analysis

| Need | Before this extension | Decision |
|---|---|---|
| HTTP 5xx spikes | `features-v1` has HTTP counts and `http_5xx_ratio`. | Reuse the concepts; add explicit 4xx/eligible denominators. |
| DB timeouts / latency | `features-v1` has latency but no DB timeout or connection-failure signal. | Add DB counts, ratios and latency percentiles. |
| Auth-failure bursts | `features-v1` has `auth_failure_count` only. | Add success count, failure ratio and distinct-source/account counts. |
| Benign traffic / latency / restart shifts | Covered only indirectly by generic counts. | Add explicit benign scenarios so false positives are measured. |
| Missing/malformed telemetry | Existing statuses cover missing values, but there was no `no_events` reason and no per-window quality counts. | Add `network-features-v1` quality features and three additive reason codes. |
| Causal network baselines | No rolling per-service history in `features-v1`. | Add causal rolling z-scores, deltas and p95 ratio. |
| Network labels | Existing synthetic generator is for the generic schema. | Add a separate, deterministic network simulator with a separate label channel. |
| Transparent evidence | Existing baseline is a model, not a rule engine. | Add a rule/MAD baseline evaluated independently; it never rewrites the model flag. |
| Explanation | None upstream. | Add an optional, evidence-grounded adapter; disabled by default. |

---

## 6. Design decisions

1. **Additive feature schema, not a mutated `features-v1`.**
   `network-features-v1` is a separate 39-column allowlist. The existing
   13-column schema, its order and its artifacts are untouched.

2. **Additive contract version.**
   `CONTRACT_VERSION` moved from `1.0.0` to `1.1.0`. The only changes are
   additive: a new feature-schema identifier, three reason codes
   (`no_events`, `insufficient_coverage`, `telemetry_quality`) and
   `EventType.AUTH_SUCCESS`. Existing statuses, score direction and decision
   rule are unchanged.

3. **Canonical LAD events in, existing window semantics out.**
   The simulator emits LAD `1.0` envelopes. The adapter maps them to the
   existing `StructuredEvent`, and windows remain epoch-aligned half-open
   per-host/service windows.

4. **Labels are a separate channel.**
   `labels.csv` is written beside `events.jsonl`, never inside it. Labels,
   scenario names and run IDs are excluded from the feature mapping and checked
   by tests.

5. **Two split strategies, one deployed default.**
   The pipeline runs both `time` and `group`. `time` is the deployed default
   because it is chronological. `group` is reported as a cross-check that keeps
   whole runs disjoint.

6. **Training is normal-only, clean-only and training-split-only.**
   Validation selects the threshold. Test is not used for selection. Missing
   values are imputed with training medians only.

7. **Unscored is unknown.**
   Empty or degraded windows return `insufficient_data` with an explicit reason
   code and `is_anomaly = None`. They are excluded from supervised metrics and
   reported separately.

8. **The rule baseline is evidence, not a second decision.**
   `is_anomaly` is the Isolation Forest decision. The rule baseline contributes
   severity and `extra.network.baseline` evidence only.

9. **LLM output is grounded and non-authoritative.**
   The adapter may explain a scored result, may never explain an unscored one as
   a detection, rejects fabricated numbers/unknown citations/evaluation
   vocabulary/PII, falls back to a deterministic evidence summary, and never
   changes `is_anomaly`.

---

## 7. What was deliberately not done

* No change to the canonical LAD event schema or incident schema.
* No change to `features-v1`, its order or its existing artifacts.
* No new HTTP API, queue, dashboard or alert delivery.
* No ingestion change; the collector keeps `POST /api/v1/events/batch`.
* No real login attempts, exploit traffic, external host contact or cloud call.
* No claim that synthetic metrics prove real-world detection quality.
* No silent substitution of a different upstream repository.

---

## 8. Open items carried forward

| # | Item | Why it remains open |
|---|---|---|
| 1 | Exact upstream `kalmak889900/automated-log-anomaly-detector` code | **Resolved in M10:** accessible via SSH; inspected at `b155a05`. See [`ml_llm_integration_reconnaissance.md`](ml_llm_integration_reconnaissance.md). |
| 2 | A real upstream LLM/explanation contract | **Resolved as a premise mismatch:** upstream has no LLM. M10 reuses the local optional explanation adapter. |
| 3 | Real production log replay | Only deterministic synthetic data was available; no production telemetry was used. |
| 4 | Destination repository and branch for publication | **Resolved:** `Incident-Correlation-AI/AI-Part`, branch `main`. |
| 5 | Posting an integration comment to Abel | Requires an unambiguous identity and target URL; otherwise the draft stays local. |

---

## 9. Sources inspected

| Path | Revision / note |
|---|---|
| `docs/ml_inference_contract.md` | Current repo, `747457f` |
| `docs/ML_HANDOFF.md`, `docs/INTEGRATION_BOUNDARY.md` | Current repo, `747457f` |
| `src/anomaly_detector/inference/contract.py` | Current repo, `747457f` |
| `src/anomaly_detector/schemas.py`, `src/anomaly_detector/features.py` | Current repo, `747457f` |
| `../LAD/contracts/schemas/event.schema.json` | `bfad05c` |
| `../LAD/contracts/schemas/incident.schema.json` | `bfad05c` |
| `../LAD/services/api/routers/ingestion.py` | `bfad05c` |
| `../lad-collector/core/engine.py`, `linux/collector.py` | `0372eac` |
