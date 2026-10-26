# Integration Boundary — Milestone 8

**Purpose:** define exactly how the ML anomaly-detection component hands work
to the downstream evidence-correlation / incident-explanation component.

**Status of the downstream component:** it is being built by another teammate
and is **not available yet**. This document and the harness in
`src/anomaly_detector/integration_harness.py` prepare and test the boundary
without implementing that component or depending on its code.

**The canonical contract remains
[`ml_inference_contract.md`](ml_inference_contract.md).** This document
describes the *boundary*; it does not restate the full request/result schema.

---

## 1. Files to read first

| Order | File | Why |
|---|---|---|
| 1 | [`ml_inference_contract.md`](ml_inference_contract.md) | The canonical request/result contract, statuses and behaviour matrix. |
| 2 | [`ML_HANDOFF.md`](ML_HANDOFF.md) | The teammate-facing quick start and checklist. |
| 3 | **This document** | The boundary, the routing rule and the ownership split. |
| 4 | [`READINESS_AUDIT.md`](READINESS_AUDIT.md) | What is and is not proven about the model. |
| 5 | `src/anomaly_detector/inference/adapter.py` | The reference adapter. |
| 6 | `src/anomaly_detector/integration_harness.py` | The local harness. |

---

## 2. Public request / result types

Full field reference: [`ml_inference_contract.md` §3–4](ml_inference_contract.md).

```python
from anomaly_detector.inference import (
    FeatureWindowRequest,   # input
    InferenceResult,         # output
    InferenceStatus,         # the five statuses
    ReasonCode,              # stable reason codes
    load_detector,           # model loading
    route_result,            # the canonical forwarding rule
)
```

### Request — `FeatureWindowRequest`

| Field | Required | Notes |
|---|---|---|
| `window_id` | ✅ | Stable traceable key; canonical form `host\|service\|window_start`. |
| `window_start` / `window_end` | ✅ | Timezone-aware; `end > start`. |
| `features` | ✅ | All 13 allowlisted features. `None` means *missing*, not zero. |
| `host_id`, `service_name` | ❌ | Metadata only — never model features. |
| `feature_schema_version` | ❌ | When supplied it must equal `features-v1`. |
| `extra` | ❌ | Caller metadata, echoed back untouched. |

### Result — `InferenceResult`

Carries the status, the reason code, the raw score + direction, the threshold
and its config version, the decision, the model / artifact / feature-schema
versions, the data-quality fields, and the safe window provenance. See
[`ml_inference_contract.md` §4](ml_inference_contract.md) for the full table.

---

## 3. Entry point and initialization lifecycle

```python
from anomaly_detector.inference import load_detector

detector = load_detector()          # loads the artifact + manifest ONCE
detector.contract()                  # versions, features, threshold, direction
result = detector.score_window(request)   # one window
results = detector.score_windows(requests)  # a batch
```

* `load_detector()` raises `ModelUnavailableError` when the artifact is
  missing, unreadable or mismatched. It never trains anything.
* **Load once, reuse.** The model and its training-fitted imputation medians
  are loaded into memory; nothing is fitted per call.
* The harness (`IntegrationHarness`) owns this lifecycle for you: it loads the
  model once per session and exposes `route_windows(requests)`.

---

## 4. The forwarding rule — one place, no exceptions

> **A result may enter the downstream correlation path only if its status is
> exactly `scored`.**

The rule lives in exactly one function:

```python
from anomaly_detector.inference import route_result, is_forwardable

decision = route_result(result)     # -> RoutingDecision
if decision.forwardable:
    ...                            # may be forwarded
```

`route_result()` **fails closed**. Everything below is *not* forwardable:

| Input | `forwardable` | Recorded status |
|---|---|---|
| `status is InferenceStatus.SCORED` | ✅ `True` | `scored` |
| `status is INSUFFICIENT_DATA` | ❌ `False` | `insufficient_data` |
| `status is INVALID_INPUT` | ❌ `False` | `invalid_input` |
| `status is MODEL_UNAVAILABLE` | ❌ `False` | `model_unavailable` |
| `status is INFERENCE_ERROR` | ❌ `False` | `inference_error` |
| `result is None` | ❌ `False` | `unknown` |
| `status is None` | ❌ `False` | `unknown` |
| `status` is a bare string / not an enum | ❌ `False` | `unknown` |

`route_result()` never converts an unknown outcome into `is_anomaly = False`.
Callers must not re-implement this check — that is exactly why it was
centralised in Milestone 8.

---

## 5. Handling per status

| Status | May be forwarded? | What the harness does |
|---|---|---|
| `scored` | ✅ yes | Sent to the correlation sink. The full `InferenceResult` is passed, so the downstream component keeps the provenance. |
| `insufficient_data` | ❌ no | Sent to the **unscored handler** with its status, reason code and safe provenance. |
| `invalid_input` | ❌ no | Same — the producer needs to be told its window was malformed. |
| `model_unavailable` | ❌ no | Same — this is a deployment problem, not a data problem. |
| `inference_error` | ❌ no | Same — the score could not be produced. |
| `unknown` (malformed / missing) | ❌ no | Same — fails closed, never forwarded. |

**The distinction matters.** A caller that collapses all four unscored
statuses into "normal" would silently hide a broken deployment. The harness
keeps them apart via `outcome.unscored_by_status()`.

---

## 6. Exception behavior

| Where | Behaviour |
|---|---|
| `load_detector()` | Raises `ModelUnavailableError` — the artifact is missing, unreadable or mismatched. |
| `detector.score_window()` | Never raises for **data** problems; they come back as a status. Raises `TypeError` if you pass the wrong Python type. |
| Scoring raises unexpectedly | The adapter converts it into an explicit `inference_error` result. It is never dropped, and never becomes a normal result. |
| The sink raises | **Batch:** recorded in `outcome.sink_failures`; the window is **not** in the forwarded list; the rest of the batch still runs. **Single window:** raises `SinkDeliveryError`. |
| The unscored handler raises | Swallowed deliberately — the record is already in the outcome, and a broken handler must not abort routing. |

A sink failure is **never** reported as a successful delivery, and it never
turns an unscored result into a scored one.

---

## 7. Duplicate / idempotency behaviour

**What exists today:** within a single batch, the **later** occurrence of a
duplicate `window_id` is rejected with reason code `duplicate_window_id` and
is not scored or forwarded. The first occurrence is processed normally.

**What does *not* exist today, and is not claimed:**

* no exactly-once delivery guarantee;
* no cross-batch or cross-process deduplication;
* no idempotency key storage;
* no persistence, replay, or retry semantics;
* no concurrency safety — the harness is single-process and in-memory.

If the downstream component needs at-least-once delivery, deduplication across
batches, or persistence, that is a decision for the downstream owner and
requires a contract change (see §10).

---

## 8. Provenance passed downstream — and what is excluded

**Forwarded with every scored result:**

| Field | Why |
|---|---|
| `window_id` | Links the result back to its source window. |
| `window_start`, `window_end` | Lets the downstream component order and join windows. |
| `host_id`, `service_name` | Safe source metadata for scoping the evidence lookup. |
| `model.name`, `model.version`, `model.artifact_version` | Which model produced the score. |
| `feature_schema_version` | Which feature schema was expected. |
| `threshold_config_version` | Which threshold configuration was applied. |
| `raw_score`, `score_direction` | The continuous score, in its original direction. |
| `threshold`, `threshold_rule` | The decision rule actually used. |
| `is_anomaly` | The binary decision (only when `scored`). |
| `data_quality.imputed_features`, `.ignored_features` | How much of the window was imputed or ignored. |
| `extra` | The caller's own metadata, echoed back untouched. |

**Deliberately excluded:**

| Excluded | Why |
|---|---|
| The raw feature payload | Not needed downstream; avoids copying potentially sensitive measurements into the correlation layer. |
| Labels, `is_anomaly` ground truth, scenario names, attack names | Evaluation-only. They are never in a production inference result. |
| Raw log lines, request bodies, credentials, tokens | Never entered the component in the first place. |
| Split names, model-training internals, medians | Not the downstream consumer's concern. |

**The ML component does not retrieve evidence.** It returns a score and
traceable window metadata only.

---

## 9. Responsibility ownership

| Concern | Owner |
|---|---|
| Event validation / normalization | **M1** (`validation.py`) |
| Window aggregation / feature computation | **M1 / M2** (`preprocessing.py`, `features.py`) |
| Split discipline, leakage audit, training-ready export | **M2** |
| Model training, threshold selection, evaluation | **M3** |
| Model comparison and selection decision | **M4 / M7** |
| Robustness / scenario validation | **M5** |
| Readiness audit | **M7** |
| Anomaly inference and score interpretation | **M3 / M6** (`inference/`) |
| **Scored-result eligibility and routing** | **M8** (this harness) |
| Evidence correlation / grouping | **Downstream teammate** |
| Incident explanation / root-cause hypotheses | **Downstream teammate** or a later agreed component |
| API, queue, dashboard, cloud/deployment | **Out of scope** |

**Downstream code must not re-implement the threshold, re-threshold a score,
or retrain the model.** The decision is made once, by the ML component, using
the threshold recorded in the model manifest.

---

## 10. Limitations

* The **real downstream interface is pending**. `DownstreamSink` is a neutral,
  documented protocol — a placeholder for the real adapter, not a guess at its
  shape. A neutral in-memory `RecordingSink` stands in during the rehearsal.
* The harness is **single-process and in-memory**: no queueing, persistence,
  retries, rate limiting or concurrency control.
* `duration_ms` is a wall-clock convenience, not an SLA.
* **All detection evidence is synthetic.** A flagged window is a lead for
  investigation, not proof of an attack. The raw score is a ranking
  statistic, not a probability.
* Milestone 7's verdict stands: **`controlled demo only`**.
* Any change to the request/result shape, the status vocabulary or the
  feature allowlist requires a **contract version bump** (currently
  `CONTRACT_VERSION = 1.1.0`, `FEATURE_SCHEMA_VERSION = features-v1`, with the
  additive `network-features-v1` network detector), migration notes and tests.
  The `1.1.0` change is **additive only**: three reason codes, one event type
  and a separate feature schema. The M8 changes are **backward compatible**:
  `AdapterOutcome.rejected` / `.rejected_count` still work and are now views
  over `unscored` / `unscored_count`.

---

## 11. What is needed from the teammate before the real adapter is written

The harness is deliberately generic. To replace the neutral sink with the real
downstream component, the following must be agreed:

| # | Question | Why it matters |
|---|---|---|
| 1 | **Accepted input** — is the whole `InferenceResult` acceptable, or a subset / a different shape? | Determines what the adapter must map. |
| 2 | **Desired output** — what does the downstream component return, and does it need anything back from the ML component? | The current `DownstreamSink.handle_scored_window` returns `None`. |
| 3 | **Sync or async** — may the call block, or must the ML component hand off and return? | Decides whether the sink is called inline or via a queue. |
| 4 | **Error handling** — what should happen when the downstream component fails? Retry? Drop? Escalate? | The harness currently records the failure and continues. |
| 5 | **Duplicate semantics** — is one result per `window_id` enough, or is cross-batch deduplication required? | Today only within-batch duplicates are rejected. |
| 6 | **Required provenance** — is the §8 set sufficient, or is something missing? | Missing provenance would mean a contract change. |
| 7 | **Ownership of persistence and retries** — who stores the result, who retries, who owns the queue? | Currently nobody: the harness is in-memory only. |
| 8 | **Volume and latency expectations** — is a batch expected, and what latency budget applies? | Affects whether batching is needed. |

Until these are answered, the harness stays neutral and the downstream
interface stays **explicitly pending**.

---

## 12. Exact commands

```bash
# focused boundary tests
python -m pytest tests/test_integration_boundary.py -v

# the full M1-M8 suite (offline)
python -m pytest tests/ -q

# the local rehearsal
python scripts/run_integration_rehearsal.py

# the M6 inference smoke test
python scripts/smoke_test_inference.py

# the M7 readiness audit
python scripts/run_readiness_audit.py
```

If the package is not installed, prefix with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python scripts/run_integration_rehearsal.py
```

---

## 13. Small example

```python
from datetime import datetime, timezone

from anomaly_detector.inference import InferenceStatus, load_detector
from anomaly_detector.inference.contract import FeatureWindowRequest

detector = load_detector()

request = FeatureWindowRequest(
    window_id="web-01|web-api|2026-10-18T10:01:00+00:00",
    window_start=datetime(2026, 10, 18, 10, 1, tzinfo=timezone.utc),
    window_end=datetime(2026, 10, 18, 10, 2, tzinfo=timezone.utc),
    features={...},                       # the 13 allowlisted features
    host_id="web-01",
    service_name="web-api",
)

result = detector.score_window(request)

# Branch on the status FIRST, before reading is_anomaly.
if result.status is InferenceStatus.SCORED:
    if result.is_anomaly:
        correlate(window_id=result.window_id, score=result.raw_score)
    else:
        ...          # scored and not flagged
else:
    record_unknown(result.window_id, result.reason_code.value, result.reason)
```

With the harness (the recommended pattern — the routing rule is centralised):

```python
from anomaly_detector.integration_harness import IntegrationHarness

harness = IntegrationHarness()             # loads the model once
outcome = harness.route_windows([request])  # routes each outcome

for scored in outcome.scored:               # only 'scored' results arrive here
    correlate(window_id=scored.window_id, score=scored.raw_score)

for record in harness.unscored.records:     # every other outcome, with its status
    record_unknown(record.window_id, record.status, record.reason)
```
