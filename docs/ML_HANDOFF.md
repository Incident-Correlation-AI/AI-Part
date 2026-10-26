# ML Component Handoff (Milestone 6, updated in Milestone 9)

**Audience:** the teammate who owns evidence correlation, incident
explanation and root-cause assistance.
**Scope:** how to call the ML anomaly-detection component — nothing else.

The full, normative contract lives in
[`ml_inference_contract.md`](ml_inference_contract.md). The **boundary**
between the two components lives in
[`INTEGRATION_BOUNDARY.md`](INTEGRATION_BOUNDARY.md). This document is the
short version: what to call, what you get back, and what you must handle.

> **Milestone 8 update.** The forwarding rule is now **centralised** in
> `route_result()`, every unscored status is handled through an explicit
> handler, sink failures are recorded instead of silently aborting a batch,
> and a local integration harness (`IntegrationHarness`) is available. See
> §8 below for what is still needed from you.

> **Milestone 9 update.** An **additive** network/service-log detector is
> available beside the existing `features-v1` interface. It uses the separate
> `network-features-v1` schema (39 columns), keeps the same `InferenceResult`
> and the same status/score semantics, and adds three reason codes
> (`no_events`, `insufficient_coverage`, `telemetry_quality`). Contract version
> is now `1.1.0`. If you only consume `features-v1`, nothing changes. See §9.

---

## 0. Read these first

| Order | File | Why |
|---|---|---|
| 1 | [`ml_inference_contract.md`](ml_inference_contract.md) | The canonical request/result contract. |
| 2 | [`INTEGRATION_BOUNDARY.md`](INTEGRATION_BOUNDARY.md) | The boundary, routing rule, ownership split. |
| 3 | **This file** | The quick start and checklist. |
| 4 | [`ml_network_integration.md`](ml_network_integration.md) | The additive network detector, if you consume network windows. |

---

## 1. The one thing to remember

> `status == "scored"` → you may forward the window.
> anything else → the ML component **does not know** whether the window is
> anomalous. Do not convert "unknown" into "normal".

---

## 2. Exact callable interface

```python
from anomaly_detector.inference import (
    load_detector,          # -> AnomalyDetector   (load once, reuse)
    FeatureWindowRequest,   # -> input object
    InferenceResult,         # -> output object
    InferenceStatus,        # -> status enum
    ReasonCode,             # -> reason-code enum
    route_result,           # -> the canonical forwarding rule
    is_forwardable,         # -> bool convenience
)
```

| Callable | Purpose | Raises? |
|---|---|---|
| `load_detector(model_path, manifest_path, options) -> AnomalyDetector` | Load + validate the selected artifact once. | `ModelUnavailableError` when the artifact is missing/unreadable/incompatible. |
| `detector.score_window(request) -> InferenceResult` | Validate and score one window. | Never for data problems — they come back as a status. Raises `TypeError` only if you pass the wrong Python type. |
| `detector.score_windows(requests) -> list[InferenceResult]` | Batch. One result per input, same order. | Same rule; per-row failures are reported per row. |
| `detector.contract() -> dict` | Versions, feature list, threshold, score direction. | No. |
| `route_result(result) -> RoutingDecision` | **The one place** that decides whether a result may be forwarded. | No. Never raises. |

### The recommended pattern (harness)

```python
from anomaly_detector.integration_harness import IntegrationHarness

harness = IntegrationHarness()             # loads the model once per session
outcome = harness.route_windows(requests)  # routes each outcome

for scored in outcome.scored:               # only 'scored' results arrive here
    correlate(window_id=scored.window_id, score=scored.raw_score)

for record in harness.unscored.records:     # every other outcome, with its status
    record_unknown(record.window_id, record.status, record.reason)
```

The harness applies the same canonical rule, so you cannot accidentally
implement a subtly different check.

---

## 3. Usage example

```python
from datetime import datetime, timezone

from anomaly_detector.inference import (
    InferenceStatus,
    load_detector,
)

detector = load_detector()          # once at start-up

request = {
    # build a FeatureWindowRequest from your feature pipeline output
}

result = detector.score_window(request)

if result.status is InferenceStatus.SCORED:
    if result.is_anomaly:
        # a lead for investigation — now go and correlate evidence
        correlate(window_id=result.window_id, score=result.raw_score)
    else:
        # scored, not flagged. Not a guarantee of normality.
        pass
else:
    # insufficient_data / invalid_input / model_unavailable / inference_error
    record_unknown(result.window_id, result.reason_code.value, result.reason)
```

Real, runnable version:

```python
from datetime import datetime, timezone
from anomaly_detector.inference import FeatureWindowRequest, InferenceStatus, load_detector

detector = load_detector()

request = FeatureWindowRequest(
    window_id="web-01|web-api|2026-10-18T10:01:00+00:00",
    window_start=datetime(2026, 10, 18, 10, 1, tzinfo=timezone.utc),
    window_end=datetime(2026, 10, 18, 10, 2, tzinfo=timezone.utc),
    features={
        "event_count": 28.0, "error_count": 0.0, "error_ratio": 0.0,
        "http_request_count": 28.0, "http_5xx_count": 0.0, "http_5xx_ratio": 0.0,
        "auth_failure_count": 0.0, "restart_count": 0.0,
        "mean_latency_ms": 50.64, "p95_latency_ms": 64.03,
        "latency_sample_count": 28.0,
        "http_request_eligible_count": 28.0, "error_eligible_count": 28.0,
    },
    host_id="web-01",
    service_name="web-api",
)

result = detector.score_window(request)
print(result.status.value, result.raw_score, result.is_anomaly)
# scored -0.4461774054674943 False
```

---

## 4. What you get back

| Field | Use it for |
|---|---|
| `status` | Your branch. `scored` vs everything else. |
| `reason_code` | Distinguishing *why* it failed: `missing_features`, `unknown_features`, `non_finite_feature`, `naive_timestamp`, `invalid_window_boundary`, `feature_schema_mismatch`, `duplicate_window_id`, `stale_window`, `model_artifact_missing`, `inference_failed`. |
| `raw_score` | Ranking the scored windows. **Lower = more anomalous.** Not a probability. |
| `threshold` | The threshold that produced `is_anomaly`. Do not re-threshold. |
| `is_anomaly` | `True`/`False` only when `status == "scored"`; otherwise `None`. |
| `window_id`, `window_start`, `window_end`, `host_id`, `service_name` | Joining back to your evidence tables. |
| `model.*` | Provenance: which model, which version, which feature schema, which threshold config. |
| `data_quality.imputed_features` | Features that were missing and imputed with the training median. Useful for trusting (or not) a score. |
| `duration_ms` | Optional, informational. |

---

## 5. Failure-handling obligations

| Status | What you must do | What you must **not** do |
|---|---|---|
| `scored`, `is_anomaly=False` | Continue; optionally record as "no lead". | Do not report as "no anomaly in the system". |
| `scored`, `is_anomaly=True` | Forward to correlation. | Do not treat as an incident or a diagnosis. |
| `insufficient_data` | Count it, surface it as unknown, fix the producer. | Do not treat as normal. |
| `invalid_input` | Fix the producer (type/shape/boundary bug). | Do not retry blindly. |
| `model_unavailable` | Raise an operational alert; the deployment is broken. | Do not fall back to "normal". |
| `inference_error` | Log and escalate. | Do not swallow it or retry in a tight loop. |
| `unknown` (malformed result) | Treat as a producer/contract bug. | Do not treat as normal. |

### Sink failures (your component is the sink)

If *your* component raises while accepting a scored window:

* in a batch, the harness records it in `outcome.sink_failures`, the window is
  **not** in the forwarded list, and the rest of the batch continues;
* for a single window, `handle_window()` raises `SinkDeliveryError`.

Either way it is never reported as a successful delivery.

---

## 6. Integration checklist

- [ ] Depend on `anomaly_detector.inference` (and, for network windows,
      `anomaly_detector.network`) only.
- [ ] Keep `features-v1` and `network-features-v1` separate; do not send one
      schema to the other detector.
- [ ] `load_detector()` once at start-up; reuse the object.
- [ ] Use the project `sample_key` as `window_id`.
- [ ] Branch on `status` first, `reason_code` second, `is_anomaly` last.
- [ ] Forward only `scored` results — use `route_result()` or the harness, do
      not re-implement the check.
- [ ] Keep the raw score and threshold separate; never re-threshold.
- [ ] Never render `raw_score` as a probability or percentage.
- [ ] Count non-`scored` statuses separately — they are different problems.
- [ ] Never retrain or re-threshold from the inference component.
- [ ] Do not expect evidence, correlation or explanations from this call.
- [ ] Treat a sink failure as a failure — it is recorded, not swallowed.

---

## 7. Verified commands

Every command below was run and passed:

```bash
python -m pytest tests/test_integration_boundary.py -v   # 35 passed
python -m pytest tests/ -q                               # 375 passed
python scripts/run_integration_rehearsal.py              # local rehearsal
python scripts/smoke_test_inference.py                  # M6 smoke test
python scripts/run_readiness_audit.py                   # M7 readiness audit
python scripts/run_network_pipeline.py --config configs/network.yaml  # M9
```

If the package is not installed, prefix with `PYTHONPATH=src`.

---

## 8. What is still needed from you (Milestone 8)

The harness is deliberately generic so that it does not guess at your
architecture. Before the real adapter is written, these need agreeing:

| # | Question | Why it matters |
|---|---|---|
| 1 | **Accepted input** — is the whole `InferenceResult` acceptable, or a subset / a different shape? | Determines what the adapter must map. |
| 2 | **Desired output** — what does your component return, and does it need anything back? | The current `DownstreamSink.handle_scored_window` returns `None`. |
| 3 | **Sync or async** — may the call block, or must the ML component hand off and return? | Decides whether the sink is called inline or via a queue. |
| 4 | **Error handling** — what should happen when your component fails? Retry? Drop? Escalate? | The harness currently records the failure and continues the batch. |
| 5 | **Duplicate semantics** — is one result per `window_id` enough, or is cross-batch deduplication required? | Today only within-batch duplicates are rejected. |
| 6 | **Required provenance** — is the set in [`INTEGRATION_BOUNDARY.md` §8](INTEGRATION_BOUNDARY.md) sufficient? | Missing provenance would need a contract version bump. |
| 7 | **Persistence and retries** — who stores the result, who retries, who owns the queue? | Currently nobody; the harness is in-memory only. |
| 8 | **Volume and latency expectations** — is a batch expected, and what latency budget applies? | Affects whether batching is needed. |

Until these are answered, the harness stays neutral and the real downstream
interface remains **explicitly pending**.

---

## 9. Milestone 9 — network / service-log detector

The network extension is a **separate, additive** entry point. It does not
change the `features-v1` call path described above.

```python
from anomaly_detector.network import NetworkWindow, load_network_detector

detector = load_network_detector()          # artifacts/network/model/*
result = detector.score_window(window)      # same InferenceResult
```

| Item | Value |
|---|---|
| Feature schema | `network-features-v1` (39 columns) |
| Feature allowlist | `detector.expected_feature_columns` |
| Artifact | `artifacts/network/model/{model.joblib,manifest.json}` |
| Decision | Isolation Forest, `is_anomaly = raw_score <= threshold` |
| Rule baseline | Evidence/severity only; never changes `is_anomaly` |
| Explanation | Optional, disabled by default, never changes a decision |

### What you must handle differently

| Condition | Status | Reason |
|---|---|---|
| `event_count` is zero or absent | `insufficient_data` | `no_events` |
| `event_count < 5` (configurable) | `insufficient_data` | `insufficient_coverage` |
| Quality records exceed accepted events | `insufficient_data` | `telemetry_quality` |
| `features-v1`-shaped input | `invalid_input` | `unknown_features` / `missing_features` |

The same absolute rule still applies: every non-`scored` result is **unknown**,
never normal. An empty window must not become a "no anomaly" event.

Network results put extra context in `result.extra["network"]`: `severity`,
`decision_source`, `model_only_anomaly`, the independent rule `baseline`, the
explicit `telemetry_quality` counts and a small `evidence_features` map. Treat
these as evidence for correlation, not as a diagnosis.

Full details: [`ml_network_integration.md`](ml_network_integration.md).
Schema, simulator and leakage controls:
[`ml_network_features.md`](ml_network_features.md).
Measured results and limitations:
[`ml_network_evaluation.md`](ml_network_evaluation.md).

---

## 10. What the component does not prove

The model was trained and validated on **synthetic** data. A flagged window
is a *lead for investigation*, not proof of an attack, and the score is a
ranking statistic rather than a confidence. Milestone 5 also found the
detector is over-sensitive to legitimate distribution shifts and to missing
telemetry — check `data_quality.imputed_features` before trusting a score
from a degraded window.

Milestone 7's readiness verdict stands: **`controlled demo only`**.

The M9 network detector is in the same position. Its figures come from a
deterministic simulator that repeats one scenario schedule; they validate the
pipeline, not production generalisation. The benign high-traffic scenario has a
0.25 false-positive rate in the time test split, so volume-driven alerts should
be expected and must be tuned before operational use.

Full details, including the exact behaviour matrix and versioning rules:
[`ml_inference_contract.md`](ml_inference_contract.md).
The boundary, ownership split and exception behaviour:
[`INTEGRATION_BOUNDARY.md`](INTEGRATION_BOUNDARY.md).
