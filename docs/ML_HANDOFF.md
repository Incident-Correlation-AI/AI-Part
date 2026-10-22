# ML Component Handoff (Milestone 6)

**Audience:** the teammate who owns evidence correlation, incident
explanation and root-cause assistance.
**Scope:** how to call the ML anomaly-detection component — nothing else.

The full, normative contract lives in
[`ml_inference_contract.md`](ml_inference_contract.md). This document is the
short version: what to call, what you get back, and what you must handle.

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
    InferenceStatus,        # -> status enum
    ReasonCode,             # -> reason-code enum
    EvidenceCorrelationAdapter,  # optional: reference boundary adapter
)
```

| Callable | Purpose | Raises? |
|---|---|---|
| `load_detector(model_path, manifest_path, options) -> AnomalyDetector` | Load + validate the selected artifact once. | `ModelUnavailableError` when the artifact is missing/unreadable/incompatible. |
| `detector.score_window(request) -> InferenceResult` | Validate and score one window. | Never for data problems — they come back as a status. Raises `TypeError` only if you pass the wrong Python type. |
| `detector.score_windows(requests) -> list[InferenceResult]` | Batch. One result per input, same order. | Same rule; per-row failures are reported per row. |
| `detector.contract() -> dict` | Versions, feature list, threshold, score direction. | No. |

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

---

## 6. Integration checklist

- [ ] Depend on `anomaly_detector.inference` only.
- [ ] `load_detector()` once at start-up; reuse the object.
- [ ] Use the project `sample_key` as `window_id`.
- [ ] Branch on `status` first, `reason_code` second, `is_anomaly` last.
- [ ] Forward only `scored` results.
- [ ] Keep the raw score and threshold separate; never re-threshold.
- [ ] Never render `raw_score` as a probability or percentage.
- [ ] Count non-`scored` statuses separately — they are different problems.
- [ ] Never retrain or re-threshold from the inference component.
- [ ] Do not expect evidence, correlation or explanations from this call.

---

## 7. How to verify your integration

```bash
pytest tests/test_inference_contract.py -v      # contract + failure modes
pytest tests/test_inference_integration.py -v   # boundary + end to end
python scripts/smoke_test_inference.py           # single window
python scripts/smoke_test_inference.py --all     # whole test split
```

The smoke test prints the full structured result and the downstream
branching pattern.

---

## 8. What the component does not prove

The model was trained and validated on **synthetic** data. A flagged window
is a *lead for investigation*, not proof of an attack, and the score is a
ranking statistic rather than a confidence. Milestone 5 also found the
detector is over-sensitive to legitimate distribution shifts and to missing
telemetry — check `data_quality.imputed_features` before trusting a score
from a degraded window.

Full details, including the exact behaviour matrix and versioning rules:
[`ml_inference_contract.md`](ml_inference_contract.md).
