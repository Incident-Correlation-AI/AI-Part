# Network Detector — Integration Guide

**Status:** integration-ready in-process component · **Contract:** `1.1.0`
**Feature schema:** `network-features-v1` · **Deployed strategy:** `time`
**Applies to:** the network/service-log extension only

This document explains how to call the network detector, what it returns, how
the optional explanation adapter fits in, and what remains the caller's
responsibility. The normative request/result contract is still
[`ml_inference_contract.md`](ml_inference_contract.md); this document describes
the additive network extension.

> **One rule remains absolute:** `status != "scored"` is **unknown**, never
> “normal”. `is_anomaly` is `None` for every unscored result.

---

## 1. What was added, and what was not changed

### Additive changes (backward compatible)

| Item | Before | After | Compatibility |
|---|---|---|---|
| `CONTRACT_VERSION` | `1.0.0` | `1.1.0` | Minor bump; additive only. |
| Feature schema | `features-v1` (13 cols) | plus `network-features-v1` (39 cols) | Existing schema untouched. |
| `ReasonCode` | existing values | plus `no_events`, `insufficient_coverage`, `telemetry_quality` | Additive enum members. |
| `EventType` | existing values | plus `auth_success` | Additive enum member. |
| Result object | `InferenceResult` | still `InferenceResult` | No new result schema. |
| Score semantics | lower = more anomalous | unchanged | `is_anomaly = raw_score <= threshold`. |
| Status vocabulary | 5 statuses | unchanged | No new status values. |

### Unchanged

* The 13-column `features-v1` allowlist, its order and its artifacts.
* `load_detector()` and the existing `AnomalyDetector` behaviour.
* `route_result()`, `is_forwardable()` and the scored-only forwarding rule.
* The five status values and the rule that unscored is never normal.
* The canonical LAD event and incident schemas.

A caller that only knows `features-v1` keeps working unchanged. A caller that
enumerates `ReasonCode` should preserve unknown members rather than crash; the
same applies to `EventType`.

---

## 2. Public surface

```python
from anomaly_detector.network import (
    NetworkWindow,
    load_network_detector,
    explain_result,
    EvidenceBundle,
)
```

| Callable | Purpose | Raises? |
|---|---|---|
| `load_network_detector(model_path, manifest_path, rules, options)` | Load and cross-check the network artifact once. | `NetworkModelUnavailableError` if missing/unreadable/incompatible. |
| `detector.score_window(window) -> InferenceResult` | Score one `NetworkWindow`. | `TypeError` only for the wrong Python type. Data problems are statuses. |
| `detector.score_windows(windows) -> list[InferenceResult]` | Batch; one result per input, same order. | Same. Duplicate `sample_key` is rejected per row. |
| `detector.contract() -> dict` | Versions, 39 feature names, threshold, rules, decision rule. | No. |
| `NetworkWindow.to_request() -> FeatureWindowRequest` | Convert to the existing request object. | No. |
| `explain_result(result, provider=...) -> Explanation` | Optional evidence-grounded explanation. | No; failures are returned as an ungrounded/fallback result. |

Default artifact paths:

```
artifacts/network/model/model.joblib
artifacts/network/model/manifest.json
```

These are git-ignored and produced by the pipeline. They must exist before
`load_network_detector()` is called without explicit paths.

---

## 3. Building a window

The producer owns event ingestion, windowing and feature construction. The
detector consumes one already-aggregated `NetworkWindow`.

```python
from datetime import datetime, timezone

from anomaly_detector.network import NetworkWindow, load_network_detector
from anomaly_detector.schemas import WindowMetadata

detector = load_network_detector()          # load once, reuse

window = NetworkWindow(
    metadata=WindowMetadata(
        window_start=datetime(2026, 1, 10, 10, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 1, 10, 10, 2, tzinfo=timezone.utc),
        host_id="sim-web-01",
        service="web-api",
        window_size_minutes=1,
    ),
    features={
        # all 39 names from detector.expected_feature_columns
        "event_count": 128.0,
        "http_request_count": 110.0,
        "http_status_eligible_count": 110.0,
        "http_5xx_count": 44.0,
        "http_5xx_ratio": 0.4,
        # ... remaining columns, with None where unavailable
    },
    quality={"duplicate_event_id": 1},
)

result = detector.score_window(window)
```

Requirements:

* `window_start` and `window_end` are timezone-aware and `end > start`.
* Feature names must be exactly `detector.expected_feature_columns`; unknown
  names are rejected, missing names are reported.
* Values are numeric or `None`; booleans are rejected.
* `event_count` must be present. Zero means “no telemetry”, not “normal”.
* `sample_key` is `host|service|window_start.isoformat()` and must be unique
  within a batch.

For callers that already use `FeatureWindowRequest`, `window.to_request()`
produces the existing object. The network detector itself expects a
`NetworkWindow` because it needs quality counts and safe metadata.

---

## 4. Status and score semantics

| Status | Meaning | `is_anomaly` | Caller obligation |
|---|---|---|---|
| `scored` | Scored by the Isolation Forest. | `True` / `False` | May forward. `False` means scored and not flagged, not guaranteed normal. |
| `insufficient_data` | Empty, sparse, degraded or incomplete telemetry. | `None` | Record as unknown; do not treat as normal. |
| `invalid_input` | Contract/type/boundary violation. | `None` | Fix the producer; do not retry blindly. |
| `model_unavailable` | Artifact missing/unreadable/incompatible. | `None` | Operational incident. |
| `inference_error` | Scoring raised. | `None` | Log and escalate. |

Network-specific reason codes:

| Reason code | Status | Trigger |
|---|---|---|
| `no_events` | `insufficient_data` | `event_count` is zero or absent. |
| `insufficient_coverage` | `insufficient_data` | `event_count < min_window_events` (default 5). |
| `telemetry_quality` | `insufficient_data` | `invalid_event_count > event_count`. |
| `missing_features` | `insufficient_data` | Required feature absent or too much missingness. |
| `unknown_features` | `invalid_input` | Feature name outside the 39-column allowlist. |
| `non_numeric_feature` | `invalid_input` | String/boolean feature value. |
| `non_finite_feature` | `invalid_input` | `±inf`. |
| `naive_timestamp` / `invalid_window_boundary` | `invalid_input` | Bad window boundaries. |
| `duplicate_window_id` | `invalid_input` | Repeated `sample_key` in one batch. |
| `model_artifact_missing` / `model_artifact_invalid` / `model_manifest_mismatch` | `model_unavailable` | Artifact problems. |
| `inference_failed` | `inference_error` | Estimator raised. |

The deployed decision is **model-only**:

```python
assert result.is_anomaly == (result.raw_score <= result.threshold)
```

The rule baseline does **not** change that flag. It contributes severity and
evidence.

---

## 5. Result payload

The result is the existing `InferenceResult`. Network-specific data lives in
`result.extra["network"]`:

```json
{
  "contract_version": "1.1.0",
  "status": "scored",
  "reason_code": "ok",
  "window_id": "sim-web-01|web-api|2026-01-09T00:15:00+00:00",
  "model": {
    "feature_schema_version": "network-features-v1",
    "name": "sklearn.ensemble.IsolationForest"
  },
  "score": {
    "raw_score": -0.5379,
    "score_direction": "lower_is_more_anomalous",
    "is_probability": false
  },
  "decision": {
    "threshold": -0.5235479383749508,
    "is_anomaly": true
  },
  "extra": {
    "network": {
      "severity": "medium",
      "decision_source": "isolation_forest",
      "model_only_anomaly": false,
      "baseline": {
        "is_anomaly": true,
        "risk_score": 1.0,
        "severity": "medium",
        "rule_hits": [
          {
            "code": "http_5xx_ratio",
            "strength": "strong",
            "weight": 1.0,
            "observed": {"http_5xx_ratio": 0.4},
            "threshold": {"http_5xx_ratio": 0.05},
            "message": "HTTP 5xx ratio 0.4000 exceeds 0.0500"
          }
        ]
      },
      "telemetry_quality": {"duplicate_event_id": 1},
      "evidence_features": {"http_5xx_ratio": 0.4, "event_count": 128.0}
    }
  }
}
```

* **`severity`** is `none`, `low`, `medium`, `high` or `unknown` (unscored).
  It is derived from rule evidence, not from a probability.
* **`decision_source`** is `isolation_forest` when scored, `none` otherwise.
* **`model_only_anomaly`** is `True` when the model flags a window the rules do
  not. It is evidence of divergence, not a second decision.
* **`baseline`** is the independent transparent rule/MAD assessment.
* **`telemetry_quality`** carries the window's quality counts.
* **`evidence_features`** is a compact numeric subset for explanation; it
  contains no labels, scenario names, run IDs or raw identifiers.

For an unscored result, severity is `unknown`, `decision_source` is `none`,
`is_anomaly` and `raw_score` are `None`, and `baseline` is omitted.

---

## 6. Optional explanation adapter

`src/anomaly_detector/network/llm.py`

The adapter is **disabled by default**. It is not part of scoring and cannot
create, suppress or upgrade a detection.

```python
from anomaly_detector.network import explain_result, MockExplanationProvider

explanation = explain_result(result)          # deterministic offline mock
print(explanation.to_dict())
```

### 6.1 What it may explain

| Input | Behaviour |
|---|---|
| `scored` result | May produce a grounded explanation. |
| Non-`scored` result | Returns `explained=False` with an explicit reason. Never presents unknown as an anomaly. |
| Provider failure | Falls back to a deterministic evidence summary. |
| Invalid provider draft | Rejected, then fallback. `rejected_reason` records why. |

### 6.2 Grounding rules

A draft is accepted only when:

* it contains an explicit `asserts_anomaly` that matches the model decision;
* every citation is in the allowlisted evidence set;
* every numeric claim is supported by the evidence bundle;
* it contains no evaluation vocabulary (`scenario_name`, labels, run IDs,
  ground truth), no PII (emails, IPs, credentials) and no fabricated hosts;
* it is within the length bound.

Otherwise the adapter uses its own grounded fallback and sets
`fallback_used=True`. The output includes `grounded`, `provider`,
`fallback_used` and `rejected_reason` so the explanation can be audited.

### 6.3 Provider configuration

```yaml
llm:
  enabled: false
  provider: mock          # mock | http
  endpoint: ""            # required for http
  model: ""
  api_key_env: LAD_LLM_API_KEY
  timeout_seconds: 20.0
```

* The offline `MockExplanationProvider` makes no network call.
* `HttpExplanationProvider` reads the API key from the named environment
  variable at call time. The key is never written to config, manifests, logs or
  the explanation payload.
* Only a bounded `EvidenceBundle.to_prompt_dict()` is sent: window metadata,
  model/score/threshold, rule evidence, selected measurements, telemetry quality
  and constraints. No labels, scenario names, run IDs or raw event text.
* HTTP failures are recoverable and fall back; they are never promoted to a
  detection.

The adapter is a **new** component. No upstream LLM integration was available to
match (see §10).

---

## 7. Batch behaviour

```python
results = detector.score_windows(windows)
```

* One result per input, same order.
* One bad window does not fail the batch.
* Duplicate `sample_key` in a batch: the later occurrence gets
  `invalid_input` / `duplicate_window_id`; the first is scored normally.
* Downstream routing should use the existing `route_result()` or the
  integration harness, not a hand-written `if is_anomaly`.

---

## 8. Relation to the LAD backend and incident contract

### 8.1 Ingestion

The collector submits canonical LAD envelopes to:

```
POST /api/v1/events/batch
```

A `202` response means the batch was **queued**, not analysed. The ML component
does not own this endpoint and does not change it.

### 8.2 Stage boundary

```
collector → POST /api/v1/events/batch → stored canonical events
                                             │
                                             ▼
                              windowing + network features
                                             │
                                             ▼
                              NetworkDetector → InferenceResult
                                             │
                                             ▼
                              downstream correlation / incident
```

### 8.3 Incident fields

`LAD/contracts/schemas/incident.schema.json` requires `evidence` with
**`supporting_event_ids`** and **`contradicting_event_ids`**, plus separate
`hypotheses` and `diagnostic_checks` whose recommendations carry
`executes: false`.

The ML component supplies:

* `window_id`, `window_start`, `window_end`, `host_id`, `service_name`;
* `raw_score`, `threshold`, `is_anomaly`, `severity`;
* rule evidence and telemetry quality.

The downstream component supplies:

* evidence correlation and event-ID selection;
* incident grouping, affected entities and status;
* hypotheses, contradiction handling and narrative;
* diagnostic recommendations (non-executing);
* alert delivery and remediation policy.

The ML component never writes an incident record and never claims an attack.

---

## 9. Compatibility checklist

- [ ] Depend on `anomaly_detector.inference` and `anomaly_detector.network`
      only.
- [ ] Keep `features-v1` and `network-features-v1` separate; do not send a
      `features-v1` request to the network detector.
- [ ] `load_network_detector()` once at start-up; reuse it.
- [ ] Use `sample_key` as `window_id`.
- [ ] Branch on `status` first, `reason_code` second, `is_anomaly` last.
- [ ] Treat every non-`scored` result as unknown.
- [ ] Keep `raw_score` and `threshold` separate; never re-threshold.
- [ ] Never render `raw_score` as a probability or percentage.
- [ ] Report `no_events`, `insufficient_coverage` and `telemetry_quality`
      separately from producer bugs.
- [ ] Treat `severity` and rule hits as evidence, not diagnosis.
- [ ] Do not let an LLM explanation alter a decision.
- [ ] Preserve unknown enum members when serialising/deserialising.
- [ ] Do not expect this component to correlate evidence, build incidents or
      deliver alerts.

---

## 10. Verified tests and commands

Measured on 2026-10-26:

```bash
python -m pytest tests/test_network_features.py \
                 tests/test_network_lad_adapter.py \
                 tests/test_network_llm.py \
                 tests/test_network_inference.py -q
# 51 passed

python -m pytest -q
# 375 passed, 27 warnings
```

The network tests cover:

* schema shape, order and label-free feature names;
* zero-denominator ratios are `None`, not zero;
* causal rolling features and truncation invariance;
* time and group split integrity;
* LAD mapping, deduplication, out-of-order detection, PII removal and quality
  counts;
* reloaded model reproduces recorded scores and flags;
* empty, sparse, degraded, naive and malformed windows are unscored;
* JSON serialisability and explicit `None` decisions;
* LLM grounding, fabricated-number rejection, contradiction rejection,
  PII/vocabulary rejection, provider failure fallback and offline HTTP transport;
* LLM disabled by default and never serialises the API key.

Pipeline artifact reload is verified on every run:

```bash
python scripts/run_network_pipeline.py --config configs/network.yaml
```

The pipeline writes both strategies, reloads the persisted model through
`load_network_detector()`, and only then writes predictions.

---

## 11. Reproduction and artifact locations

```bash
# Full local run (no network):
python scripts/run_network_pipeline.py --config configs/network.yaml

# Simulator only:
python scripts/run_network_simulator.py --runs 3 --seed 42
```

| Artifact | Purpose |
|---|---|
| `artifacts/network/model/{model.joblib,manifest.json}` | Deployed `time` model. |
| `artifacts/network/experiments/time/…` | Time-strategy model, predictions, threshold, metrics. |
| `artifacts/network/experiments/group/…` | Group-strategy cross-check. |
| `artifacts/network/features.csv` | 39-column feature matrix. |
| `artifacts/network/metadata.csv` | Window keys and provenance. |
| `artifacts/network/labels.csv` | Evaluation-only labels. |
| `artifacts/network/lad_parse_summary.json` | Accepted/rejected/quality counts. |

All of these are git-ignored. Regenerate them locally; do not commit model
binaries or synthetic data.

---

## 12. Open items and blockers

| # | Item | Status |
|---|---|---|
| 1 | Exact upstream `kalmak889900/automated-log-anomaly-detector` code | Inaccessible (`clone` auth failure, API 404, `gh` cannot resolve). Documented in [`ml_network_reconnaissance.md`](ml_network_reconnaissance.md) §3. |
| 2 | A real upstream LLM/explanation contract | No such component exists in the accessible LAD repositories. The adapter is new and optional. |
| 3 | Real production replay | Not performed. All measured figures are synthetic. |
| 4 | Service-level integration | Not implemented here; this is an in-process boundary only. |
| 5 | Publication repository and branch | No Git remote is configured; publication requires an explicit decision. |
| 6 | Abel integration comment | Drafted but not posted; target identity/URL must be confirmed. |

---

## 13. References

* [`ml_network_reconnaissance.md`](ml_network_reconnaissance.md) — what existed
  and what could not be inspected.
* [`ml_network_features.md`](ml_network_features.md) — simulator, event mapping,
  39-column schema, leakage controls.
* [`ml_network_evaluation.md`](ml_network_evaluation.md) — measured results and
  limitations.
* [`ml_inference_contract.md`](ml_inference_contract.md) — the normative base
  contract.
* [`ML_HANDOFF.md`](ML_HANDOFF.md) — the caller's quick start.
* [`INTEGRATION_BOUNDARY.md`](INTEGRATION_BOUNDARY.md) — forwarding rule and
  ownership split.
