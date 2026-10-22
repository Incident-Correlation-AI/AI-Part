# ML Inference Contract — v1.0.0

**Status:** stable · **Applies to:** the anomaly-detection ML component only
**Contract version:** `1.0.0` · **Feature schema version:** `features-v1`

This document is the versioned, callable boundary between the ML
anomaly-detection component and the rest of the Automated Log Anomaly
Detector. The downstream owner of *evidence correlation, incident grouping
and root-cause explanation* calls this interface and nothing else.

It is an **in-process Python interface**. There is no HTTP API, no
dashboard, no deployment infrastructure, no alert delivery and no remediation.

---

## 1. What the component does and does not do

| The ML component **does** | The ML component **does not** do |
|---|---|
| Validate one already-aggregated feature window | Parse raw log text or raw events |
| Run the selected, already-trained model | Retrain, tune, or change thresholds |
| Return a versioned structured result with score + status | Retrieve or correlate evidence |
| Preserve safe window provenance (`window_id`, host, service, times) | Explain, summarise or narrate incidents |
| Distinguish "scored" from "could not score" | Deliver alerts or trigger remediation |
| Expose a documented continuous score | Present the score as a probability |
| Reject invalid / non-finite / incomplete input | Silently treat missing data as zero |

**Ownership.** The caller owns: window creation, evidence lookup,
deduplication across alerts, cross-window correlation, incident grouping,
explanation, root-cause hypotheses, alert delivery and remediation policy.

---

## 2. Prerequisites

The inference component needs the **selected model artifact** produced by
Milestone 3 and retained by Milestones 4–5:

```
artifacts/baseline/model/model.joblib     # fitted estimator + its contract
artifacts/baseline/model/manifest.json    # feature schema, threshold, versions
```

Regenerate them with the documented training workflow (**never during
inference**):

```bash
python scripts/generate_synthetic_data.py --config configs/default.yaml
python scripts/prepare_experiments.py
python scripts/run_baseline.py run --config configs/baseline.yaml
```

The selected model is the **Isolation Forest baseline** (`sklearn.ensemble.IsolationForest`,
`n_estimators=100`, `random_state=42`).

> **Security note.** `model.joblib` is a pickle-based artifact. Loading it
> executes code. Only load artifacts you produced and trust. The loader
> refuses paths outside the repository's `artifacts/` tree unless you
> explicitly opt in with `DetectorOptions(allow_external_model_path=True)`.

---

## 3. Input contract — one feature window

The input is **one already-aggregated feature window** from the existing
feature pipeline (M1 `FeatureWindow` → M2 split output). It is *not* raw
log text and *not* a raw event.

```python
from anomaly_detector.inference import FeatureWindowRequest

request = FeatureWindowRequest(
    window_id="web-01|web-api|2026-10-18T10:01:00+00:00",  # required
    window_start=datetime(2026, 10, 18, 10, 1, tzinfo=timezone.utc),  # required, aware
    window_end=datetime(2026, 10, 18, 10, 2, tzinfo=timezone.utc),    # required, aware
    features={...},                                 # required: all 13 allowlisted features
    host_id="web-01",                              # optional metadata
    service_name="web-api",                        # optional metadata
    feature_schema_version="features-v1",          # optional; checked when supplied
    extra={"tenant": "acme"},                      # optional, passed through untouched
)
```

### 3.1 Field reference

| Field | Type | Required | Nullable | Notes |
|---|---|---|---|---|
| `window_id` | `str` | ✅ | ❌ | Stable, traceable key. Canonical form is the M1/M2 `sample_key`: `host\|service\|window_start`. Never invented by the ML component. |
| `window_start` | `datetime` | ✅ | ❌ | Timezone-**aware** required. Naive datetimes are rejected. UTC recommended. |
| `window_end` | `datetime` | ✅ | ❌ | Timezone-aware, strictly after `window_start`. |
| `features` | `Mapping[str, float \| int \| None]` | ✅ | values may be `None` | Exactly the 13 allowlisted features (below). `None` means *missing*, not zero. |
| `host_id` | `str` | ❌ | ✅ | Metadata only. **Never** a model feature. |
| `service_name` | `str` | ❌ | ✅ | Metadata only. **Never** a model feature. |
| `feature_schema_version` | `str` | ❌ | ✅ | When supplied it must equal `features-v1`, else the request is rejected. |
| `extra` | `Mapping[str, Any]` | ❌ | ✅ | Caller metadata, echoed back untouched. Never used as a feature, never logged by default. |

### 3.2 Model features (explicit allowlist, order matters)

| # | Feature | Unit | Valid range |
|---|---|---|---|
| 1 | `event_count` | events | ≥ 0 integer |
| 2 | `error_count` | events | ≥ 0 integer |
| 3 | `error_ratio` | ratio | 0–1 |
| 4 | `http_request_count` | requests | ≥ 0 integer |
| 5 | `http_5xx_count` | responses | ≥ 0 integer |
| 6 | `http_5xx_ratio` | ratio | 0–1 |
| 7 | `auth_failure_count` | events | ≥ 0 integer |
| 8 | `restart_count` | events | ≥ 0 integer |
| 9 | `mean_latency_ms` | milliseconds | ≥ 0 |
| 10 | `p95_latency_ms` | milliseconds | ≥ 0 |
| 11 | `latency_sample_count` | samples | ≥ 0 integer (denominator) |
| 12 | `http_request_eligible_count` | requests | ≥ 0 integer (denominator) |
| 13 | `error_eligible_count` | events | ≥ 0 integer (denominator) |

**Never** accepted as model features: `is_anomaly`, `scenario_name`,
`intensity`, `description`, `host_id`, `service`, `sample_key`,
`window_start`, `window_end`, `split`, event IDs, filenames. These are
rejected as `unknown_features`.

The feature matrix is assembled **by name** from the detector's expected
column list. Dictionary iteration order is never used, so a request whose
mapping is in a different order produces the same score.

### 3.3 Behaviour matrix

| Situation | Status | Reason code |
|---|---|---|
| Valid window, scored | `scored` | `ok` |
| A required feature is absent from `features` | `insufficient_data` | `missing_features` |
| Feature present but `None` | `scored` (value imputed with the **training** median) | `ok` — see `data_quality.imputed_features` |
| Feature value is not numeric (e.g. `"many"`, `True`) | `invalid_input` | `non_numeric_feature` |
| Feature value is `±inf` | `invalid_input` | `non_finite_feature` |
| Feature name outside the allowlist | `invalid_input` | `unknown_features` (or ignored + reported if `allow_unknown_features=True`) |
| `window_id` missing/blank | `invalid_input` | `missing_window_id` |
| `window_start`/`window_end` missing | `invalid_input` | `missing_window_boundary` |
| Naive (timezone-less) timestamp | `invalid_input` | `naive_timestamp` |
| `window_end <= window_start` | `invalid_input` | `invalid_window_boundary` |
| `feature_schema_version` mismatch | `invalid_input` | `feature_schema_mismatch` |
| Duplicate `window_id` inside one batch | `invalid_input` | `duplicate_window_id` |
| Window older than `stale_after_seconds` (opt-in) | `insufficient_data` | `stale_window` |
| Artifact missing / unreadable / mismatched | `model_unavailable` | `model_artifact_missing`, `model_artifact_invalid`, `model_manifest_mismatch` |
| Estimator raised while scoring | `inference_error` | `inference_failed` |

**Caller-owned policies** (not enforced by the ML component, by design):
staleness/freshness windows, deduplication across alerts, and how long a
score stays valid. Use `DetectorOptions(stale_after_seconds=...)` if you want
the component to enforce staleness.

---

## 4. Output contract — one inference result

```json
{
  "contract_version": "1.0.0",
  "status": "scored",
  "reason_code": "ok",
  "reason": "scored with the selected model",
  "window_id": "web-01|web-api|2026-10-18T10:01:00+00:00",
  "window_start": "2026-10-18T10:01:00+00:00",
  "window_end": "2026-10-18T10:02:00+00:00",
  "host_id": "web-01",
  "service_name": "web-api",
  "model": {
    "name": "sklearn.ensemble.IsolationForest",
    "version": "c210b7a",
    "artifact_version": "1",
    "feature_schema_version": "features-v1",
    "threshold_config_version": "configs/baseline.yaml@c210b7a:-0.62026664075"
  },
  "score": {
    "raw_score": -0.4461774054674943,
    "score_direction": "lower_is_more_anomalous",
    "is_probability": false
  },
  "decision": {
    "threshold": -0.6202666407504878,
    "is_anomaly": false
  },
  "data_quality": {
    "imputed_features": [],
    "ignored_features": []
  },
  "extra": {},
  "duration_ms": 1.42
}
```

### 4.1 Field reference

| Field | Type | Notes |
|---|---|---|
| `contract_version` | `str` | Version of *this* contract. |
| `status` | `str` | One of the five statuses in §5. |
| `reason_code` | `str` | Stable machine-readable code — branch on this, not on `reason` text. |
| `reason` | `str` | Human-readable explanation. Not intended for programmatic branching. |
| `window_id` | `str \| null` | Copied from the input. `null` when the input had none — never invented. |
| `window_start` / `window_end` | `str \| null` | ISO-8601, copied from the input. |
| `host_id` / `service_name` | `str \| null` | Safe source metadata, copied from the input. |
| `model.name` | `str` | Estimator class path. |
| `model.version` | `str` | Code revision the artifact was trained from. |
| `model.artifact_version` | `str` | Serialized-artifact format version. |
| `model.feature_schema_version` | `str` | Feature schema the model expects. |
| `model.threshold_config_version` | `str` | Config path + revision + threshold value actually used. |
| `score.raw_score` | `float \| null` | The model's raw score. `null` when not scored. |
| `score.score_direction` | `str` | `lower_is_more_anomalous` for Isolation Forest. |
| `score.is_probability` | `bool` | Always `false`. The score is a **ranking statistic**, not a confidence. |
| `decision.threshold` | `float \| null` | The threshold actually applied. |
| `decision.is_anomaly` | `bool \| null` | `null` for every non-`scored` status. **A window that could not be scored is never reported as not-anomalous.** |
| `data_quality.imputed_features` | `list[str]` | Features that were missing and imputed with the training median. |
| `data_quality.ignored_features` | `list[str]` | Unknown features that were ignored (only when allowed). |
| `extra` | `dict` | Caller metadata, echoed back untouched. |
| `duration_ms` | `float \| null` | Optional timing; a convenience, not a guarantee. |

### 4.2 Score semantics (important)

* The score is `sklearn.ensemble.IsolationForest.score_samples`, which lives
  in `(-1, 0]`.
* **Lower = more anomalous.** The sign is *not* inverted and the score is
  *not* relabelled.
* The decision rule is **`is_anomaly = raw_score <= threshold`** (inclusive).
* The score is **not** a probability, not a confidence, and not a severity
  level. Do not render it as a percentage.

---

## 5. Status vocabulary — and what the caller must do

| Status | Meaning | `is_anomaly` | Caller obligation |
|---|---|---|---|
| `scored` | Valid input was scored. | `True` / `False` | You may forward it. `is_anomaly=False` means *scored and not flagged*, **not** "no anomaly" in any absolute sense. |
| `insufficient_data` | Input incomplete or unusable under approved policy. | `null` | Do **not** treat as normal. Record it, count it, surface it as "unknown". |
| `invalid_input` | Contract/type validation failed. | `null` | Do **not** retry blindly; fix the producer. Surface as a producer bug. |
| `model_unavailable` | Artifact missing/unreadable/incompatible. | `null` | Operational incident. Do **not** treat as normal — this is a deployment failure. |
| `inference_error` | Runtime could not produce a trustworthy score. | `null` | Do **not** treat as normal. Log and escalate. |

> The single most important rule: **`status != "scored"` is never
> equivalent to "not anomalous".** The component will never emit
> `is_anomaly: false` for a window it could not score.

---

## 6. How to call it

### 6.1 Load once, score many

```python
from anomaly_detector.inference import load_detector, InferenceStatus

detector = load_detector()            # loads artifact + manifest, validates both
detector.contract()                    # versions, features, threshold, score direction

result = detector.score_window(request)

if result.status is InferenceStatus.SCORED:
    if result.is_anomaly:
        lead = {"window_id": result.window_id, "score": result.raw_score}
    else:
        ...  # scored and not flagged
else:
    ...  # result.reason_code, result.status — never "normal"
```

`load_detector()` performs no training. The model and its imputation
medians are loaded once and reused; nothing is ever fitted on production
data.

### 6.2 Batch scoring

```python
results = detector.score_windows(requests)   # one result per input, same order
```

**Per-row failure semantics (chosen deliberately):** one bad row never
fails the batch. Every row gets its own result with an explicit status.
Duplicate `window_id` values inside a single batch are rejected for the
later occurrence (`duplicate_window_id`) so that every result maps to
exactly one window.

### 6.3 Building a request from the existing pipeline

```python
from anomaly_detector.inference import FeatureWindowRequest
from anomaly_detector.inference.contract import make_window_id

# From an M1 FeatureWindow:
request = FeatureWindowRequest.from_feature_window(feature_window)

# From the M2 split output (features.csv + metadata.csv):
request = FeatureWindowRequest(
    window_id=meta["sample_key"],
    window_start=datetime.fromisoformat(meta["window_start"]),
    window_end=datetime.fromisoformat(meta["window_end"]),
    features={c: row[c] for c in features.columns},
    host_id=meta["host_id"],
    service_name=meta["service"],
)
```

### 6.4 Options

```python
from anomaly_detector.inference import DetectorOptions

detector = load_detector(options=DetectorOptions(
    allow_unknown_features=False,     # default: unknown feature names are rejected
    allow_external_model_path=False,  # default: only repo artifacts/ may be loaded
    max_imputed_fraction=None,        # opt-in: abstain above this imputed fraction
    stale_after_seconds=None,         # opt-in: abstain on stale windows
    reference_time=None,              # reference point for the staleness check
    measure_duration=True,            # record duration_ms
))
```

---

## 7. Integration boundary (downstream)

The downstream component implements the neutral `DownstreamSink` protocol
and receives **only scored results**:

```python
from anomaly_detector.inference import EvidenceCorrelationAdapter, RecordingSink

sink = RecordingSink()                       # replace with the real sink
adapter = EvidenceCorrelationAdapter(detector, sink)
outcome = adapter.handle_windows(requests)

outcome.forwarded_count   # scored windows handed across the boundary
outcome.rejected_count    # windows that could not be scored (with reason codes)
```

`ScoredWindow` can only be constructed from a `scored` result — it raises
`ValueError` otherwise. That makes it structurally impossible to forward an
unscored window as a normal prediction.

---

## 8. Versioning rules

* `CONTRACT_VERSION` (`1.0.0`) — bump **minor** for additive,
  backward-compatible changes; bump **major** for removed/renamed fields or
  changed semantics.
* `FEATURE_SCHEMA_VERSION` (`features-v1`) — bump whenever the allowlist or
  the order of the allowlisted features changes. A detector built on
  `features-v1` rejects a request declaring a different schema version.
* `model.artifact_version` — the serialized-artifact format version. The
  loader refuses an artifact with an unexpected version.
* `model.version` — the code revision the model was trained from.
* `model.threshold_config_version` — config path + revision + the threshold
  value actually used, so a result can always be traced back to the exact
  decision rule that produced it.

---

## 9. Tests and smoke test

```bash
pytest tests/test_inference_contract.py -v      # contract, compatibility, failure modes
pytest tests/test_inference_integration.py -v   # integration boundary, end to end
pytest tests/ -q                                 # full M1–M6 suite (offline)

python scripts/smoke_test_inference.py           # single window
python scripts/smoke_test_inference.py --all     # whole split
```

The tests are offline and deterministic: no network, no AWS, no dashboard,
no training job, no large fixture.

---

## 10. Integration checklist for the downstream teammate

- [ ] Import from `anomaly_detector.inference` only — never from
      `anomaly_detector.baseline_model` or `...comparison`.
- [ ] Call `load_detector()` **once** at start-up; reuse the object.
- [ ] Build requests with `FeatureWindowRequest`; use the project
      `sample_key` as `window_id` so results join back to existing tables.
- [ ] Branch on `result.status` (and `result.reason_code`), **not** on
      `is_anomaly` alone.
- [ ] Treat every `status != "scored"` as **unknown**, never as normal.
- [ ] Forward only `status == "scored"` results to the correlation stage.
- [ ] Keep `raw_score` and `threshold` separate; do not re-threshold.
- [ ] Do not present `raw_score` as a probability or percentage.
- [ ] Count and report `insufficient_data` / `invalid_input` /
      `model_unavailable` separately — they are different operational
      problems.
- [ ] Do not retrain, retune or re-threshold from the inference component.
- [ ] Do not expect the ML component to retrieve evidence or explain
      anything; it returns a score and traceable window metadata only.

---

## 11. Limitations

* The model was trained and validated on **synthetic** data. A scored
  window is a *lead for investigation*, not proof of an attack and not
  evidence of real-world detection quality.
* The threshold is fixed by the M3 validation split and is **not**
  re-optimised per caller, per host or per service.
* Milestone 5 showed the detector is over-sensitive to legitimate
  distribution shifts and to missing telemetry. `data_quality` fields and
  the optional `max_imputed_fraction` guard exist so that the caller can
  react, but the default behaviour is unchanged.
* `duration_ms` is a wall-clock convenience measurement, not an SLA.
* The component is single-process and in-memory. It has no queueing,
  batching scheduler, caching or rate limiting — those belong to the
  service layer, not here.
