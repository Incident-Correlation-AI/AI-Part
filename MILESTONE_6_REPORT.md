# Milestone 6 — Stable ML Inference Contract and Integration Boundary

**Date:** 2026-10-22 · **Scope:** ML/anomaly-detection workstream only
**Deliverable:** a stable, documented, callable ML inference interface for the
downstream evidence-correlation / explanation teammate.

---

## 1. Repository findings — were M1–M5 reproducible?

**Yes.** Everything was inspected and reproduced before any new code was written.

| Area inspected | Finding | Reproducible? |
|---|---|---|
| Repo tree, README, configs | `configs/{default,data_readiness,baseline,comparison,robustness}.yaml`, `docs/`, `scripts/`, `src/anomaly_detector/` | ✅ |
| M1 event/feature pipeline | `schemas.py`, `validation.py`, `preprocessing.py`, `features.py`, `synthetic_data.py` | ✅ |
| M2 audit/leakage/splits | `audit.py`, `leakage.py`, `splits.py`, `training_ready.py`; splits present under `data/processed/splits/random/{train,validation,test}` | ✅ |
| M3 model + artifacts | `artifacts/baseline/model/model.joblib` + `manifest.json` load in a clean process; manifest ↔ artifact feature list matches | ✅ |
| M4 selection decision | `MILESTONE_4_REPORT.md` → **retain `m3_isolation_forest_reference`** | ✅ (see discrepancy below) |
| M5 robustness results | `artifacts/robustness/run_*/reports/robustness_report.md`; 5 scenarios, failure analysis present | ✅ |
| Existing test suite | `pytest tests/ -q` → **208 passed**, 20 warnings, no pre-existing failures | ✅ |

### ⚠️ Discrepancy found in M4 (documented, not silently resolved)

The M4 **machine-generated** report
(`artifacts/comparison/run_*/reports/comparison_report.json`) records
`"recommended": "lof_novelty"` — the auto-picker chose LOF because it scored
perfectly on the synthetic test split. The M4 **human-written** report
(`MILESTONE_4_REPORT.md`, §Recommendation) explicitly recommends
**retaining the M3 Isolation Forest**, reasoning that a perfect score on 46
synthetic windows with clean separation is likely over-optimistic.

**Decision taken:** M6 binds to the **M3 Isolation Forest** — the documented
human decision, and the model M5 actually validated. The M4 code was **not**
modified (it belongs to a completed milestone). This discrepancy is listed
under §10 as a known issue for the team to close.

### Selected model (as bound by the interface)

| Property | Value |
|---|---|
| Estimator | `sklearn.ensemble.IsolationForest` |
| Parameters | `n_estimators=100`, `max_samples=256`, `contamination="auto"`, `random_state=42`, `n_jobs=2` |
| Score | `IsolationForest.score_samples`, **lower = more anomalous** |
| Decision rule | `is_anomaly = raw_score <= threshold` (inclusive) |
| Threshold | `-0.6202666407504878` (method `validation_f1`, source: validation split) |
| Features | 13, explicit allowlist (labels/IDs/timestamps/scenario excluded) |
| Missing policy | `median_impute` with **training-split** medians (stored in the artifact) |
| Non-finite | `±inf` rejected; `NaN` handled by the missing-value policy |
| Training policy | `normal_only`, training rows only |
| Artifact | `artifacts/baseline/model/model.joblib` (artifact format version `1`) |
| Model version | `c210b7a` (code revision the artifact was trained from) |

---

## 2. Interface contract and key design decisions

Full normative document: **`docs/ml_inference_contract.md`** (contract v1.0.0).
Teammate-facing summary: **`docs/ML_HANDOFF.md`**.

### 2.1 Versions

| Version | Value | Bump when |
|---|---|---|
| `CONTRACT_VERSION` | `1.0.0` | additive → minor; removed/renamed fields or changed semantics → major |
| `FEATURE_SCHEMA_VERSION` | `features-v1` | the allowlist or its order changes |
| `model.artifact_version` | `1` | the serialized artifact format changes |
| `model.version` | `c210b7a` | the model is retrained (records the training revision) |
| `model.threshold_config_version` | `configs/baseline.yaml@c210b7a:-0.62026664075` | the threshold or its config changes |

### 2.2 Input — one already-aggregated feature window

`FeatureWindowRequest`: `window_id` (required, non-empty, canonical form =
the M1/M2 `sample_key` `host|service|window_start`), `window_start` /
`window_end` (required, **timezone-aware**, `end > start`), `features`
(required mapping of the 13 allowlisted features), `host_id` /
`service_name` / `feature_schema_version` / `extra` (optional metadata).

Key decisions:
- **Reused the M1 schema.** `FeatureWindowRequest.from_feature_window(fw)`
  converts an existing M1 `FeatureWindow`; no competing window schema was
  invented.
- **Feature order enforced by name.** The matrix is assembled by indexing the
  detector's expected column list — never by dictionary iteration order. A
  request whose mapping is in a different order yields an identical score.
- **Unknown feature names are rejected** by default (`unknown_features`).
  They can be allowed explicitly, in which case they are reported in
  `ignored_features` and provably do not change the score.
- **Labels and identifiers can never become features.** `is_anomaly`,
  `scenario_name`, `host_id`, `service`, `sample_key`, `window_start`,
  `window_end` are not in the allowlist and are rejected on sight.

### 2.3 Output — one versioned `InferenceResult`

Carries: contract version, status, reason code + human reason, `window_id` /
`window_start` / `window_end` / `host_id` / `service_name` (safe provenance),
model name/version/artifact version/feature-schema version/threshold-config
version, `raw_score` + `score_direction` + `is_probability: false`,
`threshold` + `is_anomaly`, `data_quality.{imputed_features,ignored_features}`,
`extra`, and optional `duration_ms`.

Key decisions:
- **The continuous score stays separate from the decision.** The score keeps
  the model's own sign and direction — it is *not* inverted, relabelled, or
  presented as a probability.
- **Provenance is copied, never invented.** A missing `window_id` yields
  `window_id: null` plus an `invalid_input` status — no fake identifier.
- **No sensitive content.** The result carries no raw logs, credentials or
  request bodies; caller `extra` is echoed back but never logged by default.

### 2.4 Status vocabulary (5 statuses)

| Status | `is_anomaly` | Meaning |
|---|---|---|
| `scored` | `True`/`False` | valid input was scored |
| `insufficient_data` | `null` | incomplete/unusable under approved policy |
| `invalid_input` | `null` | contract/type validation failed |
| `model_unavailable` | `null` | artifact missing, unreadable or incompatible |
| `inference_error` | `null` | runtime could not produce a trustworthy score |

**The central invariant:** an unscored window is *never* reported as normal.
`is_anomaly` is `None` for every non-`scored` status.

### 2.5 Other key decisions

- **Per-row batch failure semantics.** One bad row never fails the batch;
  every row gets its own result. Duplicate `window_id`s inside a batch are
  rejected for the later occurrence so every result maps to exactly one window.
- **NaN vs infinity.** `NaN` in a feature means *missing*: it is imputed with
  the **training** median (exactly the trained model's policy) and reported in
  `imputed_features`. `±inf` is rejected outright. A missing value is never
  silently zero-filled, and the imputed score provably differs from a
  zero-filled one.
- **Trusted artifacts only.** The loader refuses model paths outside the
  repository's `artifacts/` tree unless `DetectorOptions(allow_external_model_path=True)`
  is passed. `model.joblib` is pickle-based and executes code on load.
- **Optional policies, off by default.** `stale_after_seconds` and
  `max_imputed_fraction` let a caller enforce abstention; both default to the
  trained model's own behaviour, so the default path changes nothing.
- **Caller-owned temporal policy.** Staleness, dedup across alerts and score
  validity are documented as the caller's responsibility, not the ML
  component's.
- **No web framework, no network imports.** A test asserts the inference
  package contains no `requests`/`urllib`/`socket`/`flask`/`fastapi`/`uvicorn`
  imports.

---

## 3. Files changed / created

### Created

| File | Purpose |
|---|---|
| `src/anomaly_detector/inference/__init__.py` | Public API surface (re-exports) |
| `src/anomaly_detector/inference/contract.py` | Contract v1.0.0: `FeatureWindowRequest`, `InferenceResult`, `InferenceStatus`, `ReasonCode`, version constants, `make_window_id` |
| `src/anomaly_detector/inference/detector.py` | `load_detector()`, `AnomalyDetector.score_window()` / `.score_windows()`, `DetectorOptions`, `ModelUnavailableError` |
| `src/anomaly_detector/inference/adapter.py` | `DownstreamSink` protocol, `EvidenceCorrelationAdapter`, `ScoredWindow`, `RejectedWindow`, `AdapterOutcome`, `RecordingSink` |
| `scripts/smoke_test_inference.py` | CLI smoke test / example (single window or whole split) |
| `tests/test_inference_contract.py` | 28 contract, compatibility and failure-mode tests |
| `tests/test_inference_integration.py` | 11 integration-boundary and end-to-end tests |
| `docs/ml_inference_contract.md` | Normative versioned contract (input/output, behaviour matrix, versioning rules) |
| `docs/ML_HANDOFF.md` | Teammate handoff + integration checklist |
| `MILESTONE_6_REPORT.md` | This report |

### Modified

| File | Change |
|---|---|
| `src/anomaly_detector/baseline_model.py` | `BaselineDetector.artifact_version` attribute is now set on load (additive; no behaviour change) |
| `README.md` | M6 overview, architecture, module table, commands, outputs, tests, limitations, boundary, repo structure; removed a duplicated "Outputs" table left over from M4 |

### Deliberately **not** modified

`artifacts/baseline/**`, `artifacts/comparison/**`, `artifacts/robustness/**`,
`data/**`, and all M1–M5 source modules other than the additive
`artifact_version` attribute. No M1–M5 artifact was overwritten.

---

## 4. Exact test and smoke-test commands run

```bash
# baseline before any change
PYTHONPATH=src python -m pytest tests/ -q
#   → 208 passed, 20 warnings

# M6 test files
PYTHONPATH=src python -m pytest tests/test_inference_contract.py -q
#   → 28 passed
PYTHONPATH=src python -m pytest tests/test_inference_integration.py -q
#   → 11 passed

# full suite (M1–M6 regression)
PYTHONPATH=src python -m pytest tests/ -q
#   → 247 passed, 20 warnings

# smoke tests
PYTHONPATH=src python scripts/smoke_test_inference.py
#   → prints the contract header + one full structured result
PYTHONPATH=src python scripts/smoke_test_inference.py --all
#   → scores the whole test split (46 windows), all scored
```

**Totals: 247 passed, 0 failed** (208 pre-existing + 39 new). No pre-existing
test was weakened, skipped or deleted. No test requires network access, AWS, a
running dashboard, or a training job.

One test was **fixed during development** (not weakened): an assertion in
`test_score_is_not_represented_as_a_probability` incorrectly assumed the
Isolation Forest score lives in `[0, 1]`. The actual score is in `(-1, 0]`;
the test was corrected to assert the real, documented range and direction.

---

## 5. Per-scenario / per-case results

The M6 milestone validates a **contract**, so the "scenarios" are input
cases rather than data scenarios. Verified behaviour (all 39 new tests pass):

| Case | Result |
|---|---|
| Valid window | `scored`, score `-0.446…` for the first test-split window, `is_anomaly=false` |
| Anomalous window (400+ events, 300 errors, 2.4 s latency) | `scored`, `is_anomaly=true`, score strictly **lower** than the normal window |
| Feature mapping in a different order | identical score (order is enforced by name) |
| Unknown feature name | `invalid_input` / `unknown_features`, `is_anomaly=null` |
| Unknown feature allowed | `scored`, reported in `ignored_features`, score unchanged |
| Missing required feature | `insufficient_data` / `missing_features`, `is_anomaly=null` |
| Non-numeric feature (`"many"`, `True`) | `invalid_input` / `non_numeric_feature` |
| `+inf` feature | `invalid_input` / `non_finite_feature`, no score |
| `NaN` feature (i.e. `None`) | `scored`, imputed with the **training** median, listed in `imputed_features`; score ≠ zero-filled score |
| `feature_schema_version` mismatch | `invalid_input` / `feature_schema_mismatch` |
| Naive timestamps | `invalid_input` / `naive_timestamp` |
| `window_end == window_start` | `invalid_input` / `invalid_window_boundary` |
| Blank `window_id` | `invalid_input` / `missing_window_id`, `window_id=null` (not invented) |
| Label/scenario/host/service smuggled in as a feature | `invalid_input` / `unknown_features` |
| Duplicate `window_id` in a batch | later row → `invalid_input` / `duplicate_window_id` |
| One bad row in a batch | batch continues; bad row `insufficient_data`, good row `scored` |
| Missing artifact | `ModelUnavailableError` (explicit, at load time) |
| Missing manifest | `ModelUnavailableError` |
| Manifest feature list ≠ artifact feature list | `ModelUnavailableError` ("mismatched schema") |
| External model path without opt-in | `ModelUnavailableError` (refuses untrusted path) |
| Repeated identical calls | byte-identical score and decision (determinism) |
| Whole test split (46 windows) | all `scored`, all with a decision |
| Staleness policy (opt-in) | `insufficient_data` / `stale_window` |
| Max-imputed-fraction policy (opt-in) | `insufficient_data` |
| Downstream adapter | forwards only `scored` results; `ScoredWindow.from_result` raises on a non-scored result |
| Smoke-test script in a subprocess | exit code 0, prints the contract header and the result |

---

## 6. False-alert / missed-detection analysis

**Not applicable in the M6 sense** — M6 performs no detection experiment and
must not tune the model. The relevant failure analysis is **contract-level**:
which input conditions produce a *wrong* answer versus an *explicit* answer.

| Failure mode | How the contract prevents it |
|---|---|
| A bad window silently treated as normal | `is_anomaly` is `null` for every non-`scored` status; `ScoredWindow` cannot even be built from a failed result |
| A bad window silently treated as anomalous | same invariant; there is no path that sets `is_anomaly=True` without a real score |
| One bad row poisoning a whole batch | per-row results; a bad row never fails the batch |
| A score silently reinterpreted as a probability | `score.is_probability` is always `false`; the score keeps the model's own sign and direction |
| Features silently reordered | matrix assembled by name from the model's allowlist; a reordered mapping gives an identical score |
| A missing value silently zero-filled | missing features are imputed with the **training** median and reported; the zero-filled score provably differs |
| A stale/duplicate window double-counted | duplicate `window_id` rejected in-batch; optional staleness policy available |
| Labels leaking into scoring | label/identifier names are outside the allowlist and are rejected |
| A broken deployment looking like a quiet system | `model_unavailable` is a distinct status that the caller must escalate |

The M5 findings (over-sensitivity to legitimate distribution shifts and to
missing telemetry) are **not** re-litigated here; the interface simply makes
them *visible* to the caller via `data_quality.imputed_features`.

---

## 7. Drift results

**Not applicable / deferred.** M6 introduces no new data and no new model, so
there is nothing to drift-check. The M5 drift analysis remains deferred for the
reasons recorded in `MILESTONE_5_REPORT.md` (no suitable real reference
distribution exists in this repository; the synthetic generators are not a
proxy for production drift).

The interface does make drift *detectable* by the downstream layer, because
every scored result carries the model version, feature-schema version and
threshold-config version, so scores produced before and after a model change
are comparable only when those versions match.

---

## 8. Runtime / compute observations

- The whole test split (46 windows) is scored in **one batch** in well under a
  second after the model is loaded (the smoke test with `--all` completes
  comfortably).
- The dominant cost is the **one-off** model load (scikit-learn estimator
  deserialization). `load_detector()` is called once and the object is reused,
  so per-window cost after loading is a single `score_samples` call over a
  1×13 matrix.
- `duration_ms` is recorded per result for convenience. The first call in a
  process includes lazy scikit-learn warm-up (observed ≈ 66 ms); subsequent
  calls are sub-millisecond to low-millisecond. This is an observation, not an
  SLA.
- Memory: the loaded artifact is small (100 trees, 13 features); the
  component holds one estimator in memory.
- No GPU, no network, no cloud service, no paid dependency. The package adds
  **no new runtime dependency** — it uses the standard library plus the
  already-declared `scikit-learn` / `pandas` / `numpy` / `joblib`.

---

## 9. Paths to reports, metrics, predictions, fixtures and tests

| Kind | Path |
|---|---|
| Normative contract | `docs/ml_inference_contract.md` |
| Handoff + checklist | `docs/ML_HANDOFF.md` |
| This report | `MILESTONE_6_REPORT.md` |
| Public API | `src/anomaly_detector/inference/__init__.py` |
| Contract types | `src/anomaly_detector/inference/contract.py` |
| Detector | `src/anomaly_detector/inference/detector.py` |
| Integration boundary | `src/anomaly_detector/inference/adapter.py` |
| Smoke test | `scripts/smoke_test_inference.py` |
| Contract tests | `tests/test_inference_contract.py` |
| Integration tests | `tests/test_inference_integration.py` |
| Model artifact consumed | `artifacts/baseline/model/model.joblib` |
| Model manifest consumed | `artifacts/baseline/model/manifest.json` |
| Predecessor reports | `MILESTONE_1_REPORT.md` … `MILESTONE_5_REPORT.md` |

**M6 writes no artifacts of its own** — it is an in-process interface over the
existing M3 model artifact. No M1–M5 artifact was overwritten.

---

## 10. Known issues, limitations and unverified behaviour

### Known issues (pre-existing, not introduced by M6)

1. **M4 recommendation discrepancy** — the auto-generated comparison report
   says `lof_novelty`, the written M4 report says retain Isolation Forest.
   M6 binds to Isolation Forest. *The M4 code should be corrected to encode the
   documented decision rather than re-deriving it from the test metrics.*
2. **`M5` drift analysis deferred** — no suitable reference distribution.
3. **M5 false-alert findings** — over-sensitivity to distribution shifts
   (100 % flag rate on a benign shift) and to missing telemetry (15 %).
4. **`max_samples` warning** — scikit-learn warns that `max_samples=256`
   exceeds the 110 training rows. Harmless (it clamps), but noisy.
5. **Synthetic-data caveat** — every metric in M1–M5 comes from synthetic
   data and does not establish real-world detection quality.

### Limitations of the M6 interface itself

- **Single-process, in-memory.** No queueing, caching, batching scheduler,
  rate limiting or concurrency control. Those belong to the service layer.
- **No schema migration path.** A feature-schema change requires a new
  detector artifact and a `FEATURE_SCHEMA_VERSION` bump; there is no
  adapter that maps an old window onto a new schema.
- **`duration_ms` is a convenience**, not a guarantee, and the first call in a
  process includes warm-up.
- **The staleness and max-imputed-fraction policies are opt-in** and off by
  default, so a caller that ignores them gets the trained model's behaviour
  exactly.
- **`extra` is echoed back verbatim.** The component does not redact it; the
  caller is responsible for not putting secrets in it.

### Unverified behaviour (stated plainly)

- **No real production data** has been pushed through the interface. All
  verification used synthetic data and the repository's own split output.
- **The downstream teammate's actual interface is unknown.** The
  `DownstreamSink` protocol and the reference adapter are a *neutral,
  documented* boundary, not an implementation of another workstream's
  architecture.
- **Load/throughput behaviour under concurrency** was not measured.
- **Behaviour on a model artifact trained with a different feature order** was
  not exercised against a real artifact — only against a deliberately
  corrupted manifest in a test.

---

## 11. Recommended next steps (NOT part of the completed work)

1. **Close the M4 discrepancy** — encode the documented "retain Isolation
   Forest" decision in the M4 comparison code so the artifact and the report
   agree, or re-run M4 with an explicit, pre-declared preference for the
   simpler model.
2. **Add a schema-migration adapter** so an old `features-v1` window can be
   mapped onto a future schema with an explicit, tested conversion.
3. **Calibrate abstention defaults** using the M5 findings — e.g. ship
   `max_imputed_fraction` with a validated value instead of leaving it off.
4. **Re-validate the threshold on a more realistic dataset** before the
   interface is wired to anything that acts on the decisions.
5. **Add a lightweight drift monitor** that compares the feature distribution
   behind the scored windows against the training reference, and reports
   changes without touching the model or the threshold.
6. **Agree the SLA** with the downstream team (latency budget, batch size,
   availability expectations) — the interface currently makes no promises.
7. **Do not** add an HTTP layer, a queue, an LLM or an explanation path
   without a separate, explicit milestone.

---

## 12. Exact callable interface and usage example

```python
from datetime import datetime, timezone

from anomaly_detector.inference import (
    InferenceStatus,
    load_detector,
)
from anomaly_detector.inference.contract import FeatureWindowRequest

# 1. load once at start-up
detector = load_detector()      # artifacts/baseline/model/{model.joblib,manifest.json}

# 2. build one request (feature window from the existing pipeline)
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

# 3. score
result = detector.score_window(request)
# result.status        -> InferenceStatus.SCORED
# result.raw_score     -> -0.4461774054674943   (lower = more anomalous)
# result.is_anomaly    -> False
# result.threshold     -> -0.6202666407504878

# 4. branch on the status, never on is_anomaly alone
if result.status is InferenceStatus.SCORED:
    if result.is_anomaly:
        correlate(result.window_id, result.raw_score)   # downstream boundary
else:
    record_unknown(result.window_id, result.reason_code.value)
```

Batch form: `detector.score_windows(requests)` → one result per input, same
order, per-row failure semantics.

---

## 13. Integration steps the downstream teammate must perform

1. Install the repo dependencies (`pip install -e .`).
2. Ensure the M3 model artifact exists
   (`python scripts/run_baseline.py run` if it does not).
3. `from anomaly_detector.inference import load_detector, InferenceStatus` —
   import from the `inference` package only.
4. Call `load_detector()` **once** at start-up and keep the object.
5. Build `FeatureWindowRequest` from the feature pipeline output, using the
   project `sample_key` as `window_id` so results join back to existing tables.
6. Branch on `result.status` first, `result.reason_code` second,
   `result.is_anomaly` last.
7. Forward only `status == "scored"` results to the correlation stage; keep
   `raw_score` and `threshold` separate and never re-threshold.
8. Count `insufficient_data` / `invalid_input` / `model_unavailable` /
   `inference_error` separately — they are four different operational
   problems.
9. Never retrain, retune or re-threshold from the inference component.
10. Do not expect evidence, correlation, incident grouping, explanations or
    alerts from this call — those are the downstream component's job.

---

## 14. Verified vs synthetic-only vs assumed

**Verified by running code and tests in this repository**
- The M1–M5 artifacts exist, load and are reproducible.
- The M3 artifact loads in a clean process and matches its manifest.
- The interface enforces feature order, schema version, types, finiteness,
  window boundaries, duplicate ids and label/identifier exclusion.
- Unscored results carry `is_anomaly = null`; no path reports them as normal.
- Scores are deterministic, keep the model's direction, and are flagged as
  non-probabilistic.
- Batch failures are per-row; the adapter forwards only scored results.
- The smoke test runs offline in a subprocess.
- 247 tests pass (208 pre-existing + 39 new).

**Synthetic-only**
- Every score and decision discussed here comes from synthetic data or the
  repository's synthetic splits. A flagged window is a lead for
  investigation, not proof of an attack.

**Assumed / not verified**
- That the downstream component's real interface resembles `DownstreamSink`.
- That the staleness and max-imputed-fraction defaults are the right
  production values.
- Real production latency, concurrency and availability behaviour.
- That the M3 Isolation Forest remains the right model once real data
  exists.

**Stopping here.** Milestone 6 is complete; Milestone 7 has not been started.
