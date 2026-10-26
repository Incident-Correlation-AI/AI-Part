# Evidence Correlation / Incident Draft — Integration Guide

**Status:** integration-ready in-process component (M10) · **ML contract:**
`1.1.0` · **Incident contract:** LAD `1.0`
**Upstream correlation:** `kalmak889900/automated-log-anomaly-detector` @
`b155a05c00741cc5f129610d2d90af0d18ca48ed`

This document explains how the ML anomaly detector hands scored results to the
evidence-correlation / incident-draft boundary, how evidence is redacted, how
failure states are kept separate, and how the LAD workstream consumes the
result. The reconnaissance that established these boundaries — including the
fact that **upstream contains no LLM** — is in
[`ml_llm_integration_reconnaissance.md`](ml_llm_integration_reconnaissance.md).

> **Absolute rule:** only `status == "scored"` results can become incident
> candidates. `unscored` is unknown, never normal and never a confirmed anomaly.
> An LLM failure never erases a valid ML score.

---

## 1. What was reused, and what was not

| Component | Source | Reuse |
|---|---|---|
| Anomaly grouping | Upstream `correlator.py` | Vendored **verbatim** under `src/anomaly_detector/correlation/upstream/`. Same `correlate_anomalies()` logic and `window_seconds` semantics. |
| Upstream incident construction | Upstream `incident.py`, `incident_pipeline.py` | **Not vendored.** Its shape cannot satisfy the LAD incident schema; the adapter maps groups into the canonical shape instead. |
| Optional explanation | This repository's `anomaly_detector.network.llm` | Reused unchanged through `explain_result()`; provider injection remains supported. |
| Upstream LLM / prompt / provider / redaction | — | **Does not exist.** No LLM implementation was invented or copied. |

Provenance, commit and the minimal modification are recorded in
`src/anomaly_detector/correlation/upstream/PROVENANCE.md`.

---

## 2. Quick start

```python
from anomaly_detector.correlation import (
    CorrelationConfig,
    EvidenceCorrelationPipeline,
    WindowEvidence,
)

pipeline = EvidenceCorrelationPipeline(CorrelationConfig())  # explanations off

outcome = pipeline.correlate(
    results,  # list[InferenceResult] from NetworkDetector.score_windows()
    {
        "sim-web-01|web-api|2026-01-01T00:10:00+00:00": WindowEvidence(
            window_id="sim-web-01|web-api|2026-01-01T00:10:00+00:00",
            supporting_event_ids=("run-00-evt-0000421",),
            contradicting_event_ids=(),
            redacted_refs=("s3://redacted/2026/run-00/421.json",),
        )
    },
)

print(outcome.counts())
for incident in outcome.incidents:
    print(incident["incident_id"], incident["severity"], incident["evidence"])
```

With the optional explanation layer:

```python
outcome = EvidenceCorrelationPipeline(
    CorrelationConfig(explain=True, llm_provider="mock")
).correlate(results, evidence)
print(outcome.llm_summary())
```

`llm_provider="mock"` is offline and deterministic. No provider is contacted
unless `explain=True` (or a provider is injected) and a non-`none` provider is
configured.

---

## 3. Invocation flow and module ownership

```text
NetworkDetector.score_windows(windows)          src/anomaly_detector/network/inference.py
        │  list[InferenceResult]                 contract 1.1.0
        ▼
EvidenceCorrelationPipeline.correlate()         src/anomaly_detector/correlation/pipeline.py
        │
        ├─ route_result()                        src/anomaly_detector/inference/contract.py
        │     only status == scored is forwardable
        ├─ candidate validation                  correlation/pipeline.py
        │     host/service/time/score/evidence
        ├─ validate_window_evidence()            correlation/evidence.py
        │     bounded, redacted references
        ├─ correlate_candidates()                correlation/incident.py
        │     calls vendored upstream correlate_anomalies()
        ├─ build_incident()                      correlation/incident.py
        │     canonical LAD incident draft
        └─ explain_result()                      network/llm.py   (optional)
              provider mocked/injected in tests; never changes the decision
        ▼
CorrelationOutcome                              pipeline.py
        │  incidents[] + explicit non-incident records
        ▼
LAD API / incident store                        other workstream
```

The component is a **library**, not a service. It has no import-time I/O, no
web framework, no queue and no hard-coded repository path.

---

## 4. Input contract (ML side)

### 4.1 `InferenceResult` fields preserved

| Field | Use in the incident |
|---|---|
| `contract_version` | `model_versions.contract_version` |
| `status` | Routing decision; only `scored` can become a candidate |
| `reason_code`, `reason` | Preserved on every non-incident record |
| `window_id` | Candidate identity, `source_window_ids`, deterministic incident ID input |
| `window_start`, `window_end` | Upstream timestamp, `windows[]` |
| `host_id` | `affected_entities[].host`; evidence lookup |
| `service_name` | Upstream grouping key and `affected_entities[].service` |
| `model_name`, `model_version`, `model_artifact_version` | `detector`, `model_versions` |
| `feature_schema_version`, `threshold_config_version` | `model_versions` |
| `raw_score`, `threshold`, `score_direction` | `anomaly_score`, score comparability, `model_versions.score_direction` |
| `is_anomaly` | Candidate eligibility; `False` is recorded as `scored_not_flagged` |
| `extra.network.severity` | Incident severity (mapped, see §7) |
| `extra.network.baseline.rule_hits` | Diagnostic checks and hypothesis evidence |
| `extra.network.evidence_features` | Deterministic hypothesis evidence; never labels |
| `extra.network.telemetry_quality` | Available to the explanation bundle |

No score is transformed, re-thresholded or recomputed. The decision rule stays
`is_anomaly = raw_score <= threshold` with `lower_is_more_anomalous`.

### 4.2 Evidence index

`correlate(results, evidence)` accepts:

```python
WindowEvidence(window_id, supporting_event_ids, contradicting_event_ids, redacted_refs)
{"window_id": WindowEvidence(...)}
{"window_id": ["event-id-1", "event-id-2"]}
{"window_id": "event-id-1"}
[WindowEvidence(...), ...]
```

Evidence is matched by the canonical `window_id`. Missing evidence never causes
an ID to be invented. By default `require_supporting_events=True`; a scored
anomaly without safe supporting IDs is rejected with
`integration_reason="missing_event_evidence"`. Set it to `False` only if the
caller explicitly accepts feature-only incidents; the hypothesis then records
that no event references were supplied.

> **M9 limitation:** network feature windows retain aggregate measurements, not
> the IDs of the events that produced them. The caller must build the evidence
> index from the normalized event stream (typically `event_id` filtered by
> `host`, `service` and `window_start <= event_ts < window_end`). This is shown
> in `scripts/run_correlation_rehearsal.py` and in the end-to-end test.

---

## 5. Evidence assembly and redaction

Only these categories can reach the incident or the explanation provider:

| Category | Allowed content |
|---|---|
| Identity | `window_id`, `host_id`, `service_name`, window start/end |
| ML assessment | raw score, threshold, direction, model/artifact/schema versions |
| Feature evidence | numeric `extra.network.evidence_features` and rule hit codes/values |
| Event references | opaque event IDs and already-redacted pointers |
| Quality | telemetry-quality counts supplied by the detector |
| Interpretation | validated LLM text, clearly labelled as a hypothesis |

Never accepted or forwarded:

* raw `message` text, request bodies, headers or cookies;
* credentials, passwords, API keys, bearer tokens, private keys;
* email addresses, JWTs, account identifiers;
* scenario names, run IDs, labels, ground truth or post-incident outcomes;
* arbitrary nested objects from `extra`.

Redaction rules (`correlation/evidence.py`):

* an event ID must match an opaque token pattern and must not contain an email,
  IPv4 address, JWT or secret-like word; an unsafe **supporting** or
  **contradicting** ID quarantines the candidate;
* an optional redacted pointer that fails validation is dropped and counted
  (`dropped_redacted_refs`), never copied;
* host/service identifiers are bounded and may not contain emails, JWTs,
  whitespace, `key=value` credentials or secret-like words;
* lists are truncated at configured bounds and
  `correlation.evidence_truncated=true` records it;
* provider/upstream exception text is sanitised before being stored or logged
  (URLs, emails, IPv4 addresses and `secret=value` pairs are replaced).

`llm.py` independently re-validates every draft: allowed citations only,
numbers must exist in the bundle, no PII/IP/email, no evaluation vocabulary,
and an explicit anomaly assertion that must match the model decision. A
rejected draft falls back to a deterministic evidence summary.

---

## 6. Status handling matrix

| Input condition | Result | Correlated? | LLM called? |
|---|---|---|---|
| `status="scored"`, `is_anomaly=True`, valid identity/time/score, safe evidence | Incident candidate | Yes | Only if enabled |
| `status="scored"`, `is_anomaly=False` | `scored_not_flagged` record | No | No |
| `status="scored"`, `is_anomaly=None` | `rejected: missing_decision` | No | No |
| `status="insufficient_data"` | `unscored` record with original reason code | No | No |
| `status="invalid_input"` | `unscored` record with original reason code | No | No |
| `status="model_unavailable"` | `unscored` record with original reason code | No | No |
| `status="inference_error"` | `unscored` record with original reason code | No | No |
| Missing result / `None` / malformed status | `unscored` with status `unknown` | No | No |
| Duplicate `window_id` in one batch | `duplicates` record; later occurrence ignored | No (first only) | First only |
| Missing host/service/time/score/threshold | `rejected` with a specific `integration_reason` | No | No |
| Unsafe or missing required evidence | `rejected: unsafe_*` / `missing_event_evidence` | No | No |
| Upstream correlator raises | `errors[] stage="upstream_correlation"`; candidates retained | No incidents | No |
| Incident build raises | `errors[] stage="incident_build"`; explanations retained | No incident | Already attempted |
| LLM provider times out / rate-limits / returns malformed output | LLM entry with `fallback_used` or `grounded=false`; **incident and score preserved** | Yes | Failure recorded |
| LLM disabled | No LLM entry; deterministic hypothesis/checks only | Yes | No |

Every run returns all categories simultaneously, so a dashboard can show
"3 incidents, 12 unscored, 1 rejected" without losing the reason for any item.

`CorrelationRecord.status` and `reason_code` always use the ML contract's
vocabulary. `integration_reason` explains the correlation-specific decision and
does **not** introduce a competing status enum.

---

## 7. Output contract

Each incident is a plain `dict` that validates against
`LAD/contracts/schemas/incident.schema.json` (copy and SHA-256 in
`tests/fixtures/contracts/`).

### Required canonical fields

`incident_id`, `schema_version` (`"1.0"`), `status` (`"open"`), `severity`,
`affected_entities`, `anomaly_score`, `detector`, `windows`, `evidence`,
`hypotheses`, `diagnostic_checks`, `created_at`, `updated_at`, `audit`,
`model_versions`.

### Additive fields (root schema allows additional properties)

| Field | Meaning |
|---|---|
| `correlation` | Upstream method, revision, window, candidate count, source window IDs, services, basis, `score_comparable`, `severity_source`, `evidence_truncated` |
| `llm_interpretations` | One validated entry per explained candidate (provider, grounded/fallback flags, attachment decision) |

### Semantics

* `incident_id` is a UUIDv5 of the sorted source `window_id` set, so a replay of
  the same evidence produces the same ID and does not create a duplicate
  incident. Timestamps still come from the clock.
* `anomaly_score` for a multi-window group uses the minimum score when all
  directions are `lower_is_more_anomalous`, or the maximum when all are
  `higher_is_more_anomalous`. Mixed directions keep the first score and set
  `correlation.score_comparable=false`. Individual scores remain in
  `outcome.candidates`.
* `severity` is the maximum known ML severity in the group; `none`/`unknown`
  maps to the schema minimum `low`, and `correlation.severity_source` says so.
* `hypotheses[0]` is deterministic and always says that correlation is
  temporal/identity association, not a confirmed root cause. An LLM hypothesis,
  if any, is a separate entry with `confidence="low"` and its own missing
  evidence.
* `diagnostic_checks[].executes` is always `false`.
* `audit` is empty: lifecycle changes belong to the LAD API/store.

---

## 8. Configuration

Defaults are offline and safe. Configuration can come from
`configs/correlation.yaml` and/or environment variables; environment values
override the file. Secret **values** are never stored in config; the HTTP
provider reads its key from the named environment variable at call time.

| Variable | Default | Purpose |
|---|---|---|
| `LAD_CORRELATION_CONFIG` | — | Path to a YAML config file |
| `LAD_CORRELATION_WINDOW_SECONDS` | `120` | Upstream grouping window |
| `LAD_CORRELATION_REQUIRE_EVENT_EVIDENCE` | `true` | Reject a candidate without supporting event IDs |
| `LAD_CORRELATION_MAX_EVENTS_PER_WINDOW` | `100` | Per-window reference bound |
| `LAD_CORRELATION_MAX_EVENTS_PER_INCIDENT` | `200` | Per-incident reference bound |
| `LAD_CORRELATION_MAX_REDACTED_REFS` | `50` | Redacted-pointer bound |
| `LAD_CORRELATION_MAX_LLM_HYPOTHESES` | `3` | LLM hypotheses per incident |
| `LAD_CORRELATION_MAX_DIAGNOSTIC_CHECKS` | `8` | Checks per incident |
| `LAD_CORRELATION_EXPLAIN` | `false` | Enable the optional explanation layer |
| `LAD_CORRELATION_INCLUDE_LLM_CHECKS` | `true` | Attach validated LLM checks |
| `LAD_CORRELATION_EXPLANATION_FALLBACK` | `true` | Use the deterministic fallback |
| `LAD_LLM_PROVIDER` | `mock` | `none` · `mock` · `http` |
| `LAD_LLM_ENDPOINT` | — | Required for `http` |
| `LAD_LLM_MODEL` | — | Provider model identifier |
| `LAD_LLM_API_KEY_ENV` | `LAD_LLM_API_KEY` | Name of the key's environment variable |
| `LAD_LLM_TIMEOUT_SECONDS` | `20` | Provider timeout |

An explicitly supplied config path that does not exist is an error; it never
silently falls back to defaults.

---

## 9. Error and status semantics

| Layer | Representation | Does it erase the ML result? |
|---|---|---|
| ML scoring | `InferenceResult.status` / `reason_code` | — |
| Routing | `route_result()` / `CorrelationRecord` | No |
| Upstream correlation | `CorrelationOutcome.errors[]` (`stage="upstream_correlation"`) | No; candidates retained |
| Incident construction | `CorrelationOutcome.errors[]` (`stage="incident_build"`) | No; candidate + explanation retained |
| LLM | `llm_interpretations[].grounded/fallback_used/rejected_reason` and `errors[] stage="explanation"` | No; incident still built |

An LLM outage is therefore visibly different from an ML outage: the incident
exists with `anomaly_score` and deterministic evidence, while the LLM entry says
it was unavailable. A provider timeout, 429, malformed JSON or grounding
violation is never promoted to a detection.

---

## 10. Downstream consumer

The consumer is the LAD incident API/store (`Incident-Correlation-AI/LAD`,
`POST /api/v1/events/batch` → detection → incidents). This component returns
drafts; it does not implement the API, persistence, acknowledgement/resolution,
dashboard rendering, alert delivery or AWS deployment.

Consumers should:

1. deduplicate on `incident_id` (replay-safe);
2. treat `severity="low"` with `severity_source` mentioning "unknown" as
   "unknown mapped to minimum", not as measured impact;
3. render `hypotheses` separately from `evidence` and `diagnostic_checks`;
4. display `llm_interpretations[].grounded/fallback_used` and never present a
   hypothesis as a root cause;
5. keep `correlation.score_comparable` in mind before sorting multi-window
   incidents by score.

---

## 11. Running the tests and the rehearsal

```bash
# New adapter/contract/status/redaction/LLM/end-to-end tests
python -m pytest tests/test_correlation_pipeline.py -q

# Full offline suite (no network, no live provider)
python -m pytest -q

# Local end-to-end rehearsal: simulated LAD events → model → evidence → incidents
python scripts/run_correlation_rehearsal.py
python scripts/run_correlation_rehearsal.py --explain --limit 200
```

Measured on 2026-10-26 in this workspace:

| Command | Result |
|---|---|
| `python -m pytest tests/test_correlation_pipeline.py -q` | **36 passed** |
| `python -m pytest -q` | **411 passed, 27 warnings** |
| `python scripts/run_correlation_rehearsal.py --explain --limit 200` | 200 scored windows → **53 candidates, 23 incidents, 0 errors**; 2 unscored; **53/53 explanations grounded**; upstream revision `b155a05…` |

The rehearsal uses the deterministic simulator and the real reloaded model. It
is offline. The mocked provider is the same `MockExplanationProvider` used by
the network extension; no live provider was contacted.

---

## 12. Known limitations and open questions

1. **Upstream has no LLM.** The prompt's "evidence-correlation LLM" premise is
   false; correlation is deterministic. The optional explanation layer is local.
2. **Upstream incident shape is incompatible** with the canonical LAD schema.
   The adapter maps groups; upstream's `incident.py` / `incident_pipeline.py`
   are intentionally not reused.
3. **Event IDs are not retained by M9 feature windows.** Callers must supply a
   `window_id → event IDs` index. A future feature change could emit per-window
   references, but that would alter the network schema and needs a version bump.
4. **Correlation is proximity + service only.** No trace/dependency graph, no
   cross-service causality, no quiet-period lifecycle or reopening logic.
5. **No live provider verification.** Timeout/429/malformed/grounding paths are
   tested with injected transports; a real endpoint, credentials and response
   latency remain unvalidated.
6. **Synthetic data only.** The end-to-end test proves the wiring, not
   real-world correlation precision.
7. **Upstream is private with no licence file.** Vendoring is documented, but
   publication of a public repository containing it needs owner/legal review.
8. **Redaction is narrow, not a shared library.** It blocks the documented
   patterns; field-aware organisation-wide redaction should replace it when
   available.
9. **The LAD API/store must still accept the additive fields**
   (`correlation`, `llm_interpretations`, per-incident model metadata). The root
   schema allows additional properties, but persistence and UI behaviour are
   owned by the other workstream.

## 13. Next concrete integration step

Wire `EvidenceCorrelationPipeline.correlate()` into the LAD detection worker
immediately after `NetworkDetector.score_windows()`, build the evidence index
from the ingested event batch, persist the returned incidents through the
existing incident store, and add an API contract test that reads a persisted
incident back using the canonical schema.
