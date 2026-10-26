# Evidence Correlation / LLM Integration — Reconnaissance

**Status:** reconnaissance recorded before implementation · **Integration
milestone:** M10
**Upstream inspected:** `kalmak889900/automated-log-anomaly-detector` @
`b155a05c00741cc5f129610d2d90af0d18ca48ed` (branch `master`, private)
**Local ML side:** this repository @ `13535db` (M9 complete)

This document records what actually exists on both sides, where the integration
prompt's premise does not match the code, and the smallest safe boundary chosen
as a result. The implementation and its test results are described in
[`ml_llm_integration.md`](ml_llm_integration.md).

---

## 1. The premise mismatch (read this first)

The integration prompt says the upstream repository "contains an existing
**evidence-correlation LLM** component" and instructs the implementer to reuse
its prompt, provider abstraction, redaction and error handling.

That component does **not** exist. The upstream repository was cloned read-only
and every Python module was inspected:

| Upstream file | What it contains | LLM? |
|---|---|---|
| `correlator.py` | `correlate_anomalies()`: groups anomalies by `service` and timestamp proximity | No |
| `incident_pipeline.py` | `create_incidents()`: turns groups into upstream incident dicts | No |
| `incident.py` | upstream incident shape, title and UUID helper | No |
| `detector.py` | MAD anomaly detector | No |
| `analyzer.py`, `compute.py`, `severity.py` | analysis helpers | No |
| `main.py` | CLI entry point | No |
| `test_*.py` | detector/correlator tests | No LLM tests |

There is **no** prompt template, provider abstraction, model name, API key
handling, redaction layer, retry/timeout policy, structured-output validator or
LLM test anywhere in the repository. Searching for `llm`, `prompt`, `provider`,
`openai`, `anthropic`, `bedrock`, `completion` and `redact` returns nothing
relevant.

The planning pack agrees with the code, not the prompt: `AI_SPECIFICATION.md`
says "Detection should be transparent and evaluable; it must not depend on an
LLM", and its explanation section says "An external model is deferred and must
never receive sensitive raw logs by default."

**Decision:** do not invent an upstream LLM and do not claim to reuse one.
Reuse the real upstream **correlation** implementation, and use this
repository's existing optional `anomaly_detector.network.llm` adapter as the
only LLM layer. Document the mismatch here and in the integration guide.

---

## 2. What the upstream correlation code actually does

`correlator.py` is small, deterministic and dependency-free:

```python
def correlate_anomalies(anomalies: list[dict], window_seconds: int = 120) -> list[list[dict]]:
    ...
```

* Accepts a list of anomaly dictionaries.
* Reads only `service` and `timestamp`; entries missing either field are
  silently skipped.
* Requires timezone-aware ISO-8601 timestamps (`Z` is accepted); a naive
  timestamp raises `ValueError`.
* Sorts by timestamp, then greedily extends the most recent matching group when
  the same service is within `window_seconds` of the group's last member.
* Returns candidate groups. The docstring is explicit: "This creates candidate
  groups, not confirmed root causes."
* Rejects a negative `window_seconds`.

`incident_pipeline.py` then emits an upstream incident with `incident_id`,
`title`, `status`, `created_at`, `affected_services`, `evidence` and
`evidence_count`.

### Why the upstream incident shape is not vendored

The canonical downstream contract is
`LAD/contracts/schemas/incident.schema.json`. It requires:

* `evidence.supporting_event_ids` and `evidence.contradicting_event_ids`
  (structurally separate arrays);
* `hypotheses` with statement, confidence, supporting/contradicting IDs and
  missing evidence;
* `diagnostic_checks` with `executes: false`;
* `affected_entities`, `anomaly_score`, `detector`, `model_versions`, `audit`.

The upstream incident has `affected_services` and free-form `evidence`; it
cannot satisfy that schema. Copying it would either fork the contract or
produce invalid incidents.

**Decision:** vendor `correlator.py` verbatim (with a provenance header) and
map its groups into the canonical incident shape in a new adapter. Do not
vendor `incident.py` / `incident_pipeline.py`.

---

## 3. The local ML contract

Normative document: [`ml_inference_contract.md`](ml_inference_contract.md).

| Property | Value |
|---|---|
| Contract version | `1.1.0` |
| Result type | `InferenceResult` |
| Statuses | `scored`, `insufficient_data`, `invalid_input`, `model_unavailable`, `inference_error` |
| Forwarding rule | `route_result(result).forwardable` — only `scored` |
| Score direction | `lower_is_more_anomalous` |
| Decision | `is_anomaly = raw_score <= threshold` |
| Unscored decision | `is_anomaly = None` (never `False`) |
| Identity | `window_id`, `host_id`, `service_name`, `window_start`, `window_end` |
| Model metadata | `model_name`, `model_version`, `model_artifact_version`, `feature_schema_version`, `threshold_config_version` |
| Network evidence | `extra.network.baseline.rule_hits`, `extra.network.evidence_features`, `extra.network.severity`, `extra.network.telemetry_quality` |

The M9 network windows do **not** retain the event IDs that produced them; the
feature pipeline deliberately keeps only aggregate measurements. Event
references therefore come from the caller as a separate, narrow evidence index.
The adapter never accepts raw log text.

---

## 4. Schema and behaviour mismatches

| Area | Upstream | Local / canonical | Resolution |
|---|---|---|---|
| LLM | none | optional grounded `network.llm` adapter | Reuse the local adapter; do not fabricate upstream LLM code. |
| Correlation unit | one anomaly dict | `InferenceResult` with status and metadata | Convert only scored, flagged results into candidates. |
| Grouping key | `service` + time | `service_name`, `host_id`, window bounds | Preserve both; upstream reads `service`/`timestamp` from the candidate. |
| Incident shape | `affected_services`, free-form evidence | `affected_entities`, typed evidence, hypotheses, checks, audit | Map in the adapter; validate against the real JSON Schema. |
| Evidence IDs | not used | required `supporting_event_ids` / `contradicting_event_ids` | Accept a validated `window_id → WindowEvidence` index; never invent IDs. |
| Redaction | none | planning pack requires redaction; local adapter rejects PII/secrets | Validate references before correlation; bound sizes; sanitise diagnostics. |
| Score semantics | MAD anomaly list | lower-is-more-anomalous IF score | Preserve every candidate score; summarise a group only when directions agree. |
| Error model | exceptions | explicit status/reason vocabulary | Keep ML statuses; add integration-stage errors under a separate field. |
| Packaging | flat modules, absolute imports | installable `anomaly_detector` package | Vendor the one module under `correlation/upstream/` instead of path hacks. |

---

## 5. Chosen boundary

```text
canonical LAD events (elsewhere)                caller-supplied evidence index
            │                                              │
            ▼                                              ▼
   NetworkDetector.score_windows()  ── InferenceResult ──► EvidenceCorrelationPipeline
                                                          │
                                     route_result() ──────┤ only status == scored
                                     candidate validation ┤ host/service/time/score
                                     reference redaction  ┤ evidence IDs + redacted refs
                                                          ▼
                                     upstream correlate_anomalies()  (vendored)
                                                          │
                                                          ▼
                                     canonical incident draft (validated by tests)
                                                          │
                                     optional network.llm explain_result()
                                     (mocked in tests; off by default)
                                                          ▼
                                     LAD API / incident store (other workstream)
```

The adapter is a library, not a service. It has no import-time I/O, no network
call unless an HTTP provider is explicitly configured and invoked, and no
hidden working-directory dependency.

---

## 6. Files created or changed

| Path | Purpose |
|---|---|
| `src/anomaly_detector/correlation/upstream/correlator.py` | Vendored upstream `correlator.py` (logic unchanged). |
| `src/anomaly_detector/correlation/upstream/PROVENANCE.md` | Upstream repository, commit, file and modification record. |
| `src/anomaly_detector/correlation/evidence.py` | `WindowEvidence`, reference validation, redaction boundary. |
| `src/anomaly_detector/correlation/incident.py` | Upstream group → canonical incident mapping. |
| `src/anomaly_detector/correlation/pipeline.py` | `EvidenceCorrelationPipeline`, status handling, LLM orchestration. |
| `src/anomaly_detector/correlation/config.py` | Explicit config + environment overrides. |
| `src/anomaly_detector/correlation/__init__.py` | Public API. |
| `configs/correlation.yaml` | Example configuration with no secrets. |
| `scripts/run_correlation_rehearsal.py` | Offline end-to-end rehearsal. |
| `tests/test_correlation_pipeline.py` | Contract, status, redaction, LLM failure and end-to-end tests. |
| `tests/fixtures/contracts/incident.schema.json` | Canonical schema copy with hash provenance. |
| `docs/ml_llm_integration.md` | Integration guide and actual results. |
| `docs/ml_llm_integration_reconnaissance.md` | This document. |

No upstream file was modified. No ML code, score direction, threshold,
identifier or status was changed.

---

## 7. Risks and unresolved decisions

| # | Risk / question | Current position |
|---|---|---|
| 1 | Upstream has no LLM despite the prompt | Documented; local optional adapter used. Team should correct the prompt/premise. |
| 2 | Upstream is private and has no licence file | Vendored with explicit provenance; publication of the containing repository needs legal/ownership review. |
| 3 | M9 windows do not retain event IDs | Caller must supply the evidence index; documented as an integration requirement. A future feature-pipeline change could emit per-window event references. |
| 4 | Upstream correlation is time + service proximity only | Preserved as-is and labelled "association, not root cause"; the adapter does not add trace/dependency inference. |
| 5 | Group-level `anomaly_score` | Uses the most anomalous score when directions agree, otherwise the first score with `correlation.score_comparable = false`. Every candidate score is retained. |
| 6 | No live LLM/provider was contacted | Offline mock and injected transports only; live verification is explicitly not run. |
| 7 | Incident persistence and API ownership | Out of scope; the adapter returns a schema-valid draft for the LAD workstream. |
| 8 | Redaction policy is local, not a shared organisation library | Narrow allowlist/bypass rules documented; replace with the shared policy when it exists. |
