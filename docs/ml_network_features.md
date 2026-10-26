# Network / Service-Log Features — `network-features-v1`

**Status:** implemented and tested · **Feature schema:** `network-features-v1`
**Contract:** `1.1.0` · **Window size:** 1 minute (configurable)

This document defines the data source, the event mapping, the window rules, the
39-column feature schema and the leakage controls for the network extension.
The existing `features-v1` schema is unchanged and is still documented in
[`ml_inference_contract.md`](ml_inference_contract.md).

---

## 1. Data flow

```
deterministic local simulator
    events.jsonl  ── canonical LAD envelopes (schema 1.0)
    labels.csv    ── evaluation-only channel (never joined into features)
            │
            ▼
LAD adapter (lad_adapter.py)
    validate → deduplicate → map → pseudonymise → count quality problems
            │
            ▼
window builder (features.py)
    epoch-aligned [start, end) windows per (host, service)
    + causal rolling features from prior windows only
            │
            ▼
features.csv + metadata.csv + labels.csv
            │
            ▼
split → fit (train, normal-only, clean) → threshold (validation) → score
```

The simulator is the only data source in this extension. No production log,
external host or network request is used.

---

## 2. Deterministic local simulator

`src/anomaly_detector/network/simulator.py`

* Emits canonical LAD `1.0` envelopes and a **separate** label table.
* Uses a per-run seeded `random.Random`; the same seed produces byte-identical
  event and label content.
* Runs start one day apart (`start_time + run_index` days), so time splits are
  chronological rather than minute-of-day interleaved.
* Hard event bound (`max_events`, default 250,000). A run that hits the bound is
  marked `truncated` rather than silently producing a partial dataset.
* Contains no socket, subprocess, cloud SDK or exploit traffic. Authentication
  records are log lines describing an outcome; no login is attempted.

### 2.1 Streams

| Host | Service | Primary events |
|---|---|---|
| `sim-web-01` | `web-api` | HTTP requests, errors, restarts |
| `sim-web-01` | `auth-service` | Authentication outcomes, HTTP side-traffic |
| `sim-db-01` | `orders-db` | Database queries, timeouts, latency |

### 2.2 Scenario schedule (per 90-minute run)

| Minutes | Scenario | Labelled anomalous? |
|---|---|---|
| 0–15 | `normal` | no |
| 15–25 | `http_5xx_burst` | yes, affected HTTP streams |
| 25–35 | `malformed_telemetry` | no; `telemetry_issue=True` |
| 35–45 | `db_timeout_latency` | yes, `orders-db` |
| 45–55 | `normal_recovery` | no |
| 55–65 | `auth_failure_burst` | yes, `auth-service` |
| 65–75 | `benign_high_traffic` | no |
| 75–85 | `benign_latency_shift` | no |
| 85–90 | `benign_restart` | no |

Only the three incident families are labelled anomalous. Benign traffic,
latency and restart shifts stay negative so their false-positive rate is
measurable. The malformed-telemetry family is also negative, but marked as a
telemetry issue and reported separately.

### 2.3 Deliberate telemetry damage

The malformed-telemetry interval injects duplicate IDs, out-of-order events,
naive timestamps, missing hosts/IDs, unsupported schema versions, invalid
status codes, invalid latencies and unknown categories. At minute 33,
`orders-db` emits **no accepted event at all** for that window, so the detector
must return `insufficient_data` rather than “normal”.

### 2.4 Labels are a separate channel

`labels.csv` carries `window_start`, `window_end`, `host_id`, `service`,
`is_anomaly`, `scenario_name`, `run_id`, `telemetry_issue`, `description`.
`events.jsonl` carries no label, scenario or run field. The feature builder
never reads the label columns; alignment happens after feature construction.

---

## 3. Window rules

Windows are the existing epoch-aligned, half-open `[start, end)` windows used by
`anomaly_detector.preprocessing`:

* one window per `(host_id, service, window_start)`;
* events with `start <= event_ts < end` belong to the window;
* no observation is invented: an empty window is materialised as an empty
  feature row, not dropped;
* window end is exclusive, so a 1-minute window at `00:00` covers
  `00:00:00.000`–`00:00:59.999…`.

The pipeline builds the window grid from the label channel so that every
evaluation window exists even when no event was accepted. This is what allows
`no_events` to be reported explicitly.

---

## 4. Canonical LAD event mapping

`src/anomaly_detector/network/lad_adapter.py`

The adapter accepts LAD `schema_version == "1.0"` and maps the envelope onto the
existing `StructuredEvent`. It does not create a new event schema.

| LAD field | ML field | Rule |
|---|---|---|
| `event_id` | `event_id` | Required, trimmed, deduplication key. |
| `event_ts` | `timestamp` | Required, timezone-aware, normalised to UTC. Naive is rejected. |
| `host` | `host_id` | Required, non-empty. |
| `service` | `service` | Missing/blank becomes `"unknown"` plus a quality record. |
| `category` + `type` | `event_type` | Explicit mapping, see below. |
| `severity` | `level` | Mapped to the existing level vocabulary; unknown becomes `UNKNOWN`. |
| `attributes` | allowlisted attributes | Only documented fields survive. |
| `message` | **dropped** | Free text is never retained. |
| `source` | `source_metadata.lad_source` | Truncated to 64 characters; not a feature. |

### 4.1 Event-type mapping

| LAD `category` / `type` | ML `event_type` | Additional attributes |
|---|---|---|
| `http`, or type starts with `http` | `http_request` | `status_code`, `latency_ms` |
| `auth` failure tokens (`fail`, `denied`, `invalid`, `reject`) | `auth_failure` | pseudonymised source/account |
| `auth` success tokens (`success`, `accept`, `granted`, `ok`) | `auth_success` | pseudonymised source/account |
| `db`, or type starts with `db` | `database_query` | `db_timeout`, `db_connection_failure`, `latency_ms` |
| type contains `restart` | `service_restart` | — |
| `service` | `system_event` | — |
| `network` / `system` / `syslog` | `application_log` | — |
| anything else | `other` | — |

`status_code` may be derived from a type containing `5xx`/`4xx`, and that
derivation is recorded as a quality reason. Boolean and non-finite numeric
values are rejected or ignored rather than coerced.

### 4.2 Privacy

* `message` is never stored.
* `auth_source_id` / `auth_account_id` (and `source_id` / `account_id`) are
  replaced with a stable 20-character SHA-256 prefix before storage. They are
  used only for **distinct counts**, never as model inputs.
* Unknown attribute keys are dropped.
* The retained attributes are: `status_code`, `latency_ms`, `db_timeout`,
  `db_connection_failure`, pseudonymised auth identifiers.

### 4.3 Rejections and quality records

A rejected envelope does not enter the event stream. A partially usable
envelope still enters, with a quality record.

**Fatal rejection reasons**

| Reason | Meaning |
|---|---|
| `unsupported_schema_version` | Not LAD `1.0`. |
| `missing_event_id` | No usable deduplication key. |
| `duplicate_event_id` | Already seen in this batch. |
| `missing_host` | Cannot attribute to a host window. |
| `naive_timestamp` / `invalid_timestamp` | Cannot place in time. |
| `missing_type` | No `category`/`type`. |
| `invalid_envelope` | Not a mapping. |
| `batch_bounded` | The configured event bound was reached. |

**Partial quality reasons**

`missing_service`, `unknown_category`, `unknown_severity`, `invalid_attributes`,
`missing_observed_ts`, `invalid_status_code`, `status_code_derived_from_type`,
`invalid_latency`, `unmapped_auth_type`, `out_of_order_event`.

Every rejection and quality record is counted in the parse summary. Nothing is
silently dropped.

---

## 5. Feature schema — 39 columns, order significant

All columns are numeric (`float | None`). Counts are genuine counts and may be
zero. Ratios and measurements are `None` when their denominator/sample is
absent; they are never replaced by a meaningful-looking zero.

### 5.1 Volume and generic errors

| # | Column | Definition | Missing semantics |
|---|---|---|---|
| 1 | `event_count` | Accepted events in the window. | Never `None`; `0` is genuine. |
| 2 | `error_count` | Events whose level is `ERROR`, `FATAL` or `CRITICAL`. | `0`. |
| 3 | `error_ratio` | `error_count / error_eligible_count`. | `None` when no eligible event. |
| 4 | `error_eligible_count` | Events with a known severity. | `0`. |

### 5.2 HTTP

| # | Column | Definition | Missing semantics |
|---|---|---|---|
| 5 | `http_request_count` | Events typed `http_request`. | `0`. |
| 6 | `http_5xx_count` | Requests with `500 <= status < 600`. | `0`. |
| 7 | `http_5xx_ratio` | `http_5xx_count / http_status_eligible_count`. | `None` when no status-bearing request. |
| 8 | `http_4xx_count` | Requests with `400 <= status < 500`. | `0`. |
| 9 | `http_4xx_ratio` | `http_4xx_count / http_status_eligible_count`. | `None` when no status-bearing request. |
| 10 | `http_status_eligible_count` | Requests carrying a status code. | `0`. |

### 5.3 Authentication

| # | Column | Definition | Missing semantics |
|---|---|---|---|
| 11 | `auth_failure_count` | Events typed `auth_failure`. | `0`. |
| 12 | `auth_success_count` | Events typed `auth_success`. | `0`. |
| 13 | `auth_failure_ratio` | Failures / (failures + successes). | `None` when both are zero. |
| 14 | `auth_distinct_source_count` | Distinct pseudonymised source IDs. | `0`. |
| 15 | `auth_distinct_account_count` | Distinct pseudonymised account IDs. | `0`. |

### 5.4 Database

| # | Column | Definition | Missing semantics |
|---|---|---|---|
| 16 | `db_event_count` | Events typed `database_query`. | `0`. |
| 17 | `db_timeout_count` | DB events with `db_timeout`. | `0`. |
| 18 | `db_timeout_ratio` | Timeouts / DB events. | `None` when no DB event. |
| 19 | `db_connection_failure_count` | DB events with `db_connection_failure`. | `0`. |
| 20 | `db_connection_failure_ratio` | Connection failures / DB events. | `None` when no DB event. |

### 5.5 Latency

| # | Column | Definition | Missing semantics |
|---|---|---|---|
| 21 | `latency_sample_count` | Events with a finite, non-negative latency. | `0`. |
| 22 | `mean_latency_ms` | Arithmetic mean of samples. | `None` when no sample. |
| 23 | `p50_latency_ms` | Linear-interpolation median. | `None` when no sample. |
| 24 | `p95_latency_ms` | Linear-interpolation 95th percentile. | `None` when no sample. |
| 25 | `p99_latency_ms` | Linear-interpolation 99th percentile. | `None` when no sample. |
| 26 | `max_latency_ms` | Maximum sample. | `None` when no sample. |

### 5.6 Lifecycle

| # | Column | Definition | Missing semantics |
|---|---|---|---|
| 27 | `restart_count` | Events typed `service_restart`. | `0`. |

### 5.7 Telemetry quality

These are counts of quality records attributed to the window.

| # | Column | Definition | Missing semantics |
|---|---|---|---|
| 28 | `invalid_event_count` | Sum of all quality records for the window. | `0`. |
| 29 | `duplicate_event_count` | `duplicate_event_id` records. | `0`. |
| 30 | `out_of_order_event_count` | `out_of_order_event` records. | `0`. |
| 31 | `missing_service_count` | `missing_service` records. | `0`. |
| 32 | `unknown_category_count` | `unknown_category` records. | `0`. |
| 33 | `telemetry_gap_seconds` | Gap between the previous window's end and this window's start, floored at 0. | `None` for the first window in a stream. |

### 5.8 Causal rolling features

History is scoped to the **same `(host_id, service)` stream**, uses only
**earlier** windows, and is capped at the most recent 30 windows. Rolling
features are `None` until at least 5 prior windows exist.

| # | Column | Definition |
|---|---|---|
| 34 | `event_count_zscore` | Robust z-score of the current count against prior counts: `(x - median) / (1.4826 * MAD)`, clipped to `[-10, 10]`. Falls back to standard deviation when MAD is zero. |
| 35 | `http_request_count_zscore` | Same, for HTTP request count. |
| 36 | `http_5xx_ratio_delta` | Current 5xx ratio minus median of prior 5xx ratios. |
| 37 | `auth_failure_ratio_delta` | Current auth failure ratio minus median of prior auth failure ratios. |
| 38 | `db_timeout_ratio_delta` | Current DB timeout ratio minus median of prior DB timeout ratios. |
| 39 | `p95_latency_ratio` | Current p95 divided by the median of prior **positive** p95 values, clipped to `[0, 50]`. |

The current window is never part of its own baseline. The causal property is
tested by rebuilding a truncated event stream and asserting that earlier
windows are byte-for-byte unchanged.

---

## 6. Leakage controls

| Risk | Control |
|---|---|
| Label leakage | Labels, `scenario_name`, `run_id`, `telemetry_issue` are a separate table and are never passed to `_compute_features`. |
| Identifier leakage | `host_id`, `service`, `sample_key`, `event_id` are metadata/keys, not feature columns. |
| Future leakage | Rolling features read only prior rows of the same stream; no centred or forward window. |
| PII leakage | `message` dropped; auth identifiers pseudonymised and used only for distinct counts. |
| Split leakage | `time` split is strictly chronological; `group` split keeps whole runs disjoint. Both are integrity-checked. |
| Preprocessing leakage | Imputation medians are fitted on the training split only. |
| Training contamination | The forest is fitted on **training-split, normal-only, clean, adequately-covered** windows. |
| Threshold contamination | The threshold is selected on validation only; the test split is refused by the thresholding module. |
| Evaluation contamination | Test metrics are computed once; test labels are never used for selection. |

Tests assert the feature names contain no forbidden token, that feature values
are numeric, that an event stream truncated at minute 45 reproduces all earlier
windows exactly, and that rolling features are `None` before history exists.

---

## 7. Missingness and status interaction

At inference, `NetworkDetector` applies these guards before scoring:

| Condition | Status | Reason |
|---|---|---|
| `event_count` missing or zero | `insufficient_data` | `no_events` |
| `0 < event_count < min_window_events` (default 5) | `insufficient_data` | `insufficient_coverage` |
| `invalid_event_count > event_count` | `insufficient_data` | `telemetry_quality` |
| Required feature absent | `insufficient_data` | `missing_features` |
| More than 50% of features missing | `insufficient_data` | `missing_features` |
| Unknown feature present | `invalid_input` | `unknown_features` |
| Non-numeric / boolean value | `invalid_input` | `non_numeric_feature` |
| `±inf` value | `invalid_input` | `non_finite_feature` |

A window that reaches the model still has `None` values imputed with the
**training median**, and the imputed column names are reported in
`data_quality.imputed_features`.

---

## 8. Versioning and compatibility

* Schema identifier: `network-features-v1`.
* Column count: 39; order is significant; values are assembled by name, never
  by dictionary iteration order.
* A model artifact declares this schema in its manifest. The loader refuses an
  artifact whose manifest schema is not `network-features-v1`, or whose column
  list differs from the serialized model.
* `features-v1` remains a separate 13-column schema. A `features-v1` request is
  not silently accepted by the network detector.

Any future change to the allowlist, order or semantics requires a new schema
identifier (for example `network-features-v2`) and a new artifact.

---

## 9. Measured data volume (full configured run)

| Item | Value |
|---|---:|
| Runs | 10 |
| Minutes per run | 90 |
| Streams | 3 |
| Generated events | 72,823 |
| Accepted events | 71,959 |
| Rejected envelopes | 864 |
| Duplicate event IDs | 400 |
| Out-of-order events | 382 |
| Labelled windows | 2,700 |
| Incident windows | 400 |
| Telemetry-issue windows | 300 |
| Training windows used to fit (per strategy) | 1,200 |
| Cells imputed during fitting | 5,348 |

Rejection breakdown: duplicate `400`, missing host `161`, unsupported schema
`105`, missing event ID `99`, naive timestamp `99`.

---

## 10. Reproduce

```bash
python scripts/run_network_pipeline.py --config configs/network.yaml
```

This writes:

```
data/synthetic/network/events.jsonl        # canonical LAD envelopes
data/synthetic/network/labels.csv          # evaluation-only labels
artifacts/network/features.csv             # 39-column feature matrix
artifacts/network/metadata.csv             # keys and provenance
artifacts/network/lad_parse_summary.json   # accepted/rejected/quality counts
```

The simulator can be run alone:

```bash
python scripts/run_network_simulator.py --runs 3 --seed 42
```

Tests covering this document:

```bash
python -m pytest tests/test_network_features.py tests/test_network_lad_adapter.py -q
```
