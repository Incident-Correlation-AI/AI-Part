# ML Anomaly Detector — Milestone 1: Data Pipeline Foundation

A standalone **data-preparation** component for an Automated Log Anomaly Detector.
It turns structured log events into trustworthy numerical feature windows that a
later milestone can feed to anomaly-detection models.

> **Scope:** Milestone 1 is data preparation only. There is **no** model training,
> anomaly scoring, statistical detection, reasoning, correlation, API, dashboard or
> cloud code here. The next milestone (statistical baseline + Isolation Forest)
> starts only after this data foundation is tested and documented.

---

## 1. What this component does and does not do

**Does**

- Define a documented, OS-independent **structured event schema**.
- **Validate and normalize** events with explicit error handling and tracking of
  accepted / rejected / duplicated / partially-usable records.
- Aggregate events into **configurable time windows** per `host_id`/`service`.
- Generate a documented **numerical feature set** (with availability/sample counts).
- Generate **reproducible synthetic data** covering normal and anomalous scenarios.
- Keep **ground-truth labels strictly separate** from model features (leakage-tested).
- Produce a **data-quality report** and CSV exports.
- Provide a **public-dataset adapter path** (interface + one implemented adapter +
  offline fixture tests).
- Provide **offline unit tests** and reproducible commands.

**Does not (belongs to a later milestone or teammate)**

- No Isolation Forest, clustering, statistical scoring, model training or inference.
- No evidence correlation, incident reasoning, root-cause hypotheses, or explanations.
- No API, web server, dashboard, cloud SDK, external LLM, or AWS code.

---

## 2. Architecture

```
raw events (dicts / JSON)
        │
        ▼
 validation.py      normalize + reject/duplicate/flag  ──► data-quality report
        │              (UTC timestamps, levels, types)
        ▼
 preprocessing.py   epoch-aligned host/service windows
        │              (configurable size, no invented observations)
        ▼
 features.py        numerical feature matrix  ──► features.csv
        │                   │
        │                   └─ metadata.csv (host/service/time)  ← traceability
        ▼
 synthetic_data.py  labels.json / labels.csv  ← ground truth, kept separate
```

| Module | Responsibility |
|---|---|
| `src/anomaly_detector/schemas.py` | Event, window and label data structures (the data contract). |
| `src/anomaly_detector/validation.py` | Timestamp/field/type validation, normalization, dedup, partial-usability tracking. |
| `src/anomaly_detector/preprocessing.py` | Epoch-aligned window aggregation + per-window feature computation. |
| `src/anomaly_detector/features.py` | Feature/metadata/label DataFrames; leakage checker. |
| `src/anomaly_detector/synthetic_data.py` | Deterministic synthetic generator (all required scenarios). |
| `src/anomaly_detector/reporting.py` | Data-quality report (counts, missingness, feature availability, labels). |
| `src/anomaly_detector/pipeline.py` | End-to-end wiring + exports. |
| `src/anomaly_detector/config.py` | Config loading with documented defaults. |
| `src/anomaly_detector/dataset_adapters/` | Public-dataset adapter interface, base + Loghub HDFS adapter. |
| `scripts/generate_synthetic_data.py` | Generate synthetic data, run pipeline, write report. |
| `scripts/run_pipeline.py` | Run the pipeline on an existing raw-events JSON file. |
| `scripts/prepare_dataset.py` | Convert a public dataset via an adapter (never downloads). |
| `docs/DATASETS.md` | Candidate-dataset decision, mapping, limitations, blocker. |

---

## 3. Event schema

```json
{
  "event_id": "evt-000001",
  "timestamp": "2026-10-18T10:00:12Z",
  "host_id": "web-01",
  "service": "web-api",
  "event_type": "http_request",
  "level": "INFO",
  "attributes": { "status_code": 500, "latency_ms": 850 }
}
```

| Field | Required | Notes |
|---|---|---|
| `event_id` | optional | Stable id for deduplication. No id ⇒ no dedup. |
| `timestamp` | **required** | Parseable; normalized to **UTC** internally. |
| `host_id` | **required** | Non-empty source machine id. |
| `service` | **required** | Non-empty service id. |
| `event_type` | **required** | Normalized category; defaults to `other` if absent. |
| `level` | optional | Normalized severity; **missing ⇒ `UNKNOWN`** (not `INFO`). |
| `attributes` | optional | Event-specific mapping (e.g. `status_code`, `latency_ms`). |
| `source_metadata` | optional | Small, safe, documented source context only. |

Normalized levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `UNKNOWN`
(`WARN`→`WARNING`; `FATAL`/`CRIT`/`ERR`→`ERROR`).
Normalized event types: `http_request`, `auth_failure`, `service_restart`,
`system_event`, `database_query`, `application_log`, `other`.

No secrets, passwords, tokens, or unnecessary personal data are collected.

---

## 3b. Feature-window contract

Window metadata (traceability only, **not** model features):
`window_start`, `window_end`, `host_id`, `service`, `window_size_minutes`.

Model feature matrix (numerical only, stable order):

| Feature | Definition |
|---|---|
| `event_count` | Accepted events in the window |
| `error_count` | Events with `ERROR`/`FATAL`/`CRITICAL` |
| `error_ratio` | `error_count / error_eligible_count` (missing if denominator 0) |
| `http_request_count` | HTTP request events |
| `http_5xx_count` | HTTP responses with a 5xx status |
| `http_5xx_ratio` | `5xx / http_request_eligible_count` (missing if denominator 0) |
| `auth_failure_count` | Authentication-failure events |
| `restart_count` | Service-restart events |
| `mean_latency_ms` | Mean of valid latency observations (missing if none) |
| `p95_latency_ms` | 95th percentile of valid latencies (missing if none) |
| `latency_sample_count` | Count of valid latency samples |
| `http_request_eligible_count` | HTTP requests carrying a status code |
| `error_eligible_count` | Events with a known severity |

**Missing ≠ zero.** Missing/unavailable measurements are `None`, never a
fake zero. Sample counts are preserved so callers can tell a ratio based on a few
observations from one based on many.

---

## 4. Data-handling policies

- **Missing ≠ zero.** A missing latency is not a zero latency; unavailable error
  counts are not evidence of no errors.
- **No silent discarding.** All rejected/duplicated/partial records are reported.
- **Deduplication** only when a genuine `event_id` exists.
- **Empty windows** are not invented by default; enable with `allow_empty_windows`.
- **Out-of-order** events are sorted per host/service before bucketing.
- **Window boundaries** are epoch-aligned half-open intervals `[start, end)`.
- **Partial usability:** invalid optional measurements (e.g. negative latency,
  non-numeric status) are dropped from the event, which is still counted and
  flagged — the invalid value is **not** coerced to zero.
- **Label separation:** labels never enter the feature matrix (tested).

---

## 5. Setup

Requirements: Python 3.9+, pip.

```bash
# From the repository root:
cd ml-anomaly-detector

# Install dependencies for a clean checkout
pip install -r requirements.txt

# Or install as a package (editable, with dev extras)
pip install -e ".[dev]"
```

---

## 6. Reproducible command sequence

```bash
# (1) Generate synthetic data + run the whole pipeline + write the report
python scripts/generate_synthetic_data.py --config configs/default.yaml

# (2) Run the pipeline on an existing raw-events JSON file (independent path)
python scripts/run_pipeline.py \
    --events data/synthetic/synthetic_events.json \
    --labels data/synthetic/synthetic_labels.json

# (3) Document the public-dataset adapter path / blockers
python scripts/prepare_dataset.py

# (4) (optional) Convert a local dataset fixture through the adapter
python scripts/prepare_dataset.py --dataset loghub_hdfs --path tests/fixtures/hdfs_sample

# (5) Run all tests (offline)
pytest tests/ -v
```

Outputs:

| File | Description |
|---|---|
| `data/synthetic/synthetic_events.json` | Raw synthetic events |
| `data/synthetic/synthetic_labels.json` | Ground-truth labels (separate) |
| `data/processed/features.csv` | Numerical feature matrix |
| `data/processed/metadata.csv` | Window metadata (host, service, time) |
| `data/processed/labels.csv` | Ground-truth labels |
| `artifacts/data_quality_report.json` | Data-quality report |
| `artifacts/dataset_adapter_status.json` | Adapter decision/status |

Configuration defaults live in `configs/default.yaml` (window size, seed, output
paths, empty-window behaviour). All paths are relative to the repo root.

---

## 7. Data-quality reporting

`artifacts/data_quality_report.json` contains, where applicable:

- input record count; accepted / rejected / duplicated / partially-usable counts;
- rejection reasons;
- missingness by important source field;
- number of hosts, services and feature windows;
- feature availability/missingness per feature column;
- label distribution (for labelled synthetic data);
- time range and configured window size.

---

## 8. Public-dataset adapter path

See [`docs/DATASETS.md`](docs/DATASETS.md). Summary: the adapter interface is
`anomaly_detector.dataset_adapters.base.BaseDatasetAdapter`. A concrete
**Loghub HDFS** adapter is implemented and tested against a small offline fixture
(`tests/fixtures/hdfs_sample/`).

**Exact blocker:** no dataset was downloaded or fabricated. Network access and a
multi-hundred-megabyte download are out of scope for a data-foundation milestone,
so the milestone ships the interface, one adapter, an offline fixture test, and the
documented candidate-dataset decision instead.

---

## 9. Tests

```bash
pytest tests/ -v          # all tests, fully offline (no network access)
pytest tests/ -q          # quiet summary
```

Coverage includes: valid/invalid events, timezone normalization, missing required
fields, invalid numeric measurements, duplicates, out-of-order events, window
boundaries, empty/low-volume windows, zero-denominator ratios, reproducible
synthetic generation, required scenarios, label leakage, output-schema stability,
data-quality report counts, and the dataset-adapter interface.

---

## 10. Repository structure

```text
ml-anomaly-detector/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/default.yaml
├── docs/DATASETS.md
├── data/{raw,processed,synthetic}/   (git-ignored, keep .gitkeep)
├── src/anomaly_detector/
│   ├── __init__.py
│   ├── schemas.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── synthetic_data.py
│   ├── reporting.py
│   ├── pipeline.py
│   ├── config.py
│   └── dataset_adapters/{__init__.py, base.py, loghub_hdfs.py}
├── scripts/{generate_synthetic_data.py, run_pipeline.py, prepare_dataset.py}
├── tests/ (incl. fixtures/hdfs_sample/)
└── artifacts/ (git-ignored, keep .gitkeep)
```

---

## 11. Known limitations

- Synthetic data validates pipeline behavior; it does **not** establish
  real-world detection performance.
- No public dataset was downloaded; the adapter path is documented with an exact
  blocker and an offline fixture test.
- Loghub HDFS labels are block-level (instance-level), not window-level; projecting
  them onto windows is deferred and carries a temporal-leakage risk.
- The feature set is intentionally modest for this milestone.

## 12. Boundary with the next milestone

Milestone 1 stops here with a tested, documented data pipeline. The next milestone
establishes a statistical baseline and trains/evaluates Isolation Forest on these
validated feature windows. Do not start it until this data and its tests are
trustworthy.
