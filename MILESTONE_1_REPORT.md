# Milestone 1 — Completion Report

**Project:** Automated Log Anomaly Detector — ML component (standalone)
**Milestone:** 1 — Data Pipeline Foundation
**Status:** ✅ Complete — data pipeline implemented, tested offline, and documented.
**Boundary:** No model training, anomaly scoring, reasoning, correlation, API,
dashboard, or cloud code was written. Milestone 2 begins only after this data
foundation is trusted.

---

## 1. Summary

Milestone 1 builds the trustworthy data foundation for a later ML anomaly
detector. Structured log events are validated and normalized, aggregated into
configurable per-host/service time windows, converted into a documented
numerical feature matrix, and reported on. A reproducible synthetic generator
produces all required normal and anomalous scenarios with ground-truth labels
that are **kept strictly separate** from model features. A public-dataset adapter
interface is implemented with one concrete adapter (Loghub HDFS) and an offline
fixture test.

### Key design decisions (with rationale)

| Decision | Rationale |
|---|---|
| **Missing level → `UNKNOWN`, not `INFO`** | "Missing does not mean zero". Absence of a severity must not be coerced into a valid-looking value. |
| **Invalid measurements are dropped + the event is flagged *partially usable*** | Do not silently discard a record, and do not coerce an invalid latency into a valid one. The event is still counted; only the bad value is removed. |
| **Reject only unusable records** | Missing/unparseable `timestamp`, empty `host_id`/`service`, or non-mapping `attributes`. |
| **Deduplicate only on a genuine `event_id`** | Avoid collapsing distinct events that merely look similar. |
| **Epoch-aligned half-open windows `[start, end)`** | Reproducible boundaries independent of the first event's clock; multi-minute windows align correctly. |
| **Empty windows are not invented by default** | Materialization is opt-in and explicitly distinguished from absent telemetry. |
| **Ratio denominators use *eligibility* counts** | `http_5xx_ratio` uses only HTTP requests with a status code; `error_ratio` uses only events with a known severity. Undefined ratios stay `None`. |
| **Metadata, features, and labels are three separate tables** | Prevent accidental leakage into the model matrix; tested at both object and exported-file level. |

---

## 2. Folder structure

```text
ml-anomaly-detector/
├── README.md                         # setup, contracts, reproducible commands
├── pyproject.toml                    # package + pytest config
├── requirements.txt                  # minimal deps: pandas, pyyaml, numpy
├── .gitignore
├── configs/
│   └── default.yaml                  # window size, seed, paths, empty-window flag
├── docs/
│   └── DATASETS.md                   # dataset decision, mapping, blockers
├── data/                             # git-ignored (keeps .gitkeep)
│   ├── raw/
│   ├── processed/                    # features.csv, metadata.csv, labels.csv
│   └── synthetic/                    # synthetic_events.json, synthetic_labels.json
├── src/anomaly_detector/
│   ├── __init__.py
│   ├── schemas.py                    # event/window/label contracts
│   ├── validation.py                 # normalize, reject, dedup, partial-flag
│   ├── preprocessing.py              # window aggregation + feature computation
│   ├── features.py                   # feature/metadata/label DataFrames, leakage check
│   ├── synthetic_data.py             # deterministic scenario generator
│   ├── reporting.py                  # data-quality report
│   ├── pipeline.py                   # end-to-end wiring + exports
│   ├── config.py                     # config loader + defaults
│   └── dataset_adapters/
│       ├── __init__.py
│       ├── base.py                   # BaseDatasetAdapter interface
│       └── loghub_hdfs.py            # concrete Loghub HDFS adapter
├── scripts/
│   ├── generate_synthetic_data.py    # (1) generate + run pipeline + report
│   ├── run_pipeline.py               # (2) run pipeline on raw events JSON
│   └── prepare_dataset.py            # (3) adapter path / convert local dataset
├── tests/
│   ├── conftest.py                   # makes `src` importable offline
│   ├── fixtures/hdfs_sample/         # tiny HDFS-format fixture (log + labels)
│   ├── test_schemas.py
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_synthetic_data.py
│   ├── test_no_label_leakage.py
│   ├── test_reporting.py
│   ├── test_pipeline.py
│   └── test_dataset_adapters.py
└── artifacts/                        # git-ignored (keeps .gitkeep)
    ├── data_quality_report.json
    └── dataset_adapter_status.json
```

---

## 3. Contracts

### 3.1 Event schema (`schemas.StructuredEvent`)

| Field | Required | Notes |
|---|---|---|
| `event_id` | optional | Stable id for dedup; otherwise a synthetic `evt-<index>` is assigned and no dedup occurs. |
| `timestamp` | **required** | Parseable; normalized to **UTC**. Naive timestamps assumed UTC. |
| `host_id` | **required** | Non-empty. |
| `service` | **required** | Non-empty. |
| `event_type` | **required** | Normalized category (`http_request`, `auth_failure`, `service_restart`, `system_event`, `database_query`, `application_log`, `other`). |
| `level` | optional | Normalized (`DEBUG`/`INFO`/`WARNING`/`ERROR`/`UNKNOWN`). |
| `attributes` | optional | e.g. `status_code`, `latency_ms`. Invalid values dropped, not coerced. |
| `source_metadata` | optional | Small, safe, documented context only. |

### 3.2 Feature-window contract

**Metadata (not model features):** `window_start`, `window_end`, `host_id`,
`service`, `window_size_minutes`.

**Model feature matrix (numerical only, stable order):**
`event_count`, `error_count`, `error_ratio`, `http_request_count`,
`http_5xx_count`, `http_5xx_ratio`, `auth_failure_count`, `restart_count`,
`mean_latency_ms`, `p95_latency_ms`, `latency_sample_count`,
`http_request_eligible_count`, `error_eligible_count`.

### 3.3 Label representation (`schemas.LabelInfo`)

`window_start`, `window_end`, `host_id`, `service`, `is_anomaly`,
`scenario_name`, `intensity`, `description`. Stored in a separate table and
never placed in the feature matrix.

---

## 4. Commands run and actual results

### 4.1 Test suite (offline)

```bash
$ python -m pytest tests/ -v
```

**Result: `51 passed in 4.60s`** — all offline, no network access.

Per-file summary:

| Test file | Tests | Focus | Result |
|---|---|---|---|
| `test_schemas.py` | 3 | event/window/label structures, feature/metadata separation | PASS |
| `test_validation.py` | 9 | timezone normalization, missing fields, invalid numerics, invalid attributes, duplicates | PASS |
| `test_preprocessing.py` | 10 | empty/low-volume windows, boundaries, out-of-order, multi-minute alignment, empty-window materialization | PASS |
| `test_features.py` | 8 | error/5xx/auth/restart counts, ratios, zero-denominator handling, latency + p95, eligibility semantics | PASS |
| `test_synthetic_data.py` | 7 | reproducibility (identical values), seed sensitivity, all scenarios present | PASS |
| `test_no_label_leakage.py` | 3 | feature columns clean; exported feature CSV has no label/metadata columns | PASS |
| `test_reporting.py` | 2 | output schema stability, report counts | PASS |
| `test_pipeline.py` | 3 | accepted/rejected/duplicated/partial counts, raw-field missingness, feature matrix cleanliness | PASS |
| `test_dataset_adapters.py` | 6 | adapter interface, `can_load`, parsing, block-id extraction, label separation, provenance metadata | PASS |

Also verified in a **clean virtual environment** with `pip install -e .` and
`pip install -r requirements.txt`: **51 passed**.

### 4.2 End-to-end generation

```bash
$ python scripts/generate_synthetic_data.py --config configs/default.yaml
Generating synthetic data (seed=42, window=1m)
  processed input events : 5367
  accepted events        : 5367
  rejected events        : 0
  feature windows        : 227
  labels (separate)      : 256 (anomaly=45)
  features  -> data/processed/features.csv
  metadata  -> data/processed/metadata.csv
  labels    -> data/processed/labels.csv
  report    -> artifacts/data_quality_report.json
  scenarios : stable_traffic, occasional_errors, low_traffic, missing_delayed,
              normal_restart, http_5xx_surge, auth_failure_surge, restart_surge,
              latency_increase, volume_spike
```

### 4.3 Independent pipeline path

```bash
$ python scripts/run_pipeline.py \
      --events data/synthetic/synthetic_events.json \
      --labels data/synthetic/synthetic_labels.json
Processed 5367 events (accepted=5367, rejected=0, duplicated=0)
Windows: 227  Labels: 256
Features -> data/processed/features.csv
Report   -> artifacts/data_quality_report.json
```

### 4.4 Public-dataset adapter path

```bash
$ python scripts/prepare_dataset.py
No dataset selected. Documented adapter path and blockers:
  -> artifacts/dataset_adapter_status.json

$ python scripts/prepare_dataset.py --dataset loghub_hdfs --path tests/fixtures/hdfs_sample
Converted 8 events, 3 labels, 1 windows.
  -> artifacts/dataset_adapter_status.json
```

---

## 5. Data-quality report (actual output, seed 42, 1-minute windows)

| Metric | Value |
|---|---|
| Total processed | 5,367 |
| Accepted | 5,367 |
| Rejected | 0 |
| Duplicated | 0 |
| Partially usable | 0 |
| Feature windows | 227 |
| Hosts | 1 (`web-01`) |
| Services | 2 (`auth`, `web-api`) |
| Window size | 1 minute |
| Time range | 2026-10-18T10:00Z → 2026-10-18T19:10Z |
| Labels (separate) | 256 (anomaly = 45, normal = 211) |
| Feature missingness | `http_5xx_ratio`, `mean_latency_ms`, `p95_latency_ms` each missing in 11/227 windows (4.85%) — windows with no eligible measurement, represented as `None`, not `0` |

Label distribution by scenario:

| Scenario | Windows | Kind |
|---|---|---|
| stable_traffic | 60 | normal |
| occasional_errors | 60 | normal |
| low_traffic | 60 | normal |
| missing_delayed | 30 | normal |
| normal_restart | 1 | normal |
| http_5xx_surge | 10 | anomaly |
| auth_failure_surge | 5 | anomaly |
| restart_surge | 5 | anomaly |
| latency_increase | 15 | anomaly |
| volume_spike | 10 | anomaly |

---

## 6. Evidence that labels are excluded from model features

1. **By construction.** `FeatureWindow.to_features_dict()` and the module-level
   `FEATURE_COLUMNS` define the model matrix and contain only numeric features.
   `host_id`, `service`, timestamps, window boundaries, `is_anomaly`,
   `scenario_name` and `intensity` are absent.
2. **Three separate exports.** `features.csv` / `metadata.csv` / `labels.csv`
   are written independently by `pipeline.export_result`.
3. **Runtime guard.** Both `generate_synthetic_data.py` and `run_pipeline.py`
   call `check_label_leakage()` and abort with a non-zero exit code if any label
   or metadata column appears.
4. **Tested.** `test_no_label_leakage.py::test_exported_feature_file_has_no_label_columns`
   opens the exported feature CSV and asserts the header is disjoint from
   `{is_anomaly, scenario_name, host_id, service, window_start, window_end,
   intensity, description}`, while also asserting the labels CSV *does* contain
   the labels.

Verified header of `data/processed/features.csv`:

```
event_count,error_count,error_ratio,http_request_count,http_5xx_count,http_5xx_ratio,
auth_failure_count,restart_count,mean_latency_ms,p95_latency_ms,latency_sample_count,
http_request_eligible_count,error_eligible_count
```

---

## 7. Public dataset: implemented path and exact blocker

- **Interface:** `anomaly_detector.dataset_adapters.base.BaseDatasetAdapter`
  (`can_load`, `load_data → (events, labels)`, `get_dataset_info`).
- **Implemented adapter:** Loghub HDFS (`dataset_adapters/loghub_hdfs.py`).
  Parses `YYMMDD HHMMSS PID LEVEL COMPONENT: MESSAGE`, assumes year 2008 (the raw
  format omits it), extracts `blk_-?\d+` block ids, and returns block-level labels
  from `anomaly_label.csv` separately.
- **Offline fixture:** `tests/fixtures/hdfs_sample/` (8-line log + 3-row label
   CSV, clearly synthetic) drives `test_dataset_adapters.py`.

**Exact blocker for a "verified downloaded dataset":** no public dataset was
downloaded, inspected, or validated. Network access to a multi-hundred-megabyte
dataset is out of scope for a data-foundation milestone, and fabricating
provenance is prohibited. The milestone therefore ships the interface, one
concrete adapter, an offline fixture test, and a documented candidate decision
(`docs/DATASETS.md`) covering HDFS, BGL, Thunderbird, OpenStack, and non-log
datasets (e.g. CICIDS/NSL-KDD, rejected as flow data rather than logs).

---

## 7b. Errors encountered and fixed

| # | Error | Cause | Fix |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'anomaly_detector'` | `src/` layout not on `sys.path` during tests | Added `tests/conftest.py`; scripts insert `src` on `sys.path` |
| 2 | `SyntaxError` in `synthetic_data.py` (shell `echo "done"` appended) | A heredoc terminator leaked into the Python file | Stripped the trailing shell lines |
| 3 | `AttributeError: 'SyntheticDataGenerator' object has no attribute 'generate_occasional_errors'` | Method defined as `generate_isolated_errors`, called as `generate_occasional_errors` | Renamed the method to match the call site |
| 4 | `AttributeError: 'FeatureWindow' object has no attribute 'host_id'` | Report read `w.host_id` instead of the metadata object | Use `w.metadata.host_id` / `.service` |
| 5 | `TypeError: Object of type set is not JSON serializable` | Report stored `set()` for hosts/services | Use `sorted({...})` lists |
| 5b | `DeprecationWarning: datetime.utcnow()` | Python 3.13 deprecation | Switched to `datetime.now(timezone.utc)` |
| 6 | `TypeError: expected str, bytes or os.PathLike object, not NoneType` in `run_pipeline.py` | CLI args defaulted to `None` and were used directly | Fall back to config defaults (`cfg.output_dir`, `cfg.artifacts_dir`) |
| 7 | HDFS adapter returned **0 events** | Timestamp built as `"2008"+"081109"` = 10-char date, unparseable | Drop the `YY` prefix: `f"{year}{date[2:]}"` |
| 8 | `adapter.can_load(<directory>)` returned `False` | `can_load` only handled files | Resolve file/directory paths before checking |
| 9 | `test_auth_failure_and_restart_count` failed (`restart_count == 0`) | Test used a different host/service, so the restart formed a separate window | Corrected the test to use one host/service (test bug, not code bug) |
| 10 | HDFS fixture not tracked by git | `*.log` in `.gitignore` matched the fixture | Added `!tests/fixtures/**` exception |
| 11 | `data_quality_report` missingness always `0` | Missingness measured post-normalization (fields always present) | Measure missingness on the **raw** input events |

All issues above are fixed; the suite is green.

---

## 8. Known limitations

- Synthetic data validates pipeline behavior only; it does **not** estimate
  real-world detection performance.
- No public dataset was downloaded; the adapter path is documented with an exact
  blocker and an offline fixture test.
- Loghub HDFS labels are block-level (instance-level), not window-level;
  projecting them onto windows is deferred and carries a temporal-leakage risk.
- The feature set is intentionally modest for this milestone.
- Windows are aligned to Unix-epoch boundaries; document if a different alignment
  convention is ever required.
- `error_ratio` eligibility treats any event with a known severity as eligible;
  this is documented and may be revisited.

---

## 9. Milestone 1 completion checklist

- [x] Common event schema documented.
- [x] Validation and normalization implemented.
- [x] Window aggregation implemented and configurable.
- [x] Initial numerical features generated correctly.
- [x] Synthetic normal/anomalous scenarios reproducible.
- [x] Ground-truth labels separate from feature columns.
- [x] Data-quality report and export work.
- [x] Public-dataset adapter path documented (interface + HDFS adapter + fixture).
- [x] Offline tests pass (51/51).
- [x] README contains reproducible commands.
- [x] No model training, anomaly scoring, reasoning, correlation, API, dashboard,
      or AWS code exists.

---

## 10. Boundary with Milestone 2

Milestone 1 ends here. Milestone 2 may begin **only after** this data foundation
is trusted: establish a statistical baseline and train/evaluate Isolation Forest
on the validated feature windows (`data/processed/features.csv`), using the
separate `labels.csv` for evaluation and keeping leakage prevention intact.
