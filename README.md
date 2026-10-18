# ML Anomaly Detector — Milestone 1: Data Pipeline Foundation

## Purpose

This is the data preparation foundation for a standalone machine-learning anomaly detector. **This milestone is data preparation only. No model training, inference, or anomaly scoring is implemented.**

The pipeline transforms structured log events into numerical feature windows suitable for later ML experiments.

## What this component does and does NOT do

### ✅ Does
- Define a common, OS-independent structured event schema
- Validate and normalize events with explicit error tracking
- Aggregate events into configurable time windows (host/service pairs)
- Generate numerical features from windows
- Generate reproducible synthetic datasets with ground-truth labels
- Maintain strict separation between features and labels (leakage prevention)
- Provide data quality reporting and exports
- Include offline unit tests

### ❌ Does NOT do
- Model training or inference
- Anomaly scoring/detection algorithms (e.g., Isolation Forest)
- Statistical anomaly detection (that comes in next milestone)
- Reasoning, correlation, root-cause analysis, or explanations
- APIs, web servers, dashboards, or cloud deployment
- External LLM integration

## Architecture

The pipeline follows: structured events → validation/normalization → window aggregation → feature generation → export with separate labels.

### Modules

- `src/anomaly_detector/schemas.py` - Event, window, and label data structures
- `src/anomaly_detector/validation.py` - Event validation, normalization, deduplication
- `src/anomaly_detector/preprocessing.py` - Time window aggregation
- `src/anomaly_detector/features.py` - Feature extraction utilities
- `src/anomaly_detector/synthetic_data.py` - Reproducible synthetic data generator
- `src/anomaly_detector/reporting.py` - Data quality reporting
- `src/anomaly_detector/dataset_adapters/` - Public dataset adapter interface

## Event Schema

```json
{
  "event_id": "evt-000001",
  "timestamp": "2026-10-18T10:00:12Z",
  "host_id": "web-01",
  "service": "web-api",
  "event_type": "http_request",
  "level": "INFO",
  "attributes": {
    "status_code": 500,
    "latency_ms": 850
  }
}
```

Required: `timestamp`, `host_id`, `service`, `event_type`. Optional: `event_id`, `level`, `attributes`, `source_metadata`.

## Feature Set

| Feature | Definition |
|---|---|
| `event_count` | Number of accepted events in window |
| `error_count` | Events with ERROR/FATAL/CRITICAL level |
| `error_ratio` | Errors / eligible events (None if no denominator) |
| `http_request_count` | HTTP request events |
| `http_5xx_count` | HTTP responses with 5xx status |
| `http_5xx_ratio` | 5xx / HTTP requests (None if no denominator) |
| `auth_failure_count` | Authentication failures |
| `restart_count` | Service restarts |
| `mean_latency_ms` | Mean of valid latency measurements |
| `p95_latency_ms` | 95th percentile of valid latencies |
| `latency_sample_count` | Count of valid latency samples |

## Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Usage

### Generate synthetic data

```bash
python scripts/generate_synthetic_data.py --seed 42 --window-size 1
```

This produces:
- `data/synthetic/synthetic_events.json` - Raw synthetic events
- `data/synthetic/synthetic_labels.json` - Ground truth labels (separate from features)
- `data/processed/features.csv` - Numerical feature matrix (no labels, no IDs)
- `data/processed/metadata.csv` - Window metadata (host, service, time ranges)
- `data/processed/labels.csv` - Ground truth labels
- `artifacts/data_quality_report.json` - Data quality report

### Prepare public dataset path (documentation only)

```bash
python scripts/prepare_dataset.py --dataset <name> --path <path>
```

This documents the adapter interface and requirements. No dataset is downloaded.

### Run tests

```bash
pytest tests/ -v
```

All tests run offline with no internet access.

## Data Handling Policies

- **Missing ≠ Zero**: Missing latency or unavailable denominators result in `None`/explicit missing values, not zeros
- **No silent discarding**: All rejections are tracked in the data quality report
- **Label separation**: Ground truth labels are never included in the feature matrix
- **Deduplication**: Only when reliable `event_id` exists
- **Empty windows**: Not invented by default (configurable via `allow_empty_windows`)
- **Privacy**: No secrets, tokens, or PII collected

## Synthetic Scenarios

### Normal
- Stable traffic with natural variation
- Occasional isolated errors (5% error rate - normal)
- Low-traffic periods
- Missing/delayed observations
- Normal scheduled restarts

### Anomalous
- HTTP 5xx surge
- Authentication failure surge
- Repeated service restarts
- Latency increase
- Abnormal event volume spike

## Reproducibility

All synthetic data generation is deterministic given a seed. The same seed produces identical outputs.

## Public Dataset Adapter Path

If a suitable public dataset becomes available:
1. Implement adapter extending `BaseDatasetAdapter` in `src/anomaly_detector/dataset_adapters/`
2. Document dataset name, source, license, field mappings, label semantics, limitations
3. Include small fixture for testing (no large files committed)
4. Ensure tests still run offline

Current status: documented in `artifacts/dataset_adapter_status.json` after running prepare_dataset.

## Limitations

- Synthetic data validates pipeline behavior only; does not establish real-world detection performance
- Adapter interface provided but no external dataset downloaded/verified
- OS-specific log parsers not implemented (intentionally out of scope for Milestone 1)

## Next Milestone

**Milestone 2**: Establish a statistical baseline and train/evaluate Isolation Forest (or other detectors) using the validated feature windows from Milestone 1. This milestone focuses exclusively on data preparation.

## Evidence of Label Separation

The feature dataframe contains only numerical features (event counts, ratios, statistics). Metadata (host_id, service, timestamps) and labels are stored in separate CSV files. Tests verify no leakage indicators appear in feature columns.

## Acceptance Criteria

All Milestone 1 requirements are satisfied: event schema, validation/normalization, window aggregation, feature generation, reproducible synthetic data, label separation with leakage tests, data quality reporting, public dataset adapter path (documented), offline tests, and reproducible commands.
