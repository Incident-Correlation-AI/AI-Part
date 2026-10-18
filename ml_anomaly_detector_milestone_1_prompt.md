# ML Anomaly Detector — Milestone 1: Data Pipeline Foundation

## Purpose

Build the first milestone of a standalone machine-learning anomaly detector for an Automated Log Anomaly Detector project.

**This milestone is data preparation only. Do not train or implement an anomaly-detection model yet.**

The ML component is owned separately from the reasoning/correlation component. Another teammate will handle evidence correlation, incident reasoning, root-cause hypotheses, and human-readable explanations. This component must focus on preparing trustworthy numerical data for later ML experiments.

## Recommended starting representation

Start with **structured log events**, then aggregate them into **numerical feature windows**.

Do not begin by feeding arbitrary raw log strings directly into a model. Different operating systems and applications emit different log formats, and a model such as Isolation Forest needs a consistent numerical feature matrix.

The initial pipeline is:

1. Define a common structured-event representation.
2. Validate and normalize events.
3. Aggregate events into time windows, initially one minute per host/service.
4. Derive numerical features from each window.
5. Generate reproducible synthetic normal/anomalous scenarios with separate ground-truth labels.
6. Create a documented adapter path for public datasets.
7. Validate outputs and produce a data-quality report.

The one-minute window is a configurable starting default, not a claim that it is optimal. Later experiments should test other window sizes.

## What happens at each step?

### Step 1 — Common structured-event representation

Define a small, OS-independent event schema. A representative event could look like:

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

These fields are an initial ML data contract, not a replacement for the team's eventual shared API contract.

- `event_id`: optional stable identifier useful for deduplication.
- `timestamp`: event time in a documented timezone convention; use UTC in the example.
- `host_id`: source machine identifier.
- `service`: application or service identifier.
- `event_type`: normalized event category.
- `level`: normalized severity when the source provides it.
- `attributes`: optional structured event-specific values.

Do not collect secrets, passwords, tokens, or unnecessary personal data. Keep source-specific metadata only when it is useful and safe.

### Step 2 — Validate and normalize

The pipeline should explicitly handle:

- Missing or malformed timestamps.
- Missing host/service identifiers.
- Invalid types, such as latency supplied as arbitrary text.
- Duplicate events where a stable event ID is available.
- Out-of-order events.
- Missing optional attributes.
- Invalid values such as negative latency.
- Unknown event types or levels.

Do not silently discard invalid records. Track accepted, rejected, duplicated, and partially usable records in a data-quality report.

**Missing does not mean zero.** A missing latency measurement is not a latency of zero; an unavailable error count is not evidence that no errors occurred. Represent availability or missingness explicitly where appropriate.

### Step 3 — Aggregate into time windows

Initially aggregate events into one-minute windows per host/service pair. Keep the window size configurable.

Example: hundreds of individual HTTP request events within a minute can become one feature row for that host and service.

Aggregation must define how it handles:

- Late-arriving events.
- Empty windows.
- Low-volume services.
- Duplicate events.
- Window boundaries and timestamps.
- Events with incomplete host/service metadata.

Do not invent observations for empty windows. If empty windows are materialized, document that behavior and distinguish them from absent telemetry.

### Step 4 — Create numerical features

Start with a modest, documented feature set:

| Feature | Definition |
|---|---|
| `event_count` | Number of accepted events in the window |
| `error_count` | Number of events classified as errors |
| `error_ratio` | Errors divided by eligible events, only when the denominator is valid |
| `http_request_count` | Number of eligible HTTP request events |
| `http_5xx_count` | Number of HTTP responses with a 5xx status |
| `http_5xx_ratio` | 5xx responses divided by eligible HTTP requests |
| `auth_failure_count` | Number of authentication failures |
| `restart_count` | Number of service restart events |
| `mean_latency_ms` | Mean of valid latency observations |
| `p95_latency_ms` | 95th percentile of valid latency observations when there are enough samples |

Not every feature applies to every service. Preserve feature availability and sample counts. Ratios with zero or unavailable denominators must not be silently converted into meaningful-looking zero values.

Do not use identifiers such as `host_id`, `service`, raw timestamps, or ground-truth labels as ordinary numerical model features. They can be retained as metadata for grouping, analysis, and evaluation.

### Step 5 — Generate synthetic scenarios

Create deterministic, reproducible synthetic datasets containing both normal and anomalous behavior.

Normal scenarios:
- Stable traffic with ordinary variation.
- Occasional isolated errors.
- Low-traffic and quiet periods.
- Normal service restarts.
- Missing or delayed observations.

Anomalous scenarios:
- Sudden HTTP 5xx spike.
- Authentication-failure surge.
- Repeated service restarts.
- Latency increase.
- Abnormal event-volume spike.

Each synthetic event or window should have ground-truth labels maintained separately from the input features. Use a configurable random seed and record scenario metadata separately.

**Prevent leakage:** the ground-truth label (`is_anomaly`, scenario name, or equivalent) must never appear in the model feature matrix. Keep labels separate from features, and make this separation testable.

Synthetic data validates pipeline behavior; it does not establish real-world detection performance.

### Step 6 — Prepare for public datasets

Identify suitable public log datasets, verify their licenses and documentation, and record:

- Dataset name and source.
- License/usage restrictions.
- Available fields and log format.
- Whether labels exist and what they mean.
- Known limitations and potential leakage.
- Mapping from source data to the common event format.
- Any information lost during conversion.

Implement an adapter for one suitable dataset if access and format are straightforward. If not, document the exact blocker and provide the adapter interface plus a small fixture for testing. Do not fabricate data provenance or claim that a dataset was downloaded when it was not.

### Step 7 — Validate and report

Milestone 1 should provide:

- A documented event representation.
- Validation and normalization logic.
- Window aggregation.
- Feature generation.
- Reproducible synthetic data generation.
- Separate labels and model features.
- A public-dataset adapter path and a documented first-dataset decision.
- Tests for valid and invalid inputs, duplicates, missing values, window boundaries, empty windows, and label leakage.
- A data-quality report with event counts, rejected records, missing values, window counts, feature availability, and label distribution where applicable.
- A reproducible command sequence in the README.

**Do not train Isolation Forest or implement the statistical anomaly detector in this milestone.** That belongs to the next milestone, after the input data is trustworthy.

---

# Carefully scoped implementation prompt

Copy the prompt below into your coding assistant or development agent. It is designed to implement only Milestone 1.

## PROMPT START

You are a senior machine-learning engineer building the data foundation for a standalone log anomaly-detection component.

### 1. Project context and boundaries

This component belongs to an Automated Log Anomaly Detector project. Your scope is **Milestone 1: structured-event ingestion, data validation, time-window aggregation, feature generation, synthetic dataset generation, public-dataset preparation, and tests**.

A different teammate owns reasoning, evidence correlation, root-cause hypotheses, incident explanations, APIs, the dashboard, and cloud deployment.

Do not implement any of those systems. Do not implement a trained anomaly detector in this milestone. Do not implement Isolation Forest, clustering, statistical anomaly scoring, model training, or model inference. We will develop and evaluate detectors only after the data foundation is tested.

Start in a new folder named `ml-anomaly-detector/`. Before creating files, inspect the current workspace so you do not overwrite existing work. If the workspace already contains relevant files, explain what you found and adapt without deleting them.

### 2. First, plan; then implement

Before coding:

1. Explain the proposed architecture and responsibilities of each module.
2. Define the event schema, feature-window schema, label representation, and error-handling policy.
3. State the assumptions and configurable defaults.
4. Identify likely data-leakage risks and how tests will prevent them.
5. List the acceptance criteria.
6. Then implement only the milestone described below. Do not stop at a plan or produce placeholder files instead of working code.

Do not ask questions for routine choices. Use reasonable, documented defaults. If an external dataset cannot be accessed, continue with a clean adapter interface and synthetic fixtures, and document the blocker honestly.

### 3. Suggested repository structure

Use a small, maintainable structure similar to:

```text
ml-anomaly-detector/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── src/
│   └── anomaly_detector/
│       ├── __init__.py
│       ├── schemas.py
│       ├── validation.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── synthetic_data.py
│       ├── dataset_adapters/
│       │   ├── __init__.py
│       │   └── base.py
│       └── reporting.py
├── scripts/
│   ├── generate_synthetic_data.py
│   └── prepare_dataset.py
├── tests/
│   ├── test_schemas.py
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_synthetic_data.py
│   └── test_no_label_leakage.py
└── artifacts/
    └── .gitkeep
```

Adjust this structure if there is a clear engineering reason, but avoid unnecessary abstractions. Do not create notebooks unless needed for this milestone. Keep generated data and artifacts out of version control by default, except for small intentional fixtures.

Use a supported Python version and current stable, compatible package versions. Keep dependencies minimal. Use standard typing, clear docstrings, and actionable error messages.

### 4. Common event schema

Create a documented, validated representation for a structured event with these conceptual fields:

- `event_id`: optional stable event identifier.
- `timestamp`: required parseable timestamp; normalize to UTC internally.
- `host_id`: required non-empty identifier.
- `service`: required non-empty identifier.
- `event_type`: required non-empty normalized category.
- `level`: optional normalized severity.
- `attributes`: optional mapping of event-specific structured values.
- Optional source metadata only when justified and documented.

Use a schema validation approach appropriate to the dependency budget. Be consistent about timestamp parsing and timezone behavior. Reject or quarantine invalid records with an explicit reason. Do not silently coerce arbitrary malformed values into valid-looking data.

Define an event contract that is easy to map from Linux, Windows, and application logs later. Do not build OS-specific raw-log parsers in this milestone.

Do not collect or log passwords, authentication tokens, secrets, or unnecessary personal data.

### 5. Validation and preprocessing

Implement functions with clear inputs and outputs for:

- Parsing and normalizing events.
- Validating required fields and value types.
- Tracking accepted, rejected, duplicated, and partially usable records.
- Deduplicating only when a reliable event identity exists. Do not merge distinct events merely because their timestamps and messages happen to match.
- Handling out-of-order timestamps.
- Handling missing optional attributes.
- Reporting invalid measurements such as negative latency.

Make the duplicate-handling policy explicit and test it. Do not treat missing telemetry as zero activity. Preserve missingness information when it matters to the feature output.

### 6. Time-window aggregation

Aggregate normalized events by host, service, and a configurable time window. Use one minute as the initial default, but make the interval configurable.

Document window-boundary semantics, timestamp conventions, and how late events are handled in this offline pipeline. Do not invent observations for empty windows. If empty windows are materialized, make that behavior explicit and distinguish them from absent telemetry.

Keep metadata needed to trace a feature window back to its host, service, and interval. Do not pass these metadata fields into a future model feature matrix by accident.

### 7. Feature generation

Implement an initial, documented feature set where source data supports it:

- event count;
- error count and error ratio;
- HTTP request count;
- HTTP 5xx count and 5xx ratio;
- authentication-failure count;
- service-restart count;
- mean latency and p95 latency when valid latency samples exist;
- valid-sample counts or availability indicators where needed.

Use consistent feature names and types. Define which event types and levels contribute to each feature. Do not assume every event is an HTTP request. Do not compute HTTP ratios from unrelated events.

Avoid division-by-zero and ambiguous zero substitution. Represent undefined or unavailable measurements explicitly, using missing values and/or availability indicators with documented semantics. Preserve sample counts so downstream modelers can distinguish a ratio based on a few observations from one based on many observations.

Do not use `host_id`, `service`, raw timestamps, scenario identifiers, or ground-truth anomaly labels as numeric model features. Keep window metadata separate from the numerical feature matrix.

### 8. Synthetic dataset generator

Build a reproducible generator with a configurable random seed and configurable dataset size/duration. It should produce structured events and feature windows for normal and anomalous scenarios.

Include at least these scenarios:

Normal:
- stable traffic with natural variation;
- occasional isolated errors;
- low-traffic periods;
- expected service restarts;
- missing or delayed observations.

Anomalous:
- HTTP 5xx surge;
- authentication-failure surge;
- repeated service restarts;
- latency increase;
- abnormal event-volume increase.

Scenario intensity and duration should be configurable where practical. Avoid making every anomaly trivially obvious or every normal window perfectly constant. Keep the first version simple and deterministic enough to test.

Maintain ground-truth labels in a separate table or metadata structure with a clear relationship to the affected time windows. Never include the ground-truth label, scenario name, or generator-only marker in the model feature matrix. Add tests that fail if labels leak into the feature columns.

Document that synthetic performance does not establish real-world model quality.

### 9. Public-dataset preparation

Define a minimal dataset-adapter interface and implement a first adapter only if a suitable public dataset is available and its licensing, schema, and labels can be verified.

Record the dataset name, official source, license, fields, label semantics, mapping rules, limitations, and any lossy conversions in documentation. Do not silently relabel source data. Do not claim that data was downloaded, inspected, or validated unless that actually happened.

If internet access, licensing clarity, or file availability prevents a reliable adapter in this environment, provide:
- the adapter interface;
- a small synthetic fixture that tests the interface;
- a documented candidate-dataset decision and exact blocker;
- clear instructions for adding the dataset later.

Do not make the whole milestone depend on external downloads.

### 10. Data-quality reporting and exports

Produce a report containing, where applicable:

- input record count;
- accepted and rejected record counts;
- rejection reasons;
- duplicate count;
- missingness by important field;
- number of hosts and services;
- number of feature windows;
- feature availability and missingness;
- label distribution for labeled synthetic data;
- time range and configured window size.

Support a straightforward, documented export format such as CSV or Parquet. Prefer CSV for a minimal dependency footprint unless Parquet offers a clear advantage and the dependency is justified. Preserve metadata and labels separately from the model feature matrix.

### 11. Testing requirements

Write automated tests for:

- valid and invalid events;
- timezone normalization;
- missing required fields;
- invalid numeric measurements;
- duplicate handling;
- out-of-order events;
- time-window boundaries;
- empty and low-volume windows;
- ratios with zero denominators;
- unavailable features and missing data;
- deterministic synthetic generation with the same seed;
- presence of all required scenarios;
- separation of labels from features;
- output schema stability;
- data-quality report counts.

Use small fixtures for unit tests. Tests must not require internet access. Never rely on a public dataset download for the basic test suite.

### 12. Reproducibility and usability

The README must explain:

- what this component does and does not do;
- prerequisites and environment setup;
- how to generate synthetic data;
- how to run validation, feature generation, and reporting;
- where outputs are written;
- the event and feature schemas;
- how missing values and duplicates are handled;
- how to run tests;
- how to add a public dataset adapter;
- current limitations and the next milestone.

Provide exact commands that work from a clean checkout. Include a small end-to-end command path that generates data, validates it, creates feature windows, exports results, and produces a report.

Use configuration for meaningful defaults such as window size, random seed, output paths, and scenario duration. Avoid hard-coded machine-specific paths.

### 13. Security and repository hygiene

- Add generated data, local environment directories, caches, and large artifacts to `.gitignore`.
- Do not commit secrets or real sensitive logs.
- Use small synthetic fixtures in tests.
- Do not add an API, web server, dashboard, cloud SDK, external LLM, or AWS deployment code.
- Do not add model-training or anomaly-scoring code in this milestone.

### 14. Acceptance criteria

The milestone is complete only when:

1. The event schema and policies are documented.
2. Structured events can be validated and normalized.
3. Events can be aggregated into configurable host/service windows.
4. The documented feature set is generated correctly.
5. Synthetic data generation is reproducible and includes the required scenarios.
6. Ground-truth labels are separate from model features, with tests for leakage.
7. Data-quality reporting and exports work.
8. A public-dataset adapter path is documented, with a verified first adapter or an honest blocker and tested interface.
9. Unit tests run offline and pass.
10. README commands work from a clean environment.
11. No anomaly model, reasoning engine, correlation engine, API, dashboard, or AWS code has been added.

### 15. Final response after implementation

When finished, report:

- the files created and the responsibility of each module;
- the event and feature contracts;
- the commands to set up, run, and test the pipeline;
- the tests actually run and their actual results;
- the synthetic scenarios generated;
- the public dataset used or the exact blocker;
- any known limitations;
- the evidence that labels are excluded from model features;
- the precise boundary between this milestone and the next one.

Do not claim a test passed unless you ran it. Do not claim a dataset was accessed unless it was. Do not invent performance metrics. If something fails, explain the failure and fix it where possible.

**Implement Milestone 1 only. Stop after the data pipeline is tested and documented.**

## PROMPT END

---

## Milestone 1 completion checklist

- [ ] Common event schema is documented.
- [ ] Validation and normalization are implemented.
- [ ] Window aggregation is implemented and configurable.
- [ ] Initial numerical features are generated correctly.
- [ ] Synthetic normal/anomalous scenarios are reproducible.
- [ ] Ground-truth labels are separate from feature columns.
- [ ] Data-quality report and export work.
- [ ] Public dataset adapter path is documented.
- [ ] Offline tests pass.
- [ ] README contains reproducible commands.
- [ ] No model training, anomaly scoring, reasoning, correlation, API, dashboard, or AWS code exists in this milestone.

**Next milestone:** establish a statistical baseline and train/evaluate Isolation Forest using the validated feature windows. Do not start that until this milestone's data and tests are trustworthy.
