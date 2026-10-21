# ML Anomaly Detector — Data Pipeline, Experiment Readiness & Baseline (Milestones 1–3)

A standalone **data-preparation and baseline-detection** component for an Automated
Log Anomaly Detector. It turns structured log events into trustworthy numerical feature
windows (Milestone 1), validates that data, audits it for leakage and produces
reproducible, aligned, training-ready datasets (Milestone 2), and then trains, scores
and evaluates a first **Isolation Forest** baseline over those validated windows
(Milestone 3).

> **Scope:** Milestones 1–3 cover data work and **detection/scoring only**. There is
> **no** evidence correlation, root-cause reasoning, incident narrative, API,
> dashboard or cloud code here. A flagged window is a lead for investigation, not
> proof of an attack — that reasoning layer belongs to a separate workstream.

- **Milestone 1** — data pipeline foundation (ingest → validate → window → features
  → synthetic data → public-dataset adapter path).
- **Milestone 2** — data validation, leakage audit, dataset splits and experiment
  readiness ([`docs/DATA_READINESS.md`](docs/DATA_READINESS.md)).
- **Milestone 3** — first baseline detector: Isolation Forest, thresholding,
  evaluation and artifacts ([`docs/BASELINE.md`](docs/BASELINE.md)).

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
- **(M2)** Profile every dataset column and validate it against an explicit policy,
  **never mutating** the source data.
- **(M2)** Run a **leakage audit** that classifies every column (feature / label /
  identifier / timestamp / scenario / future-derived) and refuses to export when a
  forbidden column reaches the feature matrix.
- **(M2)** Produce **reproducible random, time-aware and group-aware splits** with
  integrity checks (coverage, no overlap, chronological order, group disjointness).
- **(M2)** Export an **aligned training-ready dataset** (features / labels / metadata
  as separate tables) with a manifest, checksums and alignment verification.
- **(M3)** Train and persist an **Isolation Forest baseline** over those validated
  windows, with an explicit model-input contract (allowlist only), a documented
  missing-value policy, a train/validation/test discipline and a reproducible
  threshold.
- **(M3)** Produce a **continuous anomaly score** plus a boolean flag per sample, and
  evaluate it **per split** with precision/recall/F1, ROC-AUC, PR-AUC, false-positive
  cost and alert rate — skipping any metric that cannot be computed, with a reason.
- **(M3)** Write a **model manifest** (library version, seed, feature schema, data
  policy, split provenance, score direction, threshold method) so any run can be
  audited and reproduced.

**Does not (belongs to a later milestone or teammate)**

- No statistical baseline comparison, no hyperparameter search, no model comparison.
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
        │
        ▼  M2
 audit.py / leakage.py / splits.py / training_ready.py
        │              ──► data/processed/splits/<strategy>/{train,validation,test}/
        ▼
 baseline_model.py  IsolationForest  ──► raw_score (lower = more anomalous)
        │              thresholding.py ──► is_anomaly
        ▼
 evaluation.py      per-split metrics  ──► reports/metrics.{json,md}
        ▼
 artifacts.py       model.joblib + manifest.json + predictions_<split>.csv
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
| `src/anomaly_detector/readiness_config.py` | **M2** validated readiness config loader. |
| `src/anomaly_detector/roles.py` | **M2** feature-role classification + leakage-risk detection. |
| `src/anomaly_detector/dataset.py` | **M2** table loading + stable sample-key alignment. |
| `src/anomaly_detector/audit.py` | **M2** read-only dataset audit (JSON + Markdown). |
| `src/anomaly_detector/leakage.py` | **M2** leakage audit + `assert_no_leakage` guard. |
| `src/anomaly_detector/splits.py` | **M2** random / time / group splits with checks. |
| `src/anomaly_detector/training_ready.py` | **M2** aligned training-ready export + manifest. |
| `scripts/audit_dataset.py` | **M2** run the quality + leakage audit. |
| `scripts/export_training_ready.py` | **M2** export the training-ready dataset. |
| `scripts/make_splits.py` | **M2** create reproducible splits. |
| `scripts/prepare_experiments.py` | **M2** one-shot audit + export + splits. |
| `docs/DATA_READINESS.md` | **M2** data-readiness concepts, config and commands. |
| `src/anomaly_detector/baseline_config.py` | **M3** validated baseline config loader (allowlist, policies, paths). |
| `src/anomaly_detector/model_contract.py` | **M3** model-input contract: schema, missing-value policy, non-finite rejection. |
| `src/anomaly_detector/baseline_model.py` | **M3** Isolation Forest wrapper: fit/score/save/load with an explicit contract. |
| `src/anomaly_detector/thresholding.py` | **M3** threshold selection (validation-F1 / precision / recall / quantile) + fallback. |
| `src/anomaly_detector/evaluation.py` | **M3** per-split metrics with explicit skip reasons + reference baselines. |
| `src/anomaly_detector/artifacts.py` | **M3** manifest, predictions, metrics and experiment-note writers. |
| `src/anomaly_detector/baseline_pipeline.py` | **M3** end-to-end wiring: train → threshold → score → evaluate → artifacts. |
| `scripts/run_baseline.py` | **M3** CLI: `run` / `train` / `score` / `evaluate`. |
| `docs/BASELINE.md` | **M3** baseline concepts, contract, commands and limitations. |

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

# --- Milestone 2: experiment readiness -------------------------------------
# (6) One-shot: quality + leakage audit, training-ready export, all splits
python scripts/prepare_experiments.py

#    ...or run each step individually:
python scripts/audit_dataset.py                       # reports/data_quality_report.{json,md} + leakage_audit.{json,md}
python scripts/export_training_ready.py               # data/processed/training_ready/
python scripts/make_splits.py --strategy random
python scripts/make_splits.py --strategy time
python scripts/make_splits.py --strategy group --group-column scenario_name

# --- Milestone 3: baseline detector ----------------------------------------
# (7) One-shot: train → score every split → evaluate → write all artifacts
python scripts/run_baseline.py run

#    ...or run each step individually:
python scripts/run_baseline.py train                  # artifacts/baseline/model/{model.joblib,manifest.json}
python scripts/run_baseline.py score --split test     # artifacts/baseline/predictions_test.csv
python scripts/run_baseline.py evaluate --split test   # artifacts/baseline/reports/metrics.{json,md}
```

Outputs:

| File | Description |
|---|---|
| `data/synthetic/synthetic_events.json` | Raw synthetic events |
| `data/synthetic/synthetic_labels.json` | Ground-truth labels (separate) |
| `data/processed/features.csv` | Numerical feature matrix |
| `data/processed/metadata.csv` | Window metadata (host, service, time) |
| `data/processed/labels.csv` | Ground-truth labels |
| `artifacts/data_quality_report.json` | Data-quality report (Milestone 1) |
| `artifacts/dataset_adapter_status.json` | Adapter decision/status |
| `reports/data_quality_report.{json,md}` | **M2** dataset audit |
| `reports/leakage_audit.{json,md}` | **M2** leakage audit |
| `data/processed/training_ready/` | **M2** aligned features/labels/metadata + `manifest.json` |
| `data/processed/splits/<strategy>/` | **M2** per-split tables + `split_manifest.json` |
| `artifacts/baseline/model/model.joblib` | **M3** fitted Isolation Forest + its input contract |
| `artifacts/baseline/model/manifest.json` | **M3** model manifest (version, seed, schema, policy, threshold) |
| `artifacts/baseline/predictions_<split>.csv` | **M3** per-sample score, threshold and flag |
| `artifacts/baseline/reports/metrics.{json,md}` | **M3** per-split evaluation |
| `artifacts/baseline/experiment_note.md` | **M3** limitations + reproduction commands |

Configuration defaults live in `configs/default.yaml` (window size, seed, output
paths, empty-window behaviour), `configs/data_readiness.yaml` (Milestone 2 roles,
policy, splits and export paths) and `configs/baseline.yaml` (Milestone 3 model
parameters, feature allowlist, data policies, threshold rule and paths). All paths
are relative to the repo root.

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

## 8b. Milestone 2 — data validation, leakage audit, splits, readiness

Full details live in [`docs/DATA_READINESS.md`](docs/DATA_READINESS.md). Summary:

- **Dataset audit** (`reports/data_quality_report.{json,md}`) — per-column profile
  (dtype, missingness, infinities, cardinality, constant/near-constant), duplicate
  rows, duplicate sample keys, and policy checks (non-negative counts, unit-interval
  ratios, integer counts). **Read-only** — the source tables are never mutated.
- **Leakage audit** (`reports/leakage_audit.{json,md}`) — classifies every column by
  role and *fails* if a label, identifier, timestamp, scenario or future-derived
  column reaches the feature matrix. Eligibility is decided by an **explicit
  allowlist**, never by "is it numeric".
- **Training-ready export** (`data/processed/training_ready/`) — separate
  `features.csv` / `labels.csv` / `metadata.csv` + `sample_index.csv` + `manifest.json`
  (seed, sizes, feature list, leakage status, SHA-256 checksums). Row alignment is
  verified and a mismatch fails loudly.
- **Reproducible splits** (`data/processed/splits/<strategy>/`) — `random`, `time`
  and `group` strategies. Same config + seed ⇒ identical assignment. Integrity
  checks (coverage, no double assignment, chronological order, group disjointness)
  are recorded; single-class or empty splits are warned about.

Key policy decisions: **missing ≠ zero** and missing values are *reported, never
imputed*; nothing is silently dropped or coerced; the label table is never used as
model input.

---

## 8c. Milestone 3 — the Isolation Forest baseline

Full details live in [`docs/BASELINE.md`](docs/BASELINE.md). Summary:

- **Model-input contract** — only the columns on the explicit allowlist in
  `configs/baseline.yaml` ever reach the estimator. Labels, identifiers, timestamps
  and scenario metadata are rejected at config-load time.
- **Missing-value policy** — `median_impute` (default) uses the **training-split
  median**, fitted on training rows only; `drop` excludes samples under a documented
  rule and reports them; `error` fails loudly. Missing is **never** zero-filled, and
  infinities are always rejected.
- **Training discipline** — the estimator is fitted on training rows only
  (`normal_only` by default); the threshold is selected on the **validation** split
  only; the **test** split is refused as a threshold source.
- **Score contract** — `raw_score = IsolationForest.score_samples`, **lower = more
  anomalous**, `is_anomaly = raw_score <= threshold` (inclusive). It is a ranking
  statistic, not a probability.
- **Evaluation** — precision / recall / F1 / ROC-AUC / PR-AUC / false-positive cost /
  alert rate **per split**, with a stated reason for every skipped metric and trivial
  reference baselines (`no_alert`, `train_quantile`) for context.
- **Artifacts** — `model.joblib` (with its contract), `manifest.json`,
  `predictions_<split>.csv`, `metrics.{json,md}` and an `experiment_note.md` with the
  limitations and the exact reproduction commands.

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
data-quality report counts, and the dataset-adapter interface. Milestone 2 adds
coverage for role classification, alignment, the audit (including injected errors),
the leakage audit (including injected label/identifier columns), split
reproducibility and integrity, and the training-ready export contract. Milestone 3
adds coverage for the config guardrails, the schema/missing-value/non-finite
contract, the score direction and threshold boundary, the test-split refusal, the
documented fallback, metric edge cases (single-class splits, missing labels), the
train/validation/test discipline (poisoning the test split changes nothing) and the
CLI error paths.

---

## 10. Repository structure

```text
ml-anomaly-detector/
├── README.md
├── MILESTONE_1_REPORT.md
├── MILESTONE_2_REPORT.md
├── MILESTONE_3_REPORT.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/{default.yaml, data_readiness.yaml, baseline.yaml}
├── docs/{DATASETS.md, DATA_READINESS.md, BASELINE.md}
├── data/{raw,processed,synthetic}/   (git-ignored, keep .gitkeep)
├── reports/                          (git-ignored, keep .gitkeep)
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
│   ├── readiness_config.py        # M2
│   ├── roles.py                   # M2
│   ├── dataset.py                 # M2
│   ├── audit.py                   # M2
│   ├── leakage.py                 # M2
│   ├── splits.py                  # M2
│   ├── training_ready.py          # M2
│   ├── baseline_config.py         # M3
│   ├── model_contract.py          # M3
│   ├── baseline_model.py          # M3
│   ├── thresholding.py            # M3
│   ├── evaluation.py              # M3
│   ├── artifacts.py               # M3
│   ├── baseline_pipeline.py       # M3
│   └── dataset_adapters/{__init__.py, base.py, loghub_hdfs.py}
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_pipeline.py
│   ├── prepare_dataset.py
│   ├── audit_dataset.py           # M2
│   ├── export_training_ready.py   # M2
│   ├── make_splits.py             # M2
│   ├── prepare_experiments.py     # M2
│   └── run_baseline.py            # M3
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
- No hyperparameter search or model comparison yet; the Isolation Forest parameters
  are fixed configuration values.
- The anomaly score is a ranking statistic, **not** a probability.
- Detection delay is not measured — the label contract has no anomaly onset time.
- `model.joblib` is pickle-based; load it only from a trusted source.

## 12. Boundary with the next milestone

Milestones 1–3 stop here with a tested, documented data pipeline, a leakage-audited,
reproducible, training-ready dataset (features / labels / metadata kept separate,
splits produced under three strategies) and a reproducible baseline detector with a
stable output contract. The next milestone can add a statistical baseline (e.g.
robust z-score / EWMA over the same features), hyperparameter search and a model
comparison — all of which must reuse the *same* contract: the same allowlist, the same
missing-value policy, the same score direction and the same train/validation/test
discipline.

**Note for the next milestone:** the synthetic data is generated per-window; group
splits by `scenario_name` can place all anomalies in one fold (the tooling warns but
does not hide this). Choose the split strategy deliberately and consider a
scenario-mixed or time-based split for detection experiments. The `random` strategy is
the one used for the headline baseline numbers.
