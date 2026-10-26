# ML Anomaly Detector — Data Pipeline, Baseline Detectors, Inference & Incident Drafts (Milestones 1–10)

A standalone **data-preparation and anomaly-detection** component for an Automated
Log Anomaly Detector. It turns structured log events into trustworthy numerical
feature windows (Milestone 1), validates that data, audits it for leakage and
produces reproducible, aligned, training-ready datasets (Milestone 2), trains and
evaluates an **Isolation Forest** baseline over those windows (Milestones 3–5),
exposes a stable in-process inference interface (Milestones 6–8), adds a separate
**network/service-log detector** with its own leak-safe feature schema and
deterministic local simulator (Milestone 9), and (Milestone 10) connects scored
results to the upstream **rule-based evidence correlator** and produces
schema-valid **incident drafts** with an optional grounded explanation layer.

> **Scope:** this repository covers data preparation, **detection/scoring** and
> the **incident-draft boundary**. Correlation is deterministic (service + time
> proximity); it is association, not proof of root cause. There is **no** LLM in
> the upstream correlator — the only LLM layer is the optional, disabled-by-
> default explanation adapter. There is no API, dashboard, alert delivery or
> cloud code here; those belong to the other LAD workstreams.

- **Milestone 1** — data pipeline foundation (ingest → validate → window → features
  → synthetic data → public-dataset adapter path).
- **Milestone 2** — data validation, leakage audit, dataset splits and experiment
  readiness ([`docs/DATA_READINESS.md`](docs/DATA_READINESS.md)).
- **Milestone 4** — model comparison and improvement: MAD baseline, LOF (novelty) comparison against M3, predeclared criteria, validation-only selection, unique M4 artifacts ([`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md), [`MILESTONE_4_REPORT.md`](MILESTONE_4_REPORT.md)).
- **Milestone 5** — robustness, drift, and realistic validation: scenario testing with selected M3 model (no retraining), false-alert/miss analysis ([`MILESTONE_5_REPORT.md`](MILESTONE_5_REPORT.md)).
- **Milestone 6** — stable ML inference contract and integration boundary: a versioned, callable interface (feature window in → structured result out) for the downstream correlation/explanation teammate ([`docs/ml_inference_contract.md`](docs/ml_inference_contract.md), [`docs/ML_HANDOFF.md`](docs/ML_HANDOFF.md), [`MILESTONE_6_REPORT.md`](MILESTONE_6_REPORT.md)).
- **Milestone 7** — ML readiness audit and end-to-end local validation: root-caused and fixed the M4 test-leakage in the auto-picker, re-decided the model selection on **validation evidence only**, audited the threshold and abstention policy, and drove the real M1→M6 pipeline end to end ([`docs/READINESS_AUDIT.md`](docs/READINESS_AUDIT.md), [`MILESTONE_7_REPORT.md`](MILESTONE_7_REPORT.md)). **Verdict: `controlled demo only`.**
- **Milestone 8** — integration boundary preparation: a **centralised** forwarding rule, an explicit unscored-result handler, explicit sink-failure handling, and a local integration harness + offline rehearsal against the real M1→M6 path ([`docs/INTEGRATION_BOUNDARY.md`](docs/INTEGRATION_BOUNDARY.md), [`MILESTONE_8_REPORT.md`](MILESTONE_8_REPORT.md)).
- **Milestone 9 — network/service-log extension** — an additive, integration-ready detector for HTTP 5xx spikes, DB timeouts/latency, auth-failure bursts and benign operational shifts: a deterministic local LAD-envelope simulator with a separate label channel, a leak-safe `network-features-v1` schema (39 columns), a transparent rule/MAD baseline, an Isolation Forest, time and group splits, status-aware evaluation, and an optional evidence-grounded explanation adapter ([`docs/ml_network_reconnaissance.md`](docs/ml_network_reconnaissance.md), [`docs/ml_network_features.md`](docs/ml_network_features.md), [`docs/ml_network_evaluation.md`](docs/ml_network_evaluation.md), [`docs/ml_network_integration.md`](docs/ml_network_integration.md)).
- **Milestone 10 — evidence correlation / incident drafts** — vendors the real upstream `correlate_anomalies()` implementation, maps its groups into the canonical LAD incident schema, preserves every ML score/status/timestamp/model version, validates and redacts event references, and reuses the optional grounded explanation adapter ([`docs/ml_llm_integration_reconnaissance.md`](docs/ml_llm_integration_reconnaissance.md), [`docs/ml_llm_integration.md`](docs/ml_llm_integration.md)). **The upstream repository contains no LLM**; this is documented rather than fabricated.

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
- **(M4)** Compare a small, justified candidate set (robust MAD, LOF in novelty mode)
  against the M3 baseline using **pre-declared criteria**, selecting and thresholding
  on the **validation split only** and evaluating test once, with unique M4 artifacts.
- **(M5)** Validate the selected detector against **controlled scenarios** (normal
  variation, service anomalies, telemetry faults, low-volume boundaries, distribution
  shifts) **without retraining**, and report the false-alert and missed-detection
  failure modes.
- **(M6)** Expose a **stable, versioned inference interface**: one aggregated feature
  window in, one structured result out (`scored` / `insufficient_data` /
  `invalid_input` / `model_unavailable` / `inference_error`), with explicit feature
  order, preprocessing, score direction, threshold and model versions, and a
  documented integration boundary for the downstream correlation component.
- **(M7)** **Audit** the component for local-integration readiness: trace and fix the
  M4 auto-picker that selected a model using **test-set metrics**, re-decide the
  selection on **validation evidence only** under a pre-declared policy, re-derive
  the threshold from validation data, audit the abstention/missingness behaviour, and
  drive the **real** raw-events → features → inference → downstream-sink path with
  deterministic scenario tests.
- **(M8)** **Harden the integration boundary** for the downstream teammate: centralise
  the forwarding rule so only a `scored` result can be forwarded, handle every unscored
  status through an explicit handler, make sink failures explicit instead of silently
  aborting a batch, and provide a local harness plus an offline rehearsal that
  separates contract-routing success from ML detection success.
- **(M9)** **Detect network/service-log incidents** with an additive extension: a
  deterministic local simulator that emits canonical LAD envelopes plus a separate
  label channel, a leak-safe `network-features-v1` schema with causal rolling features,
  a transparent rule/MAD baseline and an Isolation Forest, chronological and
  run-disjoint splits, status-aware evaluation, and an optional evidence-grounded
  explanation adapter. The existing `features-v1` contract is unchanged.
- **(M10)** **Correlate scored anomalies and draft incidents** by reusing the real
  upstream `correlate_anomalies()` implementation, validating every ML result
  through the canonical routing rule, redacting event references, preserving
  scores/timestamps/model metadata, and producing drafts that validate against
  `LAD/contracts/schemas/incident.schema.json`. Explanations remain optional.

**Does not (belongs to a later milestone or teammate)**

- No LLM-based correlation. Upstream has no LLM; correlation is deterministic
  service + time proximity. The optional explanation adapter only restates scored
  evidence, is disabled by default and never changes a decision.
- No root-cause proof from correlation. A temporal/identity group is association,
  not causation, and incidents say so explicitly.
- No API, web server, dashboard, cloud SDK, or AWS code. The explanation adapter
  makes no external call unless explicitly configured.
- No incident persistence or acknowledgement/resolution lifecycle.
- No alert delivery, response automation, or remediation.
- No automatic online learning, retraining, or threshold adjustment at inference time.
- No production-readiness or SLA claim: the M7 verdict is `controlled demo only`.

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
        │
        ▼  M6
 inference/         load_detector() / score_window() / score_windows()
                    ──► InferenceResult (status, score, decision, provenance)
                         │
                         ▼
 inference/adapter.py  ──► downstream boundary (evidence correlation)
        │
        ▼  M7
 readiness_audit.py    validation-only selection → threshold audit → verdict
 e2e_runner.py         raw events → M1 → M6 → neutral sink ──► readiness report
        │
        ▼  M8
 inference/contract.py route_result()  ── the one forwarding rule
 integration_harness.py ──► correlation sink | unscored handler
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
| `src/anomaly_detector/mad_baseline.py` | **M4** Robust MAD baseline (training-fitted). |
| `src/anomaly_detector/lof_baseline.py` | **M4** LOF novelty baseline (scaler fitted on training only). |
| `src/anomaly_detector/comparison_config.py` | **M4** Validated comparison config with predeclared criteria. |
| `src/anomaly_detector/comparison.py` | **M4** Comparison orchestration (validation-only selection). |
| `scripts/run_comparison.py` | **M4** Run the comparison. |
| `docs/MODEL_COMPARISON.md` | **M4** Comparison design and usage. |
| `src/anomaly_detector/robustness_scenarios.py` | **M5** deterministic robustness scenario generators (labels never enter features). |
| `src/anomaly_detector/robustness.py` | **M5** robustness validation + failure analysis over the selected model. |
| `scripts/run_robustness.py` | **M5** Run the robustness validation. |
| `src/anomaly_detector/inference/contract.py` | **M6** versioned inference contract: request/result types, statuses, reason codes. |
| `src/anomaly_detector/inference/detector.py` | **M6** `load_detector` / `score_window` / `score_windows` (no retraining, no fitting on production data). |
| `src/anomaly_detector/inference/adapter.py` | **M6** downstream integration boundary (`ScoredWindow`, `DownstreamSink`, adapter). |
| `scripts/smoke_test_inference.py` | **M6** CLI smoke test for the inference interface. |
| `docs/ml_inference_contract.md` | **M6** normative input/output contract and behaviour matrix. |
| `docs/ML_HANDOFF.md` | **M6** handoff + integration checklist for the downstream teammate. |
| `src/anomaly_detector/selection_policy.py` | **M7** pre-declared validation-only selection policy (rejects test metrics structurally). |
| `src/anomaly_detector/readiness_audit.py` | **M7** fair comparison, threshold/abstention audit, readiness verdict, reports. |
| `src/anomaly_detector/e2e_scenarios.py` | **M7** raw-event end-to-end scenarios (ground truth kept out of features). |
| `src/anomaly_detector/e2e_runner.py` | **M7** drives the real M1 → M6 path into a neutral downstream sink. |
| `scripts/run_readiness_audit.py` | **M7** Run the readiness audit. |
| `docs/READINESS_AUDIT.md` | **M7** Audit design, M4 root cause, threshold findings, readiness verdict. |
| `src/anomaly_detector/inference/contract.py` | **M8** also holds `route_result()` — the canonical forwarding rule. |
| `src/anomaly_detector/integration_harness.py` | **M8** local harness: centralised routing, correlation sink, unscored handler, rehearsal. |
| `scripts/run_integration_rehearsal.py` | **M8** Run the offline integration rehearsal. |
| `docs/INTEGRATION_BOUNDARY.md` | **M8** Boundary design, ownership split, and what the teammate must still provide. |

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

# --- Milestone 4: model comparison ------------------------------------------
# (8) Compare candidates against M3 with predeclared criteria
python scripts/run_comparison.py --config configs/comparison.yaml

# --- Milestone 5: robustness validation --------------------------------------
# (9) Score controlled scenarios with the selected model (no retraining)
python scripts/run_robustness.py

# --- Milestone 6: inference interface ----------------------------------------
# (10) Smoke-test the stable inference interface
python scripts/smoke_test_inference.py
python scripts/smoke_test_inference.py --all

# --- Milestone 7: readiness audit ---------------------------------------------
# (11) Validation-only selection, threshold/abstention audit, end-to-end pipeline
python scripts/run_readiness_audit.py

# --- Milestone 8: integration boundary ----------------------------------------
# (12) Offline rehearsal of the ML -> downstream boundary
python scripts/run_integration_rehearsal.py

# --- Milestone 9: network / service-log extension ------------------------------
# (13) Simulator only, or the full simulator -> features -> model -> evaluation run
python scripts/run_network_simulator.py --config configs/network.yaml
python scripts/run_network_pipeline.py --config configs/network.yaml
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
| `artifacts/comparison/run_*/reports/comparison_report.{json,md}` | **M4** comparison with predeclared criteria |
| `artifacts/comparison/run_*/comparison_manifest.json` | **M4** run metadata |
| `artifacts/comparison/run_*/predictions/*` | **M4** per-candidate predictions |
| `artifacts/robustness/run_*/reports/robustness_report.{json,md}` | **M5** per-scenario robustness + failure analysis |
| `artifacts/robustness/run_*/predictions/*` | **M5** per-scenario predictions |
| `artifacts/readiness/run_*/reports/readiness_audit.{json,md}` | **M7** readiness audit (selection, threshold, e2e, verdict) |
| `artifacts/readiness/run_*/readiness_manifest.json` | **M7** run metadata |
| `artifacts/rehearsal/run_*/reports/rehearsal_report.{json,md}` | **M8** integration rehearsal (routing counts, per-scenario results) |
| `artifacts/rehearsal/run_*/rehearsal_manifest.json` | **M8** rehearsal metadata |
| `data/synthetic/network/events.jsonl` | **M9** canonical LAD envelopes (synthetic) |
| `data/synthetic/network/labels.csv` | **M9** evaluation-only labels (separate channel) |
| `artifacts/network/features.csv` | **M9** 39-column `network-features-v1` matrix |
| `artifacts/network/metadata.csv`, `labels.csv` | **M9** window keys and aligned labels |
| `artifacts/network/lad_parse_summary.json` | **M9** accepted/rejected/quality counts |
| `artifacts/network/model/{model.joblib,manifest.json}` | **M9** deployed `time` model |
| `artifacts/network/experiments/<strategy>/` | **M9** per-strategy model, predictions, threshold, metrics |

The M6 inference interface produces **no artifacts of its own** — it is an
in-process interface over the M3 model artifact. Its contract lives in
[`docs/ml_inference_contract.md`](docs/ml_inference_contract.md).

Configuration defaults live in `configs/default.yaml` (window size, seed, output
paths, empty-window behaviour), `configs/data_readiness.yaml` (Milestone 2 roles,
policy, splits and export paths), `configs/baseline.yaml` (Milestone 3 model
parameters, feature allowlist, data policies, threshold rule and paths),
`configs/comparison.yaml` (Milestone 4 predeclared criteria and candidates),
`configs/robustness.yaml` (Milestone 5 scenarios), `configs/selection.yaml`
(Milestone 7 pre-declared validation-only selection policy) and
`configs/network.yaml` (Milestone 9 simulator, rules, model, threshold, split
strategies and the disabled-by-default explanation adapter). All paths are
relative to the repo root.

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

## 8d. Milestone 6 — the stable ML inference interface

Full details live in [`docs/ml_inference_contract.md`](docs/ml_inference_contract.md)
and the teammate-facing [`docs/ML_HANDOFF.md`](docs/ML_HANDOFF.md). Summary:

- **One window in, one versioned result out.** The input is an already-aggregated
  feature window from the existing pipeline (never raw logs, never a raw event).
  The output is an `InferenceResult` carrying the status, the raw score, the
  decision, the model/feature/threshold versions and safe window provenance.
- **Five explicit statuses** — `scored`, `insufficient_data`, `invalid_input`,
  `model_unavailable`, `inference_error`. A window that could not be scored is
  reported with `is_anomaly = null`; it is **never** reported as normal.
- **Feature order and schema are enforced.** The matrix is assembled *by name* from
  the model's allowlist; unknown feature names, wrong types, `±inf`, naive
  timestamps and impossible window boundaries are rejected with a reason code.
  Labels, scenario names, host/service names and IDs are never model features.
- **No retraining, no fitting on production data.** `load_detector()` loads the
  selected artifact and its manifest once, cross-checks them, and reuses the
  training-fitted imputation medians. Nothing is ever fitted at inference time.
- **Score semantics are preserved.** The score stays
  `IsolationForest.score_samples` (lower = more anomalous), the decision stays
  `raw_score <= threshold`, and the result explicitly says it is *not* a probability.
- **Batch semantics are per row.** One bad row never fails the batch; every row gets
  its own result. Duplicate `window_id`s within a batch are rejected.
- **Integration boundary only.** `inference/adapter.py` defines a neutral
  `DownstreamSink` protocol and a reference adapter that forwards *only* scored
  results. M8 itself adds no evidence correlation, explanation, alerting,
  remediation or web framework; M10 adds the evidence-correlation/draft boundary
  as a separate package (`anomaly_detector.correlation`).

```python
from anomaly_detector.inference import InferenceStatus, load_detector

detector = load_detector()                 # load once
result = detector.score_window(request)    # score one feature window

if result.status is InferenceStatus.SCORED:
    ...  # result.raw_score, result.is_anomaly, result.window_id
else:
    ...  # result.reason_code — never treat this as "normal"
```

---

## 8e. Milestone 9 — network / service-log anomaly detection

Full details live in
[`docs/ml_network_reconnaissance.md`](docs/ml_network_reconnaissance.md),
[`docs/ml_network_features.md`](docs/ml_network_features.md),
[`docs/ml_network_evaluation.md`](docs/ml_network_evaluation.md) and
[`docs/ml_network_integration.md`](docs/ml_network_integration.md). Summary:

- **Additive schema.** `network-features-v1` is a separate 39-column allowlist.
  The 13-column `features-v1` schema, its order and its artifacts are untouched.
  `CONTRACT_VERSION` moved to `1.1.0` with additive reason codes
  (`no_events`, `insufficient_coverage`, `telemetry_quality`) and
  `EventType.AUTH_SUCCESS`.
- **Deterministic local simulator.** 10 runs × 90 minutes × 3 streams. It emits
  canonical LAD `1.0` envelopes on one channel and evaluation labels on a
  separate channel. It makes no network request and performs no login.
- **Covers the required families.** HTTP 5xx bursts, DB timeouts/latency,
  auth-failure bursts, benign high traffic / latency / restart, and
  missing/malformed telemetry.
- **Leak-safe.** No labels, scenario names, run IDs, future values, raw
  identifiers, message text or PII enter the feature matrix. Rolling features
  use only prior windows of the same host/service. Imputation is fitted on the
  training split only.
- **Two split strategies.** `time` is the deployed default and is strictly
  chronological; `group` keeps whole runs disjoint as a cross-check. The
  threshold is selected on validation only.
- **Status-aware evaluation.** Empty or degraded windows return
  `insufficient_data` with `is_anomaly = None`; they are excluded from
  supervised metrics, never counted as normal.
- **Measured test results (time split).** Precision 0.7059, recall 0.9000,
  F1 0.7912, ROC-AUC 0.9674, PR-AUC 0.8326; benign-window FPR 0.0767. The
  transparent rule baseline reaches F1 0.7195. These are synthetic figures and
  do not establish real-world quality; see the evaluation document for the
  failure modes, including a 25% false-positive rate on the benign high-traffic
  scenario.
- **Optional explanation adapter.** Disabled by default, evidence-grounded,
  offline mock by default, bounded HTTP provider only when explicitly enabled.
  It never creates or upgrades a detection.

```bash
python scripts/run_network_simulator.py --config configs/network.yaml
python scripts/run_network_pipeline.py --config configs/network.yaml
```

---

## 8f. Milestone 10 — evidence correlation and incident drafts

Full details live in
[`docs/ml_llm_integration_reconnaissance.md`](docs/ml_llm_integration_reconnaissance.md)
and [`docs/ml_llm_integration.md`](docs/ml_llm_integration.md). Summary:

- **Real upstream correlation, not a reimplementation.** `correlate_anomalies()`
  from `kalmak889900/automated-log-anomaly-detector@b155a05` is vendored verbatim
  with provenance. It groups by service and time proximity.
- **No upstream LLM exists.** The integration prompt's premise was checked against
  every upstream module; there is no prompt, provider, redaction or LLM test. The
  optional explanation layer is this repository's existing grounded adapter.
- **Canonical incident output.** Upstream groups are mapped into the LAD
  `incident.schema.json` shape: typed `evidence`, separate `hypotheses`,
  non-executing `diagnostic_checks`, `affected_entities`, model versions and
  audit. The upstream incident shape is deliberately not reused.
- **Explicit statuses.** Only `scored` results can become candidates. `unscored`,
  invalid, error, missing, duplicate and scored-but-not-flagged results are
  preserved in separate outcome lists with their original reason codes.
- **Redacted and bounded evidence.** Event IDs must be opaque and safe; optional
  redacted pointers are validated; unsafe references quarantine the candidate;
  all lists are bounded and truncation is recorded. Raw messages, headers,
  cookies, tokens, emails and IP addresses are never accepted.
- **LLM failure cannot erase an ML result.** Incidents are built from
  deterministic evidence first; provider timeouts, 429s, malformed JSON and
  grounding violations are recorded on the explanation entry.
- **Validated against the real schema.** Tests load a byte-for-byte copy of the
  canonical incident schema (hash recorded) and validate every draft.

```bash
python scripts/run_correlation_rehearsal.py
python scripts/run_correlation_rehearsal.py --explain --limit 200
```

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
CLI error paths. Milestone 4 adds coverage for the comparison config, the MAD and
LOF baselines and the comparison artifacts. Milestone 5 adds coverage for
scenario reproducibility, label isolation from features, missing-value handling and
the robustness artifacts. Milestone 6 adds coverage for the inference contract
(feature order, schema version, missing/invalid/NaN/infinite inputs, window
boundaries, artifact/manifest mismatch, duplicate window ids, determinism), the
status vocabulary (unscored is never normal) and the downstream integration
boundary. Milestone 7 adds coverage for the validation-only selection policy
(test metrics are structurally rejected), the incumbent-retention and
robustness-floor rules, the corrected M4 auto-picker, the threshold audit, the
readiness verdict, and the end-to-end path (malformed event rejected,
duplicate de-duplicated, late event re-windowed, missing value imputed and
reported, no unscored window forwarded as normal). Milestone 9 adds 51 offline
network tests covering the LAD envelope adapter (deduplication, out-of-order
detection, PII removal, partial-quality records), the 39-column schema and
zero-denominator semantics, causal rolling features, time/group split integrity,
scenario coverage, status handling (empty, sparse, degraded, naive and malformed
windows), model reload/reproducibility and the evidence-grounded explanation
adapter. Milestone 10 adds 36 offline tests covering upstream-correlator reuse,
canonical incident-schema validity, field preservation, scored/unscored/invalid/
error separation, duplicate and out-of-order handling, evidence redaction and
bounds, unsafe identity rejection, configuration, mocked-provider orchestration,
timeout/rate-limit/malformed-provider failures, diagnostic redaction, and a full
simulated-events → model → evidence → mocked-LLM → validated-incident path. The
full suite currently reports **411 passed**.

---

## 10. Repository structure

```text
ml-anomaly-detector/
├── README.md
├── MILESTONE_1_REPORT.md
├── MILESTONE_2_REPORT.md
├── MILESTONE_3_REPORT.md
├── MILESTONE_4_REPORT.md
├── MILESTONE_5_REPORT.md
├── MILESTONE_6_REPORT.md
├── MILESTONE_7_REPORT.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/{default.yaml, data_readiness.yaml, baseline.yaml, comparison.yaml,
│            robustness.yaml, selection.yaml, network.yaml, correlation.yaml}
├── docs/{DATASETS.md, DATA_READINESS.md, BASELINE.md, MODEL_COMPARISON.md,
│        ml_inference_contract.md, ML_HANDOFF.md, READINESS_AUDIT.md,
│        ml_network_reconnaissance.md, ml_network_features.md,
│        ml_network_evaluation.md, ml_network_integration.md,
│        ml_llm_integration_reconnaissance.md, ml_llm_integration.md}
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
│   ├── comparison_config.py       # M4
│   ├── mad_baseline.py            # M4
│   ├── lof_baseline.py            # M4
│   ├── comparison.py              # M4
│   ├── robustness_scenarios.py    # M5
│   ├── robustness.py              # M5
│   ├── inference/                 # M6
│   │   ├── __init__.py
│   │   ├── contract.py            # request/result types, statuses, reason codes
│   │   ├── detector.py            # load_detector / score_window / score_windows
│   │   └── adapter.py             # downstream integration boundary
│   ├── selection_policy.py        # M7 validation-only selection policy
│   ├── readiness_audit.py         # M7 fair comparison, threshold audit, verdict
│   ├── e2e_scenarios.py           # M7 raw-event end-to-end scenarios
│   ├── e2e_runner.py              # M7 real M1 -> M6 -> sink driver
│   ├── network/                   # M9 network/service-log extension
│   │   ├── simulator.py           # deterministic LAD-envelope simulator
│   │   ├── lad_adapter.py         # canonical LAD event mapping + quality
│   │   ├── features.py            # network-features-v1 (39 columns)
│   │   ├── baseline.py            # transparent rule/MAD baseline
│   │   ├── inference.py           # NetworkDetector / load_network_detector
│   │   ├── splits.py              # time and group split integrity
│   │   ├── evaluation.py          # status-aware metrics
│   │   ├── llm.py                 # optional evidence-grounded explanation
│   │   └── pipeline.py            # end-to-end orchestration
│   ├── correlation/               # M10 evidence correlation / incident drafts
│   │   ├── __init__.py            # public API
│   │   ├── config.py              # explicit config + environment overrides
│   │   ├── evidence.py            # WindowEvidence, validation, redaction
│   │   ├── incident.py            # upstream group → canonical incident draft
│   │   ├── pipeline.py            # status routing, correlation, LLM orchestration
│   │   └── upstream/              # vendored upstream correlator + provenance
│   └── dataset_adapters/{__init__.py, base.py, loghub_hdfs.py}
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_pipeline.py
│   ├── prepare_dataset.py
│   ├── audit_dataset.py           # M2
│   ├── export_training_ready.py   # M2
│   ├── make_splits.py             # M2
│   ├── prepare_experiments.py     # M2
│   ├── run_baseline.py            # M3
│   ├── run_comparison.py          # M4
│   ├── run_robustness.py          # M5
│   ├── smoke_test_inference.py    # M6
│   ├── run_readiness_audit.py     # M7
│   ├── run_network_simulator.py   # M9
│   ├── run_network_pipeline.py    # M9
│   └── run_correlation_rehearsal.py  # M10
├── tests/ (incl. fixtures/hdfs_sample/ and fixtures/contracts/)
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
- The Isolation Forest parameters are fixed configuration values; M4 compared a
  small candidate set but retained the baseline, and no broad hyperparameter search
  was performed.
- The anomaly score is a ranking statistic, **not** a probability.
- Detection delay is not measured — the label contract has no anomaly onset time.
- `model.joblib` is pickle-based; load it only from a trusted source. The M6 loader
  refuses paths outside the repository's `artifacts/` tree unless explicitly allowed.
- Milestone 5 found the detector is over-sensitive to legitimate distribution shifts
  and to missing telemetry; the M6 interface exposes `data_quality.imputed_features`
  and optional abstention policies, but the default scoring behaviour is unchanged.
- The M6 interface is in-process and single-threaded per call: no queueing, caching,
  rate limiting or concurrency control.
- **Milestone 7 verdict: `controlled demo only`.** The interface, split discipline
  and pipeline are sound, but every result is synthetic, one injected anomaly was
  missed, and the M5 false-alert weaknesses are unresolved. **No production-readiness
  claim is made.** See [`docs/READINESS_AUDIT.md`](docs/READINESS_AUDIT.md).
- M7 fixed the M4 auto-picker that selected a model using test-set metrics, and the
  unfair training rows given to the MAD/LOF candidates. The historical M4 run
  directory is preserved unchanged; a fresh run now agrees with the written M4 report.
- **M9 network figures are synthetic and scenario-repeating.** Every simulated run
  uses the same 90-minute schedule, so the splits share scenario structure. The
  measured F1/recall values validate the pipeline, not production generalisation.
  Legitimate traffic volume remains a real weakness: the benign high-traffic scenario
  has a 0.25 false-positive rate in the time test split. The rule baseline also beats
  the Isolation Forest on the clearest auth-burst pattern. See
  [`docs/ml_network_evaluation.md`](docs/ml_network_evaluation.md) §7.
- **M10 correlation is association, not causation.** The upstream correlator uses
  service identity and timestamp proximity only (120 s by default). It has no trace
  or dependency graph, no quiet-period lifecycle and no reopening logic.
- **The M9 feature windows do not retain event IDs.** Callers must supply a
  `window_id → supporting event IDs` index; the adapter never invents references.
- **Upstream has no LLM and no licence file.** The single vendored module is
  documented with provenance; publication of this repository containing it still
  needs owner/legal review.
- **No live LLM/provider was contacted.** Timeout, rate-limit, malformed-response and
  grounding failures are tested with injected transports and mock providers only.

## 12. Boundary with the next milestone

Milestones 1–10 stop here with a tested, documented data pipeline, a leakage-audited,
reproducible, training-ready dataset (features / labels / metadata kept separate,
splits produced under three strategies), a reproducible Isolation Forest baseline, a
fair comparison against a robust MAD baseline and LOF (novelty), a
robustness/failure-mode analysis against controlled scenarios, a **stable, versioned
inference interface** for the downstream correlation/explanation component, a
**readiness audit** that root-caused and fixed the M4 test-leakage in the auto-picker
and re-decided the model selection on validation evidence only, (M9) an additive
**network/service-log detector** with its own leak-safe feature schema, local
simulator, transparent baseline, split discipline and optional explanation adapter,
and (M10) a **canonical incident-draft boundary** that reuses the real upstream
correlator and validates its output against the LAD incident schema.

M1–M5 artifacts are preserved; M4, M5 and M7 write unique outputs under
`artifacts/comparison/`, `artifacts/robustness/` and `artifacts/readiness/`; M6
produces no artifacts of its own — it is an in-process interface over the M3 model
artifact. M9 writes only under `artifacts/network/` and `data/synthetic/network/`,
and does not modify any M1–M8 artifact. Future work must reuse the *same* contract:
the same allowlists, the same missing-value policy, the same score direction and the
same train/validation/test discipline. Do not use test labels for tuning or
selection, and do not retrain or re-threshold at inference time.

The M6 contract is versioned (`CONTRACT_VERSION = 1.1.0`,
`FEATURE_SCHEMA_VERSION = features-v1`). The `1.1.0` bump is **additive only**:
the network extension adds the separate `network-features-v1` schema, three
reason codes (`no_events`, `insufficient_coverage`, `telemetry_quality`) and
`EventType.AUTH_SUCCESS`; existing fields, statuses, score direction and the
`features-v1` allowlist are unchanged. Any further change to the request/result
shape or to a feature allowlist must bump those versions and update
[`docs/ml_inference_contract.md`](docs/ml_inference_contract.md). The M7 selection
policy is versioned by `configs/selection.yaml`; changing the weights, the margin or
the robustness floors requires a new audited run, not an in-place edit.

**Note for the next milestone:** the synthetic data is generated per-window; group
splits by `scenario_name` can place all anomalies in one fold (the tooling warns but
does not hide this). Choose the split strategy deliberately and consider a
scenario-mixed or time-based split for detection experiments. The `random` strategy is
the one used for the headline numbers. M7 re-ran the audit on the `random` split
only — the **M3/M4 baseline** time-based and group-based splits have **not** been
re-audited and remain the first thing the next milestone should do. M9 independently
uses audited chronological (`time`) and run-disjoint (`group`) splits for the network
extension, but that does not retro-audit the earlier baseline pipeline.

M10 is an adapter, not a second detector or a second LLM. It vendors exactly one
upstream module (`correlator.py`) with provenance, leaves every ML score, status,
threshold, identifier, timestamp and model version intact, and returns incidents
that validate against the canonical LAD schema. The next integration step is to
call `EvidenceCorrelationPipeline.correlate()` from the LAD detection worker and
persist its drafts through the existing incident store. See
[`docs/ml_llm_integration.md`](docs/ml_llm_integration.md) §13.
