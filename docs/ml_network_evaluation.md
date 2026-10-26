# Network / Service-Log Evaluation

**Status:** measured on the full configured synthetic run · **Date:** 2026-10-26
**Feature schema:** `network-features-v1` · **Contract:** `1.1.0`
**Deployed strategy:** `time` · **Cross-check strategy:** `group`

> These numbers validate the pipeline and the evaluation discipline on
> deterministic **synthetic** data. They do not establish real-world detection
> quality. Read §7 before quoting any figure.

---

## 1. Evaluation protocol

| Step | Rule |
|---|---|
| Data | 10 runs × 90 minutes × 3 streams; 72,823 events; 2,700 labelled windows. |
| Training rows | Training split only, normal-only, telemetry-clean, `event_count >= 5`. |
| Preprocessing | Median imputation fitted on the training rows only. |
| Model | `IsolationForest(n_estimators=200, max_samples=256, random_state=42, bootstrap=False)`. |
| Score | `score_samples`; lower = more anomalous; `is_anomaly = raw_score <= threshold`. |
| Threshold | Selected on the **validation** split by maximum F1, subject to minimum precision and recall. Test is refused as a selection source. |
| Test use | Evaluated once, after selection. No test-driven tuning. |
| Status handling | Only `scored` windows enter supervised metrics. Unscored windows are reported separately as unknown. |
| Baseline | Transparent rule/MAD baseline evaluated independently on the same scored windows. It never rewrites the model flag. |

### 1.1 Splits

Both strategies use `0.6 / 0.2 / 0.2` and produce 1,620 / 540 / 540 windows.

`time` is strictly chronological. Because simulator runs start one day apart,
the split is also run-contiguous:

* train: 2026-01-01 … 2026-01-06
* validation: 2026-01-07 … 2026-01-08
* test: 2026-01-09 … 2026-01-10

`group` assigns whole runs to exactly one split and is integrity-checked for
disjointness:

* train: `run-02`, `run-03`, `run-05`, `run-06`, `run-07`, `run-08`
* validation: `run-04`, `run-09`
* test: `run-00`, `run-01`

Training selection is identical for both:

| Training split | Windows |
|---|---:|
| Rows in split | 1,620 |
| Excluded: labelled anomaly | 240 |
| Excluded: telemetry issue | 180 |
| Excluded: below minimum coverage | 6 |
| **Rows fitted** | **1,200** |

### 1.2 Thresholds

| Strategy | Threshold | Method | Validation F1 at threshold |
|---|---:|---|---:|
| `time` | `-0.5235479383749508` | `validation_f1` | 0.7558 |
| `group` | `-0.5235918264524634` | `validation_f1` | 0.7845 |

Both thresholds were chosen from 481 candidates with the tie-break
“max F1, then max recall, then lowest threshold”. Threshold selection excludes
windows labelled `telemetry_issue`; the evaluation tables include every
`scored` window, including the few telemetry-issue windows that still had
usable data. That is why validation F1 at selection (`0.7558` / `0.7845`) is
slightly higher than the validation F1 reported in §2 (`0.7514` / `0.7760`).
The threshold is **not** re-fitted to make those two numbers agree.

---

## 2. Headline results

### 2.1 Deployed strategy — `time`

| Split | Scored / windows | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train (in-sample) | 1,614 / 1,620 | 0.7218 | 0.8000 | 0.7589 | 0.9580 | 0.8090 | 192 | 74 | 48 | 1300 |
| validation | 538 / 540 | 0.6989 | 0.8125 | 0.7514 | 0.9344 | 0.7724 | 65 | 28 | 15 | 430 |
| **test** | **538 / 540** | **0.7059** | **0.9000** | **0.7912** | **0.9674** | **0.8326** | **72** | **30** | **8** | **428** |

The test split has 80 positive windows and 458 negative scored windows.

### 2.2 Cross-check strategy — `group`

| Split | Scored / windows | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train (in-sample) | 1,614 / 1,620 | 0.7216 | 0.8208 | 0.7680 | 0.9573 | 0.8277 | 197 | 76 | 43 | 1298 |
| validation | 538 / 540 | 0.6893 | 0.8875 | 0.7760 | 0.9623 | 0.8355 | 71 | 32 | 9 | 426 |
| **test** | **538 / 540** | **0.6869** | **0.8500** | **0.7598** | **0.9602** | **0.8095** | **68** | **31** | **12** | **427** |

### 2.3 Model versus transparent baseline (test)

| Strategy | Detector | Precision | Recall | F1 | ROC-AUC | PR-AUC | Flagged |
|---|---|---:|---:|---:|---:|---:|---:|
| `time` | Isolation Forest | 0.7059 | 0.9000 | **0.7912** | 0.9674 | 0.8326 | 102 |
| `time` | Rule/MAD baseline | 0.7024 | 0.7375 | 0.7195 | 0.8395 | 0.6074 | 84 |
| `group` | Isolation Forest | 0.6869 | 0.8500 | **0.7598** | 0.9602 | 0.8095 | 99 |
| `group` | Rule/MAD baseline | 0.7195 | 0.7375 | 0.7284 | 0.8418 | 0.6263 | 82 |

Window-level agreement on the test split:

| Strategy | Both flag | Model only | Baseline only | Neither |
|---|---:|---:|---:|---:|
| `time` | 72 | 30 | 12 | 424 |
| `group` | 66 | 33 | 16 | 423 |

The model finds more of the incident windows; the baseline is more precise on
authentication bursts but misses more overall. They are reported side by side,
not merged.

---

## 3. Per-scenario results (test)

The per-scenario table includes **all** windows of that scenario, including
unaffected streams. For an HTTP burst, `orders-db` windows are negatives, so the
scenario FPR is not the same as the clean-benign FPR in §4.

### 3.1 `time`

| Scenario | Family | Windows | Positives | Flagged | Precision | Recall | F1 | FPR | Baseline F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `http_5xx_burst` | incident | 60 | 40 | 40 | 0.9000 | 0.9000 | 0.9000 | 0.2000 | 0.5846 |
| `db_timeout_latency` | incident | 60 | 20 | 20 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| `auth_failure_burst` | incident | 60 | 20 | 17 | 0.9412 | 0.8000 | 0.8649 | 0.0250 | 0.9302 |
| `benign_high_traffic` | benign | 60 | 0 | 15 | — | — | — | 0.2500 | — |
| `benign_latency_shift` | benign | 60 | 0 | 3 | — | — | — | 0.0500 | — |
| `benign_restart` | benign | 30 | 0 | 1 | — | — | — | 0.0333 | — |
| `normal` | benign | 90 | 0 | 3 | — | — | — | 0.0333 | — |
| `normal_recovery` | benign | 60 | 0 | 1 | — | — | — | 0.0167 | — |
| `malformed_telemetry` | telemetry | 58 | 0 | 2 | — | — | — | 0.0345 | — |

### 3.2 `group`

| Scenario | Family | Windows | Positives | Flagged | Precision | Recall | F1 | FPR | Baseline F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `http_5xx_burst` | incident | 60 | 40 | 36 | 0.9722 | 0.8750 | 0.9211 | 0.0500 | 0.6333 |
| `db_timeout_latency` | incident | 60 | 20 | 20 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| `auth_failure_burst` | incident | 60 | 20 | 14 | 0.9286 | 0.6500 | 0.7647 | 0.0250 | 0.9302 |
| `benign_high_traffic` | benign | 60 | 0 | 11 | — | — | — | 0.1833 | — |
| `benign_latency_shift` | benign | 60 | 0 | 4 | — | — | — | 0.0667 | — |
| `benign_restart` | benign | 30 | 0 | 3 | — | — | — | 0.1000 | — |
| `normal` | benign | 90 | 0 | 4 | — | — | — | 0.0444 | — |
| `normal_recovery` | benign | 60 | 0 | 3 | — | — | — | 0.0500 | — |
| `malformed_telemetry` | telemetry | 58 | 0 | 4 | — | — | — | 0.0690 | — |

---

## 4. Benign false positives

Clean benign windows are those with a benign scenario label and no telemetry
issue.

| Strategy | Split | Benign windows | Flagged | FPR | Baseline FPR |
|---|---|---:|---:|---:|---:|
| `time` | validation | 300 | 23 | 0.0767 | 0.0500 |
| `time` | test | 300 | 23 | **0.0767** | 0.0500 |
| `group` | validation | 300 | 25 | 0.0833 | 0.0600 |
| `group` | test | 300 | 25 | **0.0833** | 0.0500 |

Most false positives come from `benign_high_traffic` (25% in the time test
split, 18.3% in the group test split). Benign latency shifts and routine
restarts are much less likely to be flagged. The model is more sensitive to
legitimate traffic volume than to legitimate latency or restart variation.

---

## 5. Detection delay

A scenario group is `(run_id, scenario)`. Delay is the difference between the
first labelled anomalous window and the first flagged window at or after onset.
A missed group is counted as missed, not as zero delay.

| Strategy | Split | Groups | Detected | Missed | Mean | Median | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| `time` | test | 6 | 6 | 0 | 0.0000 min | 0.0000 | 0.0000 |
| `group` | test | 6 | 6 | 0 | 0.3333 min | 0.0000 | 2.0000 |

Every incident family was detected in every test run under both strategies.

---

## 6. Telemetry-damaged windows

| Strategy | Split | Damaged windows | Scored | Unscored | Unscored reason | Flagged |
|---|---|---:|---:|---:|---|---:|
| `time` | validation | 60 | 58 | 2 | `no_events` | 1 |
| `time` | test | 60 | 58 | 2 | `no_events` | 2 |
| `group` | validation | 60 | 58 | 2 | `no_events` | 2 |
| `group` | test | 60 | 58 | 2 | `no_events` | 4 |

The two unscored windows are the deliberately empty `orders-db` window at
minute 33 in each test run. They return `insufficient_data` / `no_events` with
`is_anomaly = None`; they are **not** counted as normal. The remaining damaged
windows are scored from partially usable telemetry, and their quality counts
are included as features.

Overall status coverage on the test split is 538 / 540 scored (99.6%). The only
unscored reason is `no_events`.

---

## 7. Interpretation and limitations

### What the results support

* The pipeline is deterministic, reproducible and leak-safe end to end.
* The three required incident families are detectable on the synthetic data:
  DB timeouts perfectly, HTTP 5xx bursts strongly, auth bursts well.
* Unscored telemetry is surfaced explicitly and never counted as normal.
* The threshold selected on validation transfers to test without collapsing:
  time F1 0.7514 → 0.7912; group F1 0.7760 → 0.7598.
* The Isolation Forest improves overall F1 over the transparent baseline on
  this data, while the baseline remains useful evidence.

### What the results do **not** support

* **No real-world quality claim.** Every run repeats the same scenario
  schedule, so the splits share scenario structure. This measures the pipeline,
  not generalisation to production traffic.
* **No claim that `group` is worse or better in general.** Its test recall is
  5 percentage points lower than `time`; with only two test runs, that is a
  small-sample difference, not a stable ranking.
* **No claim about unseen incident types.** Only three incident families exist.
  A novel incident with no matching signal is not represented here.
* **No claim about adversarial behaviour.** The simulator does not generate
  evasion, credential stuffing or exploit traffic.
* **No claim of calibrated severity.** `severity` is derived from transparent
  rule evidence, not from a probability.
* **High-traffic false positives are a real weakness.** A 25% FPR on the
  benign high-traffic scenario means an operator would see alerts during
  legitimate load increases. This should be tuned or suppressed before any
  production use.

### Known failure modes visible in the numbers

| Failure mode | Evidence | Consequence |
|---|---|---|
| Traffic-volume sensitivity | `benign_high_traffic` FPR 0.25 (time), 0.1833 (group) | False alerts during legitimate load spikes. |
| Auth-burst recall drop under group split | Recall 0.80 → 0.65 | Possible missed auth bursts on unseen runs. |
| Rule baseline beats the model on auth F1 | 0.9302 vs 0.8649 (time), 0.7647 (group) | The model adds little for the clearest auth pattern. |
| Model false positives outside the affected stream | `http_5xx_burst` scenario FPR 0.20 | Cross-stream windows may be flagged during an incident. |
| Empty telemetry | 2 unscored test windows | Correctly unknown, but they produce no detection coverage. |

---

## 8. Artifacts

The full run writes (all git-ignored):

```
artifacts/network/
├── features.csv
├── metadata.csv
├── labels.csv
├── lad_parse_summary.json
├── reports/metrics.{json,md}          # combined report
├── model/{model.joblib,manifest.json} # deployed copy (time strategy)
└── experiments/
    ├── time/
    │   ├── model/{model.joblib,manifest.json}
    │   ├── predictions_{train,validation,test,all}.csv
    │   ├── threshold.json
    │   ├── reports/metrics.{json,md}
    │   └── experiment_note.md
    └── group/ …                       # same layout
```

Prediction rows carry `status`, `reason_code`, `raw_score`, `is_anomaly`,
`baseline_is_anomaly`, `baseline_raw_score` and the evaluation label columns.
Unscored rows have `raw_score = None` and `is_anomaly = None`.

---

## 9. Reproduce

```bash
rm -rf artifacts/network data/synthetic/network data/processed/network_splits
python scripts/run_network_pipeline.py --config configs/network.yaml
```

Approximate runtime on the reference machine: 2 minutes 9 seconds for 10 runs,
both strategies and 200-tree forests.

Inspect the measured report:

```bash
python - <<'PY'
import json
for strategy in ("time", "group"):
    with open(f"artifacts/network/experiments/{strategy}/reports/metrics.json") as f:
        m = json.load(f)
    test = m["splits"]["test"]
    print(strategy, {k: test["overall"][k] for k in
          ("precision", "recall", "f1", "roc_auc", "pr_auc")})
PY
```

Evaluation tests:

```bash
python -m pytest tests/test_network_inference.py tests/test_network_features.py -q
```

Prediction reload is verified on every pipeline run: the persisted model is
loaded back through `load_network_detector()` and must reproduce the same raw
score and flag before artifacts are written.

---

## 10. Evaluation definitions

| Metric | Definition |
|---|---|
| Precision | `TP / (TP + FP)` over scored windows. |
| Recall | `TP / (TP + FN)` over scored windows. |
| F1 | Harmonic mean of precision and recall; `None` when both are undefined. |
| FPR | `FP / (FP + TN)` over scored negatives. |
| ROC-AUC | Rank quality over scored windows and continuous `raw_score`. |
| PR-AUC | Precision-recall AUC over scored windows. |
| Coverage | `n_scored / n_windows`. |
| Detection delay | Minutes from the first labelled anomalous window to the first flag at/after onset, per `(run_id, scenario)`. |
| `both` / `model_only` / `baseline_only` / `neither` | Flag agreement between the Isolation Forest and the rule baseline. |

An unscored window contributes to coverage and status counts, never to
precision, recall, F1, FPR, ROC-AUC or PR-AUC.
