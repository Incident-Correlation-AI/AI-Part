# ML Readiness Audit (Milestone 7)

**Purpose:** determine whether the ML component is ready for **downstream local
integration** — not production readiness, which synthetic data cannot establish.

**Run it:**
```bash
python scripts/run_readiness_audit.py
# writes to artifacts/readiness/run_<timestamp>/
```

---

## 1. What the audit does

| Phase | What it does | Test data used? |
|---|---|---|
| 0 | Reproduce M1–M6, inspect artifacts, manifests, config, tests | read-only |
| 1 | Re-do the M4 IF-vs-LOF comparison under a **validation-only** policy, with both candidates on the **same** rows | ❌ never |
| 1 | Compute the final test evaluation **after** the selection is locked | ✅ once, at the end |
| 2 | Evaluate pre-declared threshold policies on **validation only** | ❌ never |
| 3 | Drive raw events → validation → windowing → features → M6 inference → downstream sink | ❌ n/a |
| 4 | Write JSON + Markdown reports to a **new** run directory | — |

Existing M1–M6 artifacts and reports are **never overwritten**.

---

## 2. The M4 discrepancy — root cause and resolution

### Symptom
M4's machine-generated report said `recommended: lof_novelty`; M4's written
report said *retain the M3 Isolation Forest*. Two documents, two answers.

### Root cause 1 — the auto-picker used test metrics

`src/anomaly_detector/comparison.py :: build_comparison_report()` selected the
winner with:

```python
key = (tf1, tpr, -tfp, trc)   # test_f1, test_precision, test_fp, test_recall
```

while the pre-declared criteria in `configs/comparison.yaml` said
*"evaluate test once after selection is fixed"*, and the same function emitted
a rationale claiming *"Selected by predeclared criteria using validation for
threshold/selection"*. The code contradicted both the config and its own
output string. The pre-declared **weights** were never applied, and the
declared *"if no candidate clearly outperforms baseline, retain the baseline"*
rule was never implemented.

### Root cause 2 — the comparison was unfair

`select_training_rows()` was imported in `comparison.py` but **never called**.
Candidates B (MAD) and C (LOF) were therefore fitted on **all 136 training
rows, including the 26 labelled anomalies**, while the Isolation Forest
incumbent was fitted on only the 110 normal-only rows. This violated the
declared `training_data_policy: normal_only` and gave the challengers
different training data to the incumbent.

### Resolution

| # | Change |
|---|---|
| 1 | The M4 auto-picker now selects on **validation** metrics only, applies the pre-declared weights, and prefers the incumbent on a tie. The report carries `selection_evidence_split: "validation"` and `used_test_metrics_for_selection: false`. |
| 2 | MAD and LOF now apply the same `normal_only` row selection as the incumbent, with the mask re-indexed through the prepared matrix so a `drop` policy cannot misalign rows. |
| 3 | A dedicated, pre-declared policy (`configs/selection.yaml` + `src/anomaly_detector/selection_policy.py`) makes the decision. The evidence type **structurally rejects** any test metric, so test leakage is impossible rather than merely avoided. |
| 4 | The historical M4 run directory and its report are preserved byte-for-byte. A fresh M4 run now reports `recommended: m3_isolation_forest_reference` — the generated and written recommendations **agree**. |

### Did the model change?
**No.** The selected model remains the M3 Isolation Forest. The M4 written
report had already reached that conclusion; M7 makes the *process* that
reaches it sound, and the validation-only evidence independently supports it.

---

## 3. The selection policy (pre-declared)

`configs/selection.yaml`:

* **Weighted validation score** — F1 (0.30), precision (0.25), recall (0.20),
  inverted false-positive rate (0.15), PR-AUC (0.10).
* **Required margin** — a challenger must beat the incumbent by ≥ 0.02.
* **Robustness floors** — a challenger that fails *any* of these cannot be
  selected, however good its F1:
  * validation FPR ≤ 0.25
  * validation recall ≥ 0.50
  * validation precision ≥ 0.50
  * benign-shift flag rate ≤ 0.50
  * missing-telemetry flag rate ≤ 0.50
* **Fairness** — same features, same samples, same preprocessing,
  `training_data_policy: normal_only`, matching training rows enforced.
* **Guards** — test metrics, test labels and the test split as a threshold
  source are all forbidden.

### Result on the available data

| Candidate | Val F1 | Val precision | Val recall | Val FPR | Weighted score | Floors failed |
|---|---|---|---|---|---|---|
| `m3_isolation_forest_reference` (incumbent) | 1.000 | 1.000 | 1.000 | 0.000 | 0.9500 | none |
| `lof_novelty` | 1.000 | 1.000 | 1.000 | 0.000 | 0.9500 | **missing-telemetry (1.00), normal-variation (1.00)** |

Both candidates reach a perfect validation F1 — the validation split **cannot
separate them**. The policy therefore retains the incumbent, and the
robustness floors independently disqualify LOF (it flags 100 % of
missing-telemetry and 100 % of normal-variation windows, against the
incumbent's 15 % and 22.5 %).

**This is the honest outcome**: on 45 validation windows with 11 positives,
the two models are indistinguishable on the labelled metrics, and the tie is
broken by robustness, not by the test set.

---

## 4. Threshold and abstention audit

### Provenance
The threshold `-0.6202666407504878` was **re-derived from validation data
only** and matches the manifest exactly. Poisoning the test split leaves it
unchanged. Method `validation_f1`, source `validation`, `is_heuristic: false`.

### Score semantics
`IsolationForest.score_samples` returns a value in `(-1, 0]`. **Lower = more
anomalous.** The decision rule is `is_anomaly = raw_score <= threshold`
(inclusive). The score is a **ranking statistic, not a probability**.

### Pre-declared threshold policies (validation only)

| Policy | Threshold | TP | FP | TN | FN | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|
| `incumbent_validation_f1` | -0.620267 | 11 | 0 | 34 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |
| `train_quantile_0.01` | -0.414596 | 11 | 33 | 1 | 0 | 0.250 | 1.000 | 0.400 | 0.971 |
| `train_quantile_0.02` | -0.416918 | 11 | 33 | 1 | 0 | 0.250 | 1.000 | 0.400 | 0.971 |
| `train_quantile_0.05` | -0.428987 | 11 | 31 | 3 | 0 | 0.262 | 1.000 | 0.415 | 0.912 |

**Finding:** the validation-F1 threshold is dramatically better than every
quantile fallback. The M3 choice is validated; the quantile fallback would
alarm on ~97 % of benign windows. The fallback remains available for the
degenerate "no labels" case only.

### Imputation and missingness
* The policy is `median_impute` with **training-split** medians — the
  trained model's own policy. A missing measurement is **never** zero-filled.
* Every imputed feature is reported in `data_quality.imputed_features`.
* **Missing telemetry is not proof of an anomaly** — the M5 measurement shows
  the detector flags 15 % of missing-telemetry windows, and 100 % of benign
  shifted windows. That is a false-alert problem, not a detection.

### Should `max_imputed_fraction` / `stale_after_seconds` get defaults?
**No — left unchanged, deliberately.** The validation split has 45 windows and
11 positives; that is far too thin to justify an abstention threshold, and
setting one would be an unvalidated behaviour change. Both options remain
available to callers via `DetectorOptions(...)`. The risk is documented
instead of being papered over.

### Status distinctions preserved
`scored` / `insufficient_data` / `invalid_input` / `model_unavailable` /
`inference_error` remain distinct. No automatic retraining or threshold
adjustment is performed anywhere in the audit.

---

## 5. End-to-end local pipeline

The real path, with no parallel implementation:

```
raw events
  -> validate_and_normalize_events        (M1)
  -> aggregate_events_to_windows          (M1)
  -> FeatureWindow.to_features_dict()     (M1, features only)
  -> inference.score_window()             (M6)
  -> EvidenceCorrelationAdapter           (M6)
  -> neutral sink
```

| Scenario | Windows | Flagged | Detected | Missed | False positives | Unscored | Expected behaviour |
|---|---|---|---|---|---|---|---|
| `healthy_traffic` | 6 | 0 | – | – | 0 | 0 | scored, not flagged |
| `http_5xx_spike` | 6 | 1 | 1 | 0 | 0 | 0 | flagged |
| `latency_degradation` | 6 | 0 | 0 | **1** | 0 | 0 | flagged |
| `database_timeout` | 6 | 1 | 1 | 0 | 0 | 0 | flagged |
| `missing_latency` | 6 | 0 | – | – | 0 | 0 | scored with imputation reported |
| `malformed_event` | 5 | 0 | – | – | 0 | 0 | malformed event rejected, window still scored |
| `duplicate_and_late_events` | 6 | 0 | – | – | 0 | 0 | duplicate de-duplicated, late event re-windowed |
| `benign_distribution_shift` | 8 | 0 | – | – | 0 | 0 | scored; a high flag rate would be a false-positive risk |

**Totals:** 47 windows, 2 flagged, 2 detected, **1 missed**, 0 false
positives, 0 unscored.

### What this shows — and what it does not

* ✅ The pipeline is wired correctly end to end and reaches the downstream
  boundary.
* ✅ The malformed event is **rejected** by the M1 validator, not coerced.
* ✅ The duplicate event is de-duplicated; the late event is assigned to the
  window its **event time** belongs to.
* ✅ The missing-latency window is scored with the missing value **imputed
  and reported**, not zero-filled.
* ✅ No unscored window was forwarded; the sink only ever saw scored results.
* ⚠️ **The latency-degradation scenario was NOT detected.** A sustained
  ~1.8–2.3 s latency spike with HTTP 200 responses stays inside the model's
  learned range. This is a genuine miss, reported rather than hidden.
* ⚠️ The benign shift here produced **0** false positives, but this scenario
  is milder than the M5 synthetic shift (which produced 100 %). The M5
  weakness is therefore **not** refuted — the E2E shift simply does not
  reproduce it. The report keeps both numbers visible.

### Scenario coverage gaps
| Requested family | Representable? | Note |
|---|---|---|
| healthy traffic | ✅ | |
| HTTP 5xx spike | ✅ | |
| latency degradation | ✅ | but **not detected** |
| database/dependency timeout | ✅ | `database_query` events carry no HTTP status; the signal is event count + latency |
| missing telemetry / latency | ✅ | |
| malformed / out-of-range event | ✅ | |
| duplicate / late events | ✅ | |
| benign distribution shift | ✅ | milder than the M5 shift |
| out-of-order (non-late) events | ❌ | the M1 windower is timestamp-bucketed, so ordering within a window is not observable at feature level |
| schema / feature-version mismatch | ⚠️ | covered by the M6 contract tests, not by an event-level scenario |

---

## 6. Readiness verdict

> ### `controlled demo only`
>
> The interface, the split discipline and the end-to-end pipeline are sound
> and reproducible, and the model selection is now defensible on validation
> evidence alone. However, every result comes from synthetic data and the M5
> false-alert weaknesses are unresolved, so the component is suitable for a
> controlled local demo / integration rehearsal, **not** for trusted operation
> on real traffic.

Cautions recorded with the verdict:

* 1 injected anomaly window was **not** detected — detection coverage is
  incomplete.
* Every result is synthetic; real-world detection quality is unproven.
* The validation split has 45 windows / 11 positives, so threshold and
  selection decisions carry wide uncertainty.
* M5's benign-shift (100 %) and missing-telemetry (15 %) false-alert rates
  are unresolved and still visible in the reports.

**Explicitly not claimed:** production readiness, an SLA, an availability
guarantee, or any statement that the detector works on real traffic.

---

## 7. What real data is needed to lift the verdict

| Need | Why it unblocks |
|---|---|
| A labelled real log corpus with window-level labels | The only way to get trustworthy precision/recall. Everything today is synthetic. |
| Enough labelled windows (≫ 45) | The validation split cannot separate two models that both score F1 = 1.0. |
| Benign-shift episodes with ground truth | To set `max_imputed_fraction` and judge the false-alert rate honestly. |
| Time-based and group-based split results | A single random split cannot rule out split-dependent conclusions. |
| Real missing-telemetry episodes | To distinguish "collection gap" from "attack" with data rather than assumption. |

---

## 8. Artifacts and reproduction

| Kind | Path |
|---|---|
| Config | `configs/selection.yaml` |
| Selection policy | `src/anomaly_detector/selection_policy.py` |
| Audit orchestration | `src/anomaly_detector/readiness_audit.py` |
| E2E scenarios | `src/anomaly_detector/e2e_scenarios.py` |
| E2E runner | `src/anomaly_detector/e2e_runner.py` |
| CLI | `scripts/run_readiness_audit.py` |
| Tests | `tests/test_selection_policy.py`, `tests/test_readiness_audit.py`, `tests/test_e2e_pipeline.py` |
| Reports | `artifacts/readiness/run_*/reports/readiness_audit.{json,md}` |
| Manifest | `artifacts/readiness/run_*/readiness_manifest.json` |

```bash
python -m pytest tests/ -q                          # full M1-M7 suite, offline
python -m pytest tests/test_selection_policy.py -q
python -m pytest tests/test_readiness_audit.py -q
python -m pytest tests/test_e2e_pipeline.py -q
python scripts/run_readiness_audit.py               # writes a new run directory
```
