# Milestone 7 — ML Readiness Audit and End-to-End Local Validation

**Date:** 2026-10-22 · **Scope:** ML/anomaly-detection workstream only
**Verdict:** **`controlled demo only`** — not ready for trusted operation on
real traffic; ready for a controlled local integration rehearsal.

---

## 1. Repository findings — were M1–M6 reproducible?

**Yes, all of them.** Every documented command was run *before* any edit.

| Command | Result |
|---|---|
| `python -m pytest tests/ -q` | **247 passed**, 20 warnings, 0 failures |
| `python scripts/run_baseline.py run` | model + manifest + predictions + metrics regenerated |
| `python scripts/run_comparison.py` | M4 comparison ran (see §3 for the defect found) |
| `python scripts/run_robustness.py` | M5 robustness ran, 5 scenarios |
| `python scripts/smoke_test_inference.py` | M6 smoke test passed, contract header + one scored result |

**Pre-existing failures: none.** The only warning is the known
`max_samples (256) > 110 training rows` scikit-learn notice (M7 issue #6).

### Artifacts verified
- `artifacts/baseline/model/model.joblib` loads in a clean process;
  `manifest.json` ↔ artifact feature list match exactly (13 columns, same order).
- Threshold `-0.6202666407504878` re-derived **from validation data only** and
  matched the manifest to 1e-12. Poisoning the test split left it unchanged.
- Training medians in the artifact are the training-split medians; the
  feature schema, score direction (`lower_is_more_anomalous`) and threshold
  rule (`raw_score <= threshold`) match the M6 contract.

### M4 discrepancy — root cause traced (Phase 0.4)

`build_comparison_report()` in `src/anomaly_detector/comparison.py` selected
the winner with:

```python
key = (tf1, tpr, -tfp, trc)   # test_f1, test_precision, test_fp, test_recall
```

**The auto-picker used test-set metrics to choose the model.** A second,
independent defect was also found: `select_training_rows()` was imported into
`comparison.py` but **never called**, so the MAD and LOF candidates were fitted
on all 136 training rows (including the 26 labelled anomalies) while the
Isolation Forest incumbent saw only the 110 normal-only rows.

---

## 2. Files changed / created

### Created
| File | Purpose |
|---|---|
| `configs/selection.yaml` | Pre-declared, validation-only selection policy (weights, margin, robustness floors, fairness, guards) |
| `src/anomaly_detector/selection_policy.py` | The policy engine. `CandidateEvidence` **structurally rejects** any test metric |
| `src/anomaly_detector/readiness_audit.py` | Phases 1–4: fair comparison, robustness evidence, threshold audit, readiness verdict, report rendering |
| `src/anomaly_detector/e2e_scenarios.py` | 8 raw-event scenarios (healthy, 5xx, latency, DB timeout, missing telemetry, malformed, duplicate/late, benign shift) |
| `src/anomaly_detector/e2e_runner.py` | Drives the real M1 → M6 path into a neutral sink |
| `scripts/run_readiness_audit.py` | CLI entry point |
| `tests/test_selection_policy.py` | 14 tests |
| `tests/test_readiness_audit.py` | 15 tests |
| `tests/test_e2e_pipeline.py` | 13 tests |
| `docs/READINESS_AUDIT.md` | Full audit documentation |
| `MILESTONE_7_REPORT.md` | This report |

### Modified
| File | Change |
|---|---|
| `src/anomaly_detector/comparison.py` | **Root-cause fix**: the auto-picker now selects on validation metrics only, applies the pre-declared weights, prefers the incumbent on a tie, and reports `selection_evidence_split` / `used_test_metrics_for_selection`. MAD and LOF now apply the same `normal_only` row selection, with the mask re-indexed through the prepared matrix. |
| `README.md` | M7 overview, module table, commands, outputs, tests, limitations, boundary, repo structure |

### Deliberately **not** modified
The M1–M5 source modules, the M6 inference contract, all existing artifacts
and reports, and the selected model. The historical M4 run directory
`artifacts/comparison/run_20261021T201001Z/` is preserved byte-for-byte,
including its original `recommended: lof_novelty`.

---

## 3. M4 discrepancy — root cause and resolution

### Symptom
M4's machine-generated report said `recommended: lof_novelty`; M4's written
report said *retain the M3 Isolation Forest*.

### Root cause 1 — the auto-picker used test metrics
The selection key was built from `test_f1`, `test_precision`, `test_fp` and
`test_recall`, directly contradicting the pre-declared rule in
`configs/comparison.yaml` (*"evaluate test once after selection is fixed"*)
and the rationale string the same function emitted (*"Selected by predeclared
criteria using validation for threshold/selection"*). The declared **weights**
were never applied, and the declared *"retain the baseline unless clearly
outperformed"* rule was never implemented.

### Root cause 2 — the comparison was unfair
`select_training_rows()` was imported but never called. Candidates B (MAD) and
C (LOF) were fitted on **all 136 training rows including the 26 labelled
anomalies**; the incumbent saw only the 110 normal-only rows. This violated
the declared `training_data_policy: normal_only` and gave the challengers
different training data.

### Resolution
1. The M4 auto-picker now selects on **validation** metrics only, applies the
   pre-declared weights, and prefers the incumbent on a tie. The report
   carries `selection_evidence_split: "validation"` and
   `used_test_metrics_for_selection: false`.
2. MAD and LOF now apply the same `normal_only` selection, with the mask
   re-indexed through `prepared.row_indices` so a `drop` policy cannot
   misalign rows.
3. A dedicated pre-declared policy (`configs/selection.yaml`) makes the
   decision, with a required improvement margin and robustness floors. The
   evidence type rejects test metrics *structurally*, so leakage is
   impossible rather than merely avoided.
4. The generated and written recommendations can no longer contradict: a
   fresh M4 run now reports `recommended: m3_isolation_forest_reference`,
   matching the M4 written report.

### Was the model switched?
**No.** The selected model remains the M3 Isolation Forest. The M4 written
report had already reached that conclusion; M7 makes the *process* sound, and
the validation-only evidence independently supports it. The artifact and
model version are unchanged.

---

## 4. Was selection validation-only? Did the test set stay untouched?

**Yes — both, and this is enforced structurally rather than by convention.**

| Guard | Where | Test |
|---|---|---|
| Test metrics cannot enter the selection evidence | `CandidateEvidence.__post_init__` raises on any `test_*` key | `test_evidence_rejects_test_metrics` |
| The decision records its evidence split | `SelectionDecision.to_dict()["selection_evidence_split"] == "validation"` | `test_decision_records_the_evidence_split` |
| The M4 auto-picker ignores test metrics | selection key built from `val_*` only | `test_m4_comparison_no_longer_selects_on_test_metrics` (behavioural: a candidate with perfect **test** metrics but worse **validation** metrics does **not** win) |
| Thresholds come from validation only | `provenance["thresholds_selected_on"] == "validation"` | `test_thresholds_come_from_the_validation_split_only` |
| Both candidates trained on the same rows | `provenance["candidates_share_training_rows"]` | `test_both_candidates_use_the_same_training_rows` |
| The incumbent's threshold is independent of the test split | re-derived from validation; poisoning the test labels left it unchanged | verified manually in §1 |

**Test data usage:** the test split was scored **once**, after the selection
decision was locked, purely as a final evaluation. Its metrics appear in the
report under `final_test_evaluation`, each explicitly annotated
*"computed AFTER the selection decision was locked; not used for selection"*.

### Selection result

| Candidate | Val F1 | Val P | Val R | Val FPR | Weighted score | Robustness floors failed |
|---|---|---|---|---|---|---|
| `m3_isolation_forest_reference` (incumbent) | 1.000 | 1.000 | 1.000 | 0.000 | 0.9500 | none |
| `lof_novelty` | 1.000 | 1.000 | 1.000 | 0.000 | 0.9500 | missing-telemetry (1.00), normal-variation (1.00) |

Both candidates reach a **perfect validation F1** — the 45-window validation
split **cannot separate them**. The policy retains the incumbent, and the
robustness floors independently disqualify LOF (100 % of missing-telemetry and
100 % of normal-variation windows flagged, against the incumbent's 15 % and
22.5 %). The tie was broken by robustness, not by the test set.

**Final test evaluation (computed after the decision, not used for it):**
incumbent F1 0.9333, LOF F1 1.0000. LOF's better test score is precisely the
kind of signal that must not drive a decision — on 46 synthetic windows with
clean separation it is exactly the over-optimism the M4 report warned about.

---

## 5. Threshold and abstention findings

### Provenance (Phase 2.1)
`-0.6202666407504878` was **re-derived from validation data only** and matches
the manifest exactly. Method `validation_f1`, source `validation`,
`is_heuristic: false`. Poisoning the test split does not change it.

### Score semantics (Phase 2.2)
`IsolationForest.score_samples` returns a value in `(-1, 0]`. **Lower = more
anomalous.** The decision rule is `is_anomaly = raw_score <= threshold`
(inclusive). The score is a **ranking statistic, not a probability** — the
M6 result carries `score.is_probability: false`.

### Pre-declared policies, validation only (Phase 2.3–2.4)

| Policy | Threshold | TP | FP | TN | FN | Precision | Recall | F1 | FPR | False alerts/window |
|---|---|---|---|---|---|---|---|---|---|---|
| `incumbent_validation_f1` | -0.620267 | 11 | 0 | 34 | 0 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `train_quantile_0.01` | -0.414596 | 11 | 33 | 1 | 0 | 0.250 | 1.000 | 0.400 | 0.971 | 0.733 |
| `train_quantile_0.02` | -0.416918 | 11 | 33 | 1 | 0 | 0.250 | 1.000 | 0.400 | 0.971 | 0.733 |
| `train_quantile_0.05` | -0.428987 | 11 | 31 | 3 | 0 | 0.262 | 1.000 | 0.415 | 0.912 | 0.689 |

**Finding:** the M3 validation-F1 threshold is decisively better than every
quantile fallback; the fallback would alarm on ~97 % of benign windows. The
M3 threshold choice is **validated**. The fallback remains for the degenerate
"no labels" case only.

### Imputation and missingness (Phase 2.5)
* `median_impute` with **training-split** medians — the trained model's own
  policy. A missing measurement is **never** zero-filled; the E2E
  missing-latency window confirms the value is imputed and reported in
  `data_quality.imputed_features`.
* **Missing telemetry is not proof of an anomaly.** The detector flags 15 % of
  missing-telemetry windows (M5) — a false-alert problem.

### Should the abstention policies get defaults? (Phase 2.6)
**No. Left unchanged, deliberately.** The validation split (45 windows / 11
positives) is far too thin to justify an abstention threshold, and setting one
would be an unvalidated behaviour change. Both `max_imputed_fraction` and
`stale_after_seconds` remain available to callers via `DetectorOptions(...)`;
the risk is documented instead of hidden.

### Status distinctions preserved (Phase 2.7)
`scored` / `insufficient_data` / `invalid_input` / `model_unavailable` /
`inference_error` remain distinct. No automatic retraining or threshold
adjustment anywhere in the audit.

---

## 6. End-to-end pipeline results

Real path, no parallel implementation:
`raw events → validate_and_normalize_events → aggregate_events_to_windows →
FeatureWindow.to_features_dict() → inference.score_window() →
EvidenceCorrelationAdapter → neutral sink`

| Scenario | Windows | Flagged | Detected | Missed | False positives | Unscored | Expected behaviour |
|---|---|---|---|---|---|---|---|
| `healthy_traffic` | 6 | 0 | – | – | 0 | 0 | scored, not flagged |
| `http_5xx_spike` | 6 | 1 | 1 | 0 | 0 | 0 | flagged |
| `latency_degradation` | 6 | 0 | 0 | **1** | 0 | 0 | flagged |
| `database_timeout` | 6 | 1 | 1 | 0 | 0 | 0 | flagged |
| `missing_latency` | 6 | 0 | – | – | 0 | 0 | scored, imputation reported |
| `malformed_event` | 5 | 0 | – | – | 0 | 0 | malformed event rejected |
| `duplicate_and_late_events` | 6 | 0 | – | – | 0 | 0 | de-duplicated, late event re-windowed |
| `benign_distribution_shift` | 8 | 0 | – | – | 0 | 0 | scored |

**Totals:** 47 windows · 2 flagged · 2 detected · **1 missed** · 0 false
positives · 0 unscored.

### Verified
* The malformed event is **rejected** by the M1 validator (not coerced), and
  the window is still scored.
* The duplicate event is de-duplicated; the late event lands in the window its
  **event time** belongs to.
* The missing-latency window is scored with the value **imputed and reported**,
  not zero-filled.
* No unscored window was forwarded; the sink only ever saw scored results
  (asserted inside the runner).
* Window ids are unique and traceable (`host|service|window_start`).

### Misses and false positives — reported, not hidden
* **`latency_degradation` was NOT detected.** A sustained ~1.8–2.3 s latency
  spike with HTTP 200 responses stays inside the model's learned range. This
  is a genuine miss.
* **0 false positives in the E2E benign shift** — but this scenario is milder
  than the M5 synthetic shift (100 % flag rate there). The E2E result does
  **not** refute the M5 weakness; both numbers stay visible in the report.
* The harness does **not** assert that every injected scenario must be
  detected; it reports detections, misses, false positives and abstentions.

### Documented scenario gaps
* **Out-of-order (non-late) events** are not observable at feature level —
  the M1 windower is timestamp-bucketed.
* **Schema / feature-version mismatch** is covered by the M6 contract tests,
  not by an event-level scenario.

---

## 7. Runtime observations

| Observation | Value |
|---|---|
| Phase 1 (fair comparison + robustness) | ~2 s wall clock |
| Phase 3 (47 windows, full M1 → M6 path) | ~1 s wall clock |
| Windows scored end to end | 47 |
| Environment | Python 3.13.14, scikit-learn 1.9.1, numpy 2.3.5, pandas 2.3.3, joblib 1.5.3 |
| Machine | 8 GB RAM / 4 logical CPU threads (the documented dev target) |
| New runtime dependencies | **none** (standard library + the already-declared set) |

These are **measurements on a small local dataset, not SLAs**. No
large-scale extrapolation is made.

---

## 8. Artifact and report locations

| Kind | Path |
|---|---|
| Selection policy config | `configs/selection.yaml` |
| Selection policy code | `src/anomaly_detector/selection_policy.py` |
| Audit orchestration | `src/anomaly_detector/readiness_audit.py` |
| E2E scenarios / runner | `src/anomaly_detector/e2e_scenarios.py`, `src/anomaly_detector/e2e_runner.py` |
| CLI | `scripts/run_readiness_audit.py` |
| Tests | `tests/test_selection_policy.py` (14), `tests/test_readiness_audit.py` (15), `tests/test_e2e_pipeline.py` (13) |
| JSON report | `artifacts/readiness/run_*/reports/readiness_audit.json` |
| Markdown report | `artifacts/readiness/run_*/reports/readiness_audit.md` |
| Manifest | `artifacts/readiness/run_*/readiness_manifest.json` |
| Documentation | `docs/READINESS_AUDIT.md` |
| Preserved M4 history | `artifacts/comparison/run_20261021T201001Z/` (untouched) |
| Corrected M4 run | `artifacts/comparison/run_20261022T122819Z/` (new recommendation) |

---

## 9. Test commands and results

```bash
# Before any change
python -m pytest tests/ -q                                # → 247 passed

# M7 test files
python -m pytest tests/test_selection_policy.py -q       # → 14 passed
python -m pytest tests/test_readiness_audit.py -q        # → 15 passed
python -m pytest tests/test_e2e_pipeline.py -q           # → 13 passed

# Full suite (M1-M7 regression)
python -m pytest tests/ -q                                # → 289 passed, 0 failed

# Documented commands
python scripts/run_baseline.py run
python scripts/run_comparison.py
python scripts/run_robustness.py
python scripts/smoke_test_inference.py
python scripts/run_readiness_audit.py
```

**289 passed, 0 failed** (247 pre-existing + 42 new). No test was weakened,
skipped or deleted. All tests are offline: no network, no AWS, no dashboard,
no training job.

**One test was corrected during development** (not weakened): an assertion in
`test_readiness_audit.py` initially inspected the *source text* of
`build_comparison_report` and tripped over the explanatory comment that
mentions the old test-metric key. It was rewritten as a **behavioural** test
(a candidate with perfect test metrics but worse validation metrics must not
win), which is a stronger guarantee than the text check.

---

## 10. Limitations and real-data needs

### Limitations
- **All data is synthetic.** Real-world detection quality is unproven.
- The validation split has 45 windows / 11 positives — too thin to separate
  two models that both score F1 = 1.0.
- The test split has 46 windows / 8 positives and was scored once.
- A single random split cannot rule out split-dependent conclusions; the
  time-based and group-based splits were not re-run.
- The E2E scenarios are hand-built and milder than production traffic.
- No production SLA, latency guarantee or availability claim is made.
- The downstream teammate's real interface is still unknown; the
  `DownstreamSink` protocol is a neutral, documented boundary.

### What real data is needed
| Need | Unblocks |
|---|---|
| A labelled real log corpus with window-level labels | Trustworthy precision/recall |
| Many more labelled windows | Separating candidates; justifying abstention defaults |
| Benign-shift episodes with ground truth | An honest false-alert rate and a calibrated `max_imputed_fraction` |
| Time-based and group-based split results | Ruling out split-dependent conclusions |
| Real missing-telemetry episodes | Separating "collection gap" from "attack" with data |

---

## 11. Verified vs synthetic-only vs assumed vs unverified

**Verified by running code and tests in this repository**
- M1–M6 reproduce; the M6 artifact loads and matches its manifest.
- The M4 auto-picker selected on test metrics, and MAD/LOF were trained on
  different rows than the incumbent — both confirmed by reading the code and
  by re-running it.
- The fix works: a fresh M4 run now agrees with the M4 written report.
- The threshold is reproducible from validation data only and is independent
  of the test split.
- Both candidates are now trained on the same 110 normal-only rows with the
  same features and preprocessing.
- The selection decision is validation-only and structurally cannot consume
  a test metric.
- The end-to-end path reaches the downstream boundary; no unscored window is
  forwarded as normal; the malformed event is rejected; the duplicate is
  de-duplicated; the late event is re-windowed; the missing value is imputed
  and reported.
- 289 tests pass (247 pre-existing + 42 new).

**Synthetic-only**
- Every metric, score, flag rate and detection in this report.
- The 5xx and DB-timeout detections and the latency miss.
- The 0 % E2E benign-shift false-positive rate.

**Assumed**
- That the downstream component's real interface resembles `DownstreamSink`.
- That `normal_only` training remains the right policy for the real data.
- That the M1 window size (1 minute) is the right aggregation for production.

**Unverified**
- Real production latency, concurrency, memory and availability behaviour.
- Behaviour under a genuinely adversarial distribution.
- Whether the Isolation Forest remains the right model once real data exists.
- Whether the M5 false-alert weaknesses persist at production traffic volume.

---

## 12. Recommendation

> ### `controlled demo only`
>
> The ML component is **ready for a controlled local demo / integration
> rehearsal** with the downstream teammate. It is **not** ready for trusted
> operation on real traffic, and no production-readiness claim is made.

**Why not "ready for downstream local integration"?** The interface, the split
discipline and the pipeline are sound — but one injected anomaly was missed,
every result is synthetic, and the M5 false-alert weaknesses (100 % on a
benign shift, 15 % on missing telemetry) are unresolved. Those are exactly the
kinds of findings that a real integration rehearsal is meant to surface, which
is why the verdict is "demo" rather than "blocked".

**Why not "blocked"?** No safety precondition failed: the selection is
validation-only, the test split was reserved, the contract distinguishes
scored from unscored, and the pipeline reaches the downstream boundary
correctly. Nothing prevents the downstream teammate from integrating against
the documented contract now.

### Recommended next steps — clearly separated from the completed work
1. **Re-run the M4 comparison and the M7 audit on the time-based and
   group-based splits.** A single random split is the weakest link in the
   current evidence.
2. **Obtain a labelled real corpus.** Until then, no threshold or abstention
   default should be changed.
3. **Add the missing E2E scenario families** — out-of-order events and
   schema-version mismatch at the event level.
4. **Investigate the latency-degradation miss.** A latency-only anomaly with
   HTTP 200 responses is invisible to the current model; that is a feature
   gap, not a threshold gap.
5. **Close the M5 false-alert gap** with a shift-aware or per-host
   normalisation — a *new* milestone with its own validation, not a threshold
   tweak.
6. **Re-derive the M4 report** so the historical artifact and the corrected
   code are distinguishable by a reader (the historical run is preserved
   unchanged; the new run carries the corrected recommendation).
7. **Do not** add an HTTP layer, a queue, an LLM or an explanation path
   without a separate, explicit milestone.

---

**Stopping here.** Milestone 7 is complete; Milestone 8 has not been started.
