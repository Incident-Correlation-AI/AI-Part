# Milestone 8 — Integration Boundary Preparation

**Date:** 2026-10-22 · **Scope:** ML/anomaly-detection workstream only
**Verdict:** **COMPLETE** — the boundary is documented, hardened, tested and
rehearsed. The real downstream interface remains **explicitly pending**.

---

## 1. Executive summary and verdict

The ML component already had a usable inference contract (M6) and a working
end-to-end path (M7). Milestone 8 inspected both, found **three real defects
in the boundary behaviour**, fixed them, centralised the forwarding rule so
no caller can implement a subtly different check, added a local integration
harness, and ran an offline rehearsal against the real M1 → M6 path.

**Verdict: complete.** Every acceptance criterion in §12 of the prompt is
met, with the single explicit exception that the downstream interface is
still unknown — which the prompt requires to remain pending and is documented
as such rather than guessed at.

**No production-readiness claim is made.** Milestone 7's verdict stands:
`controlled demo only`.

---

## 2. Baseline commands and results (Phase 0)

Run **before** any edit:

| # | Command | Result |
|---|---|---|
| 1 | `git status --short` | Clean — only the untracked `*_prompt.*` files. No unrelated user changes to preserve. |
| 2 | `git log --oneline -4` | `b114865` (M7 docs) on top of `ecf7473` (M7 code) |
| 3 | `python -m pytest tests/ -q` | **289 passed**, 27 warnings, 0 failures |
| 4 | `python scripts/smoke_test_inference.py` | Contract header + one scored result |
| 5 | `python scripts/run_readiness_audit.py` | `SELECTED: m3_isolation_forest_reference`, readiness `controlled demo only` |

**Pre-existing failures: none.** The only warnings are the known
`max_samples (256) > 110 training rows` scikit-learn notice.

---

## 3. Inspected files

**Docs:** `README.md`, `docs/ml_inference_contract.md`, `docs/ML_HANDOFF.md`,
`docs/READINESS_AUDIT.md`, `MILESTONE_7_REPORT.md`, `MILESTONE_6_REPORT.md`.

**Code:** `src/anomaly_detector/inference/{__init__,contract,detector,adapter}.py`,
`src/anomaly_detector/e2e_runner.py`, `src/anomaly_detector/e2e_scenarios.py`,
`src/anomaly_detector/readiness_audit.py`, `src/anomaly_detector/selection_policy.py`.

**Model path:** `artifacts/baseline/model/{model.joblib,manifest.json}`.

**Tests:** the full `tests/` suite, with a focus on
`tests/test_inference_contract.py`, `tests/test_inference_integration.py` and
`tests/test_e2e_pipeline.py`.

### Discrepancies between the prompt and the repository
None material. The prompt's description of the M6/M7 state matched the
repository exactly (289 tests, the same statuses, the same threshold, the same
`window_id` format, the same latency-degradation miss). No file needed to be
corrected to match the prompt.

---

## 4. Defects found in the existing boundary, and how they were fixed

Three genuine defects were found by reading and by direct reproduction. All
three were in the M6 adapter, and all three are now fixed and tested.

### Defect A — a sink failure aborted the whole batch

`EvidenceCorrelationAdapter.handle_windows()` called the sink without
protecting the call, so a sink exception propagated out of the method:

```
RAISED RuntimeError: sink down
-> the whole batch outcome is LOST; w1/w2 were already scored but never reported
```

Worse: the window had already been appended to `outcome.scored`, so a caller
that recovered the outcome would have seen a **failed delivery reported as a
successful one**.

**Fix:** a sink exception is now caught and recorded in
`outcome.sink_failures` as a `SinkFailure` (window id, exception type,
message). The window is **not** added to the forwarded list. The rest of the
batch continues. For the single-window path, `handle_window()` raises an
explicit `SinkDeliveryError`.

### Defect B — a malformed status crashed instead of failing closed

`result.status.value` raised `AttributeError` when the status was `None` or
a non-enum value:

```
RAISED AttributeError: 'NoneType' object has no attribute 'value'
```

**Fix:** the routing decision is now made by `route_result()`, which reads
every field defensively and returns `status="unknown"` for anything it
cannot interpret. A malformed status is **not forwarded** and is recorded.

### Defect C — the forwarding rule was duplicated, not centralised

The adapter had its own `if result.status is not InferenceStatus.SCORED`
check, and the M7 `e2e_runner.py` had a second copy. A teammate writing a
third call site would easily write a subtly different one.

**Fix:** the rule now lives in exactly one function,
`anomaly_detector.inference.contract.route_result()`, and both the adapter
and the harness use it. A test asserts the adapter contains no
hand-rolled status check.

### Two further hardenings
* `ScoredWindow.from_result()` used a bare `assert` for its invariant —
  which Python strips under `-O`. It now raises an explicit `ValueError`.
* An exception escaping `detector.score_window()` aborted the batch. It is
  now converted into an explicit `inference_error` result, so it is neither
  dropped nor turned into a normal result.

---

## 5. Files created and modified

### Created

| File | Purpose |
|---|---|
| `src/anomaly_detector/integration_harness.py` | The local harness: `IntegrationHarness`, `UnscoredCollector`, `run_rehearsal()`. |
| `scripts/run_integration_rehearsal.py` | CLI for the offline rehearsal. |
| `tests/test_integration_boundary.py` | **35 tests** covering the 16 required cases. |
| `docs/INTEGRATION_BOUNDARY.md` | The boundary design: types, lifecycle, forwarding rule, per-status handling, exceptions, duplicate behaviour, provenance, ownership, limitations, and what the teammate must still provide. |
| `MILESTONE_8_REPORT.md` | This report. |

### Modified

| File | Change | Why |
|---|---|---|
| `src/anomaly_detector/inference/contract.py` | Added `FORWARDABLE_STATUS`, `UNKNOWN_STATUS`, `RoutingDecision`, `route_result()`, `is_forwardable()`. **Purely additive** — no existing name, signature or behaviour changed. | Centralise the forwarding rule (§4.1). |
| `src/anomaly_detector/inference/adapter.py` | Added `UnscoredWindow`, `SinkFailure`, `SinkDeliveryError`; `AdapterOutcome` gained `unscored` / `sink_failures`; added the `unscored_handler` callback; sink failures and scoring exceptions are now explicit; batch duplicate ids are rejected. `rejected` / `rejected_count` are now **views** over `unscored` / `unscored_count`. | Fix defects A–C. The M6 API is preserved: `rejected` and `rejected_count` still work and still mean "not forwarded". |
| `src/anomaly_detector/inference/__init__.py` | Exported the new names. **Purely additive.** | — |
| `docs/ML_HANDOFF.md` | Added the harness pattern, the sink-failure behaviour, the verified commands, and §8 "what is still needed from you". | Phase 4 handoff. |
| `README.md` | M8 overview, module table, command, outputs, tests, limitations, boundary. | — |

### Deliberately **not** modified
The model, the threshold, the preprocessing, the training artifacts, the
feature schema, the M1–M5 modules, the M6 contract semantics, and every
existing artifact and report. No new dependency was added — the harness uses
only the standard library and the already-declared packages.

---

## 6. Boundary architecture and responsibility ownership

```
raw events
   │  M1  validation.validate_and_normalize_events
   ▼
   │  M1  preprocessing.aggregate_events_to_windows
   ▼
FeatureWindow ──► FeatureWindowRequest
   │
   ▼  M6  inference.detector.AnomalyDetector.score_window()   ← the real entry point
   │        (one load per session; nothing is fitted here)
   ▼
InferenceResult
   │
   ▼  M8  contract.route_result()        ← THE ONE forwarding rule
   │
   ├── status == "scored" ──► correlation sink      (full result + provenance)
   └── everything else   ──► unscored handler        (status, reason, provenance)
```

| Concern | Owner |
|---|---|
| Event validation / normalization | M1 |
| Window aggregation / features | M1 / M2 |
| Split discipline / leakage audit | M2 |
| Model training / threshold / evaluation | M3 |
| Model comparison and selection | M4 / M7 |
| Robustness validation | M5 |
| Readiness audit | M7 |
| Anomaly inference and score interpretation | M3 / M6 |
| **Scored-result eligibility and routing** | **M8 (this milestone)** |
| Evidence correlation / grouping | Downstream teammate |
| Incident explanation / root-cause hypotheses | Downstream teammate |
| API / queue / dashboard / cloud | Out of scope |

Full detail: [`docs/INTEGRATION_BOUNDARY.md`](docs/INTEGRATION_BOUNDARY.md).

---

## 7. Routing behaviour for every status

| Status | Forwarded? | Recorded as | Sink receives it? |
|---|---|---|---|
| `scored` | ✅ yes | forwarded | ✅ yes |
| `insufficient_data` | ❌ no | `unscored` + explicit handler | ❌ no |
| `invalid_input` | ❌ no | `unscored` + explicit handler | ❌ no |
| `model_unavailable` | ❌ no | `unscored` + explicit handler | ❌ no |
| `inference_error` | ❌ no | `unscored` + explicit handler | ❌ no |
| `unknown` (malformed / missing status) | ❌ no | `unscored`, status `unknown` | ❌ no |
| `unknown` (missing result) | ❌ no | `unscored`, status `unknown` | ❌ no |
| scoring exception | ❌ no | `unscored`, status `inference_error` | ❌ no |
| duplicate `window_id` in a batch | ❌ no (later occurrence) | `unscored`, `duplicate_window_id` | ❌ no |
| sink raises | n/a — it *was* scored | `sink_failures` | attempted, failed, **not** reported as delivered |

`route_result()` never converts an unknown outcome into `is_anomaly = False`.

---

## 8. Tests

`tests/test_integration_boundary.py` — **35 tests**, covering every one of
the 16 required cases:

| # | Required case | Test |
|---|---|---|
| 1 | Scored non-anomalous result is forwarded | `test_scored_non_anomalous_result_is_forwarded` |
| 2 | Scored anomalous result is forwarded | `test_scored_anomalous_result_is_forwarded` |
| 3 | `insufficient_data` not forwarded, explicitly recorded | `test_unscored_status_is_not_forwarded_and_is_recorded` |
| 4 | `invalid_input` not forwarded, explicitly recorded | ↑ (parametrised) |
| 5 | `model_unavailable` not forwarded, explicitly recorded | ↑ (parametrised) |
| 6 | `inference_error` not forwarded, explicitly recorded | ↑ (parametrised) |
| 7 | Unknown/malformed status not forwarded | `test_unknown_status_is_not_forwarded`, `test_none_status_is_not_forwarded`, `test_malformed_status_does_not_crash_the_adapter` |
| 8 | `None` / missing result / exception never becomes a normal result | `test_none_result_is_not_forwarded`, `test_scoring_exception_is_not_converted_to_a_normal_result`, `test_routing_never_converts_anything_to_is_anomaly_false` |
| 9 | Sink exception surfaced, no false success | `test_sink_exception_is_recorded_and_not_a_false_success`, `test_sink_exception_on_a_single_window_raises_explicitly`, `test_sink_failure_does_not_abort_the_rest_of_the_batch` |
| 10 | Mixed batch routes each outcome correctly | `test_mixed_batch_routes_each_outcome_correctly`, `test_one_failure_does_not_corrupt_unrelated_windows` |
| 11 | Duplicate `window_id` follows existing behaviour | `test_duplicate_window_id_follows_the_existing_contract` |
| 12 | Model/version/threshold/score direction/provenance survive routing | `test_model_threshold_and_score_direction_survive_routing` |
| 13 | Window identifiers remain traceable | `test_window_identifiers_remain_traceable` |
| 14 | No downstream re-thresholding / score-as-probability | `test_forwarded_result_keeps_the_original_threshold_and_score` |
| 15 | Harness runs offline | `test_harness_package_has_no_network_imports`, `test_rehearsal_script_runs_offline` |
| 16 | Unscored never reaches the sink, whatever `is_anomaly` says | `test_unscored_results_never_reach_the_sink_whatever_is_anomaly_says` |

Plus: `test_unscored_statuses_remain_distinguishable`,
`test_the_forwarding_rule_is_centralised_in_one_function`,
`test_is_forwardable_agrees_with_route_result`, and 7 harness-level tests.

**No existing test was deleted, skipped or weakened.** The full suite grew
from 289 to **324**.

---

## 9. Rehearsal — Phase 5

Run: `python scripts/run_integration_rehearsal.py` (offline, no network, no
AWS, no downstream service; a neutral in-memory sink stands in).

### Routing counts (contract behaviour)

| Measure | Value |
|---|---|
| Events attempted | 647 |
| Events accepted | 645 |
| Events rejected (malformed) | 1 |
| Events de-duplicated | 1 |
| Windows attempted | 49 |
| **Scored** | 49 |
| **Forwarded to the correlation sink** | 49 |
| **Unscored** | 0 |
| Unscored by status | `{}` (none occurred) |
| **Sink failures** | 0 |
| Duplicate window ids | 0 |
| **Forwarded statuses observed** | `{'scored': 49}` |
| **Every forwarded item had status `scored`** | ✅ **True** |
| **Any unscored item reached the sink** | ✅ **False** |
| All scenarios routing ok | ✅ **True** |

### Per-scenario results (ML detection — synthetic only)

| Scenario | Windows | Scored | Forwarded | Unscored | Flagged | Detected | Missed | False positives | Routing |
|---|---|---|---|---|---|---|---|---|---|
| `healthy_traffic` | 6 | 6 | 6 | 0 | 0 | – | – | 0 | ok |
| `http_5xx_spike` | 6 | 6 | 6 | 0 | 1 | 1 | 0 | 0 | ok |
| `latency_degradation` | 6 | 6 | 6 | 0 | 0 | 0 | **1** | 0 | ok |
| `database_timeout` | 6 | 6 | 6 | 0 | 1 | 1 | 0 | 0 | ok |
| `missing_latency` | 6 | 6 | 6 | 0 | 0 | – | – | 0 | ok |
| `malformed_event` | 5 | 5 | 5 | 0 | 0 | – | – | 0 | ok |
| `duplicate_and_late_events` | 6 | 6 | 6 | 0 | 0 | – | – | 0 | ok |
| `benign_distribution_shift` | 8 | 8 | 8 | 0 | 0 | – | – | 0 | ok |
| **Total** | **49** | **49** | **49** | **0** | **2** | **2** | **1** | **0** | **all ok** |

**Contract-routing success and ML detection success are reported separately.**
All 49 windows routed correctly; 2 of 3 injected anomalies were detected and
1 was missed. A scenario is never called detected merely because the harness
ran.

### Failures, mismatches and handling
* **Sink failures: 0.** The failure path is exercised by the unit tests, not
  by the rehearsal.
* **Exceptions: 0.**
* **Duplicates: 0** in the rehearsal (the M1 de-duplication removed the one
  duplicated event before windowing). The within-batch duplicate rejection is
  covered by the unit tests.
* **The one unmatched event** was the malformed record in
  `malformed_event`; the M1 validator rejected it, as designed.

---

## 10. Verified vs synthetic-only vs assumed vs unverified

### Verified (by running code and tests in this repository)
- M1–M7 reproduce; the M6 artifact loads and matches its manifest.
- The three boundary defects are real, were reproduced, and are now fixed.
- The forwarding rule is centralised in one function, and a test asserts no
  hand-rolled check remains in the adapter.
- Every non-`scored` status is not forwarded and is explicitly recorded.
- `None`, missing results, unknown/malformed statuses and exceptions all fail
  closed and are never turned into a normal result.
- A sink failure is recorded, is not reported as a successful delivery, and
  does not abort the rest of the batch.
- Provenance (model, version, artifact version, feature schema, threshold
  config, raw score, direction, threshold, decision, data quality) survives
  routing; window ids stay traceable.
- The threshold is applied exactly once, by the ML component; the score is
  explicitly flagged as not a probability.
- The rehearsal ran against the real M1 → M6 path: 49/49 windows scored,
  49/49 forwarded, **0 unscored reached the sink**.
- 324 tests pass (289 pre-existing + 35 new). No test was weakened or
  deleted.

### Synthetic-only
- Every detection, miss, false-positive count and flag rate in the rehearsal.
- The 2 detections and the 1 miss.
- The scenario set itself (hand-built M1 events).

### Assumed
- That the downstream component's real interface resembles `DownstreamSink`.
- That `window_id` will be the project `sample_key`, so results join back to
  the existing evidence tables.
- That the 13-feature allowlist is what the downstream component needs.
- That the 1-minute window size remains the right aggregation.

### Unverified
- Real downstream behaviour of any kind — the component does not exist yet.
- Concurrency, persistence, retry, delivery-guarantee and latency behaviour.
  The harness is single-process and in-memory, and **no such guarantee is
  claimed**.
- Cross-batch deduplication (only within-batch duplicates are rejected).
- Real production traffic, and whether the M7 readiness verdict holds there.

---

## 11. Limitations and risks

| Risk | Impact | Mitigation in place |
|---|---|---|
| The downstream interface is unknown | The adapter may need to map to a different shape. | `DownstreamSink` is a neutral protocol; the 8 questions the teammate must answer are listed in the handoff and the boundary doc. |
| Every detection result is synthetic | Real detection quality is unproven. | Labelled as synthetic everywhere; M7's `controlled demo only` verdict stands. |
| The harness is in-memory | No persistence, retries or delivery guarantee. | Documented explicitly; **no** exactly-once or durability claim is made. |
| Duplicate handling is per-batch only | The same window could be forwarded again in a later batch. | Documented; the teammate owns cross-batch deduplication decisions. |
| A scoring exception becomes an `inference_error` result | A programming error could be mistaken for a runtime failure. | The exception type and message are preserved in `reason`. |
| The 1-minute window and 13 features are inherited | The downstream component may need different granularity. | Would require a feature-schema version bump and a new audited run. |
| The latency-degradation miss persists | A latency-only anomaly is invisible to the model. | Reported honestly in the rehearsal and in M7; it is a feature gap, not a boundary gap. |

---

## 12. Acceptance criteria

| Criterion | Status |
|---|---|
| M1–M7 repository/contracts inspected before edits | ✅ §3 |
| Baseline tests and relevant commands run and recorded | ✅ §2 |
| Canonical inference contract remains authoritative and compatible | ✅ purely additive changes; the M6 API still works; 289 existing tests unchanged |
| Boundary documented for future teammate integration | ✅ `docs/INTEGRATION_BOUNDARY.md` |
| Only `scored` results can be forwarded | ✅ centralised rule; 4 tests |
| Every unscored status explicitly handled and distinguishable | ✅ §7; `unscored_by_status()` |
| `None`, exceptions and unknown/malformed statuses fail closed | ✅ 4 tests |
| Sink-failure behaviour explicit and tested | ✅ 3 tests |
| Provenance and window identifiers remain traceable | ✅ 2 tests |
| No downstream thresholding or score-as-probability interpretation | ✅ 1 test + the contract's `is_probability: false` |
| Offline rehearsal uses the real M1 → M6 path | ✅ §9 |
| Misses and false positives reported honestly | ✅ the latency miss is reported |
| Full suite passes | ✅ 324 passed |
| Smoke test and readiness audit run | ✅ §13 |
| Report distinguishes verified / synthetic-only / assumed / unverified | ✅ §10 |
| No out-of-scope service/cloud/explanation work added | ✅ no new dependency, no network, no service |
| Actual downstream interface remains explicitly pending | ✅ documented as pending in three places |

---

## 13. Exact commands run (Phase 6)

```bash
# 1. full suite BEFORE changes
python -m pytest tests/ -q                                  # → 289 passed

# 2. focused boundary tests
python -m pytest tests/test_integration_boundary.py -v       # → 35 passed

# 3. full suite AFTER changes
python -m pytest tests/ -q                                  # → 324 passed

# 4. inference smoke test
python scripts/smoke_test_inference.py                      # → contract header + scored result

# 5. readiness audit
python scripts/run_readiness_audit.py                       # → selected: m3_isolation_forest_reference
                                                         #    readiness: controlled demo only

# 6. local rehearsal
python scripts/run_integration_rehearsal.py                 # → 49 scored, 49 forwarded,
                                                         #    0 unscored, 0 sink failures

# 7. newly documented workflow commands
python -m pytest tests/test_inference_boundary.py -q         # → 35 passed
python scripts/run_integration_rehearsal.py                 # → exit code 0
```

If the package is not installed, prefix with `PYTHONPATH=src`
(e.g. `PYTHONPATH=src python scripts/run_integration_rehearsal.py`).

**Every command above was run and passed.** No command is claimed to pass on
the basis of code inspection.

---

## 14. Downstream teammate handoff steps

1. Read [`docs/ml_inference_contract.md`](docs/ml_inference_contract.md) (the
   canonical contract), then
   [`docs/INTEGRATION_BOUNDARY.md`](docs/INTEGRATION_BOUNDARY.md) (the
   boundary), then [`docs/ML_HANDOFF.md`](docs/ML_HANDOFF.md) (the quick
   start).
2. `pip install -e .` (or use `PYTHONPATH=src`). Ensure the M3 artifact
   exists: `python scripts/run_baseline.py run`.
3. Import from `anomaly_detector.inference` only — never from
   `anomaly_detector.baseline_model` or `...comparison`.
4. Call `load_detector()` once at start-up, or construct an
   `IntegrationHarness()` which does it for you.
5. Build `FeatureWindowRequest` from the feature pipeline output, using the
   project `sample_key` as `window_id`.
6. Route with `harness.route_windows(requests)` (recommended) or
   `route_result(result)` — do not write your own status check.
7. Forward only `outcome.scored`. Handle `outcome.unscored` separately; the
   statuses are `insufficient_data`, `invalid_input`, `model_unavailable`,
   `inference_error` and `unknown`.
8. Check `outcome.sink_failures` — a failure there was **not** delivered.
9. Keep the raw score and threshold separate; never re-threshold; never
   render the score as a probability.
10. Answer the 8 questions in `docs/ML_HANDOFF.md` §8 so the neutral sink can
    be replaced with the real adapter.

---

## 15. Recommended next step

> **Wait for the downstream teammate's answers to the 8 questions in
> `docs/ML_HANDOFF.md` §8, then replace the neutral sink with the real
> adapter.**

Concretely, the highest-value follow-ups, in order:

1. **Get the downstream interface definition** (the 8 questions). Until then
   the adapter cannot be finalised, and guessing at it would be worse than
   the neutral protocol.
2. **Run the M7 audit on the time-based and group-based splits.** M7 re-ran
   only the `random` split; that is the weakest link in the current evidence.
3. **Obtain a labelled real corpus.** No threshold or abstention default
   should change before then.
4. **Close the latency-degradation miss** — a feature gap, not a threshold
   gap.
5. **Close the M5 false-alert gap** with shift-aware or per-host
   normalisation — a new milestone with its own validation.
6. **Consider a contract version bump** (`1.0.0` → `1.1.0`) when the real
   downstream provenance requirements are known.

**Milestone 9 has not been started.** Awaiting approval.

---

**Stopping here.** Milestone 8 is complete.
