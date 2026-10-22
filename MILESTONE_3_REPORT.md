# Milestone 3 Summary Report — First Baseline Anomaly Detector

**Date:** 2026-10-21  
**Branch / HEAD:** `fd9acf0` Milestone 2 complete → `c210b7a` Milestone 3 documentation  
**Working tree:** clean (untracked Milestone 2/3 prompt files only)  
**Result:** **195 tests passed**, 0 failures (51 M1 + 52 M2 + 92 M3)  
**Implementation scope:** Isolation Forest baseline only. No reasoning, correlation, API, dashboard, cloud or statistical-detection code was added.

---

## What was implemented

| Area | Files | Purpose |
|---|---|---|
| **Configuration** | `configs/baseline.yaml`, `src/anomaly_detector/baseline_config.py` | All parameters live in a validated config. A forbidden column in the explicit allowlist, an empty allowlist, an invalid policy/method/ratio or bad contamination fails at load time. |
| **Model-input contract** | `src/anomaly_detector/model_contract.py` | Only allowlisted columns reach the estimator. Schema mismatch (missing column) fails loudly; extras (sample key/metadata) are ignored. NaN is handled by a documented policy (never zero-filled). `±inf` is always rejected. Rows/samples dropped by policy are reported, never silently lost. |
| **Baseline model** | `src/anomaly_detector/baseline_model.py` | Thin wrapper around `sklearn.ensemble.IsolationForest`. Stores the feature schema + imputation statistics + score contract with the saved artifact. `score_samples` is the raw score (**lower = more anomalous**). Safe serialization via `joblib`. |
| **Thresholding** | `src/anomaly_detector/thresholding.py` | `validation_f1` / `validation_precision` / `validation_recall` / `train_quantile`. The **test split is refused** as a threshold source. If validation is missing/single-class the documented fallback (`train_quantile`) is used and marked `heuristic: true`. Boundary is **inclusive** (`raw_score <= threshold`). |
| **Evaluation** | `src/anomaly_detector/evaluation.py` | Metrics per split: precision/recall/F1, confusion matrix, ROC-AUC/PR-AUC (computed on `-raw_score`), prevalence, false-positive count/rate, alerts per host-hour (when metadata allows). Single-class/missing-label/no-flags cases are **skipped with a reason**. `detection_delay` is explicitly skipped (no onset timestamp). Trivial baselines (`no_alert`, `train_quantile`) are reported for context only. |
| **Artifacts** | `src/anomaly_detector/artifacts.py` | `model.joblib`, `manifest.json` (model version, seed, schema, data policy, training provenance, split info, score direction, threshold method), `predictions_<split>.csv`, `reports/metrics.{json,md}`, `experiment_note.md` (limitations + exact reproduction commands). No raw logs or secrets. |
| **Pipeline + CLI** | `src/anomaly_detector/baseline_pipeline.py`, `scripts/run_baseline.py` | `run` / `train` / `score` / `evaluate`. Split discipline enforced: train-only fitting, validation-only threshold selection, test never used for training/thresholding. |

**Lines added:** ~3,296 across the M3 modules, CLI and tests.

---

## Reproducible run (on the current synthetic dataset)

```bash
python scripts/generate_synthetic_data.py --config configs/default.yaml
python scripts/prepare_experiments.py
python scripts/run_baseline.py run
```

### Output (headline numbers, `random` split, seed 42)

**Model:** `sklearn.ensemble.IsolationForest` (1.9.1) | Seed 42 | Training: **110 / 136** rows (`normal_only`) | Threshold: **-0.6202666407504878** (`validation_f1`, source=`validation`, heuristic=false)

| Split | n (pos/neg) | Precision | Recall | F1 | ROC-AUC | PR-AUC | Flagged | FP rate |
|---|---|---|---|---|---|---|---|---|
| train | 136 (26/110) | 0.840 | 0.808 | 0.824 | 0.973 | 0.869 | 25 (18.4%) | 0.036 |
| validation | 45 (11/34) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 11 (24.4%) | 0.000 |
| test | 46 (8/38) | 1.000 | 0.875 | 0.933 | 0.980 | 0.946 | 7 (15.2%) | 0.000 |

> **These are synthetic-data results.** They demonstrate the pipeline works end to end; they do **not** establish real-world detection quality.

Artifacts written to `artifacts/baseline/`: `model/model.joblib`, `model/manifest.json`, `predictions_{train,validation,test}.csv`, `reports/metrics.{json,md}`, `experiment_note.md`.

---

## Test results

```
pytest tests/ -q
====================== 195 passed, 20 warnings in 18.31s =======================
```

Breakdown: 51 (M1) + 52 (M2) + **92 (M3)**. All offline.

New M3 tests cover: config guardrails (forbidden/empty/duplicate allowlist, bad policies/ratios/contamination), schema/missing-value/non-finite contract (never zero-filled; infinities always rejected; drop/error/policies reported), score direction ("lower = more anomalous", boundary inclusive), threshold rules (validation methods, tie-break, test-split refusal, single-class → documented fallback, train-quantile), metric edge cases (single-class, missing labels, no flags → skipped with reasons, detection_delay skipped), split discipline (poisoning test split changes nothing; changing validation changes threshold), and CLI paths (missing model/split/predictions → nonzero + clear error).

---

## Contract highlights (enforced by code + tests)

- **Explicit allowlist:** `data.feature_columns` only; config rejects `is_anomaly`, identifiers, timestamps, `scenario_name`. Eligibility is not inferred from "numeric".
- **Missing ≠ zero:** imputed with a **training-fitted median** or excluded under a documented rule or failed with `error`. Never `0`.
- **Non-finite:** `±inf` rejected always; matrix re-checked before scoring.
- **Fitting discipline:** normal-only fit on training rows only; imputation learned on training rows only.
- **Threshold discipline:** validation-only; test refused as source; fallback documented and marked heuristic.
- **Score:** `raw_score = score_samples` (lower = more anomalous), `is_anomaly = raw_score <= threshold` (inclusive), not a probability.
- **Evaluation discipline:** per-split, skip-with-reason, trivial baselines for context, synthetic caveat.
- **Reproducible:** seed 42, same config+seed ⇒ same assignment; manifest records everything needed to audit.

---

## Dependencies / verification

- Added `scikit-learn>=1.3.0`, `joblib>=1.3.0` to `requirements.txt` and `pyproject.toml`.
- Editable install verified in a clean venv (`--system-site-packages`): all M3 modules import, `run_baseline.py` works.
- Grepped for out-of-scope deps/features (`flask/fastapi/boto3/torch/sklearn outside the baseline/metrics used`, LLM keys, root-cause/correlation): only pre-existing leakage correlation heuristics (M2) remain; no new out-of-scope code.
- Artifacts (`artifacts/`, `data/processed/`) remain gitignored; `reports/` gitignored except `.gitkeep`.

---

## Files added/changed

**New:** `configs/baseline.yaml`, `src/anomaly_detector/{baseline_config,model_contract,baseline_model,thresholding,evaluation,artifacts,baseline_pipeline}.py`, `scripts/run_baseline.py`, `docs/BASELINE.md`, 6 M3 test files (`tests/test_baseline_config.py`, `tests/test_model_contract.py`, `tests/test_baseline_model.py`, `tests/test_thresholding.py`, `tests/test_evaluation.py`, `tests/test_baseline_pipeline.py`).  
**Modified:** `README.md`, `pyproject.toml`, `requirements.txt`, `tests/conftest.py`.  
**Git:** 3 commits on top of `fd9acf0` (`8123733`, `2511ab1`, `c210b7a`), working tree clean except untracked prompt files/zips.
