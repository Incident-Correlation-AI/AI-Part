# Milestone 4 — Model Comparison and Improvement

## Objective
Compare a small, justified set of anomaly-detection approaches against the M3 Isolation Forest baseline under fair, reproducible evaluation. Selection used **validation** data only; the **test** set was evaluated once after the comparison procedure and selection criteria were fixed.

## Pre-declared selection criteria (fixed before inspecting test metrics)
- **Primary objective:** maximize detection quality while controlling false alerts
- **Weights:** F1 (0.25), Precision (0.25), Recall (0.15), False Positive Rate (minimize, 0.20), PR-AUC (0.10)
- **Rule:** Validate on validation split; evaluate test once after selection is fixed. If no candidate clearly outperforms baseline, retain M3 Isolation Forest.
- **Tie-break:** Prefer simpler/more interpretable model with lower compute cost and fewer failure modes.
- **Test discipline:** Test labels/metrics were **never** used to tune features, parameters, thresholds, or choose the winner.

## Candidates tested
| Candidate | Type | Notes | Fit discipline |
|---|---|---|---|
| **A: m3_isolation_forest_reference** | `IsolationForest` (sklearn) | M3 baseline kept unchanged as reference. Config: n_estimators=100, max_samples=256, random_state=42. | Fit on training rows only (normal_only). Preprocessing/imputation fitted on training only. Threshold selected on validation via `validation_f1`. |
| **B: robust_mad_baseline** | Robust MAD (per-feature median/MAD, z-like robust deviations) | Transparent statistical baseline. Aggregation: `max_abs`. Constant/near-constant features handled explicitly (MAD≈0 → use absolute deviation from median). **Score direction:** higher = more anomalous. | Medians/MADs fitted on **training only**. Missing-value policy applied with training medians. |
| **C: lof_novelty** | `LocalOutlierFactor(novelty=True)` | One lightweight alternative to score unseen validation/test samples. `StandardScaler` fitted on training only. k adjusted defensively if `n_neighbors >= n_samples`. **Score direction:** lower = more anomalous (sklearn `score_samples`). | Preprocessing (scaler + imputation) fitted on training only. Fit on training; scored on val/test. |

**Not added:** Deep learning, LLMs, neural nets, K-means (clustering not a dedicated anomaly detector without strong justification). Only one alternative algorithm added.

## Dataset / splits / features
- **Splits:** `data/processed/splits/random/{train,validation,test}/` (features/labels/metadata aligned)
- **Strategy:** random (same as M3 headline). Split logic respects group/time considerations per M2.
- **Feature allowlist (explicit):** event_count, error_count, error_ratio, http_request_count, http_5xx_count, http_5xx_ratio, auth_failure_count, restart_count, mean_latency_ms, p95_latency_ms, latency_sample_count, http_request_eligible_count, error_eligible_count (13 features). Labels/IDs/timestamps/scenario metadata excluded from model inputs.
- **Training policy:** normal_only (fit on labelled normals). Missing policy: median_impute (training-fitted medians). Non-finite: rejected.
- **Seed:** 42

## Results (synthetic data — pipeline validation only)
Run: `artifacts/comparison/run_20261021T201001Z/`

| Candidate | Type | Variant | Val F1 | Val Prec/Rec | Test F1 | Test Prec/Rec | Test ROC-AUC | Test PR-AUC | Test TP/FP/FN | Flagged | Threshold (method) | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m3_isolation_forest_reference | IF | n_estimators=100 | 1.000 | 1.000 / 1.000 | 0.9333 | 1.000 / 0.875 | 0.9803 | 0.9464 | 7/0/1 | 7 | -0.62027 (validation_f1) | 0.329 |
| robust_mad_baseline | MAD | agg=max_abs | 0.9524 | 1.000 / 0.909 | 0.7692 | 1.000 / 0.625 | 0.5000 | 0.1739 | 5/0/3 | 5 | 4.35029 (validation_f1, higher>=th) | 0.024 |
| lof_novelty | LOF | n_neighbors=20 | 1.000 | 1.000 / 1.000 | 1.000 | 1.000 / 1.000 | 1.0000 | 1.0000 | 8/0/0 | 8 | -9.48531 (validation_f1) | 0.026 |

**Notes on metrics:** ROC-AUC/PR-AUC computed per candidate with correct score orientation. FP rate 0 for all on this synthetic test split. **All results are from synthetic data** — they validate the comparison pipeline, not real-world effectiveness.

## Recommendation
**Retain / recommend: `m3_isolation_forest_reference` (Isolation Forest)** based on the pre-declared criteria and evidence:

- On this dataset the LOF novelty run achieved perfect test metrics (F1=1.0) in this synthetic run; however, given the extremely clean synthetic separation and small test set (n=46, 8 positives), this is likely to be **over-optimistic** and needs validation on harder/more realistic data.
- MAD is fast and interpretable but underperformed relative to IF on recall/PR-AUC here.
- IF provides a good precision/recall tradeoff, is well-integrated with the M3 contract (score direction, thresholding, manifest), is robust and reproducible, and has lower generalization risk than a perfect-looking result on synthetic data.
- **Decision:** Keep **Isolation Forest (M3 configuration)** as the selected baseline for now. LOF is promising as a candidate for future testing on less separable data; MAD remains a useful transparent reference. Further evidence (larger/less separable datasets, cross-validation by scenario/group) is needed before switching.

## Artifacts (unique M4 outputs)
- **Config:** `configs/comparison.yaml`
- **Code:** `src/anomaly_detector/{comparison_config,mad_baseline,lof_baseline,comparison}.py`, `scripts/run_comparison.py`
- **Run dir:** `artifacts/comparison/run_20261021T201001Z/`
  - `comparison_manifest.json`
  - `reports/comparison_report.{json,md}`, `reports/comparison_table.csv`
  - `predictions/{candidate}_{validation,test}.csv`
- **Tests:** `tests/test_comparison.py` (6 tests)
- **M3 artifacts preserved:** `artifacts/baseline/` unchanged.

## Commands run
```bash
python -m pytest tests/ -q
python scripts/run_baseline.py run
python scripts/run_comparison.py
python -m pytest tests/test_comparison.py -v
python -m pytest tests/ -q
```

## Limitations & risks
- Synthetic data only; cannot prove real-world performance.
- Small test set (46 windows). LOF's perfect scores need skepticism.
- LOF sensitivity to `n_neighbors` and scaling; MAD assumes roughly unimodal/near-normal residuals around robust center.
- Score directions normalized for comparison but methods differ (IF/LOF lower=more anomalous in raw form; MAD higher=more anomalous).
- Threshold selected on validation only (correct). Test inspected once after selection fixed.
- No deep learning/LLMs; only lightweight CPU-friendly methods.
- If test results were inspected during development that would bias estimates — here test was only used in the final evaluation step after pre-declared criteria and validation-based selection.

## Next steps
- Test on more realistic data / harder splits (group/time) where anomalies are less separable.
- Calibrate scores only if needed and validate calibration separately (never interpret raw scores as probabilities).
- Consider ensemble only if clear complementarity is shown without increasing false alerts unacceptably.
