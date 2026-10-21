# Model Comparison (Milestone 4)

Milestone 4 compares a small candidate set against the M3 baseline with **validation-only selection** and **test evaluated once** after criteria are fixed. See [`MILESTONE_4_REPORT.md`](../MILESTONE_4_REPORT.md) for full details.

## Quick start
```bash
python scripts/run_comparison.py --config configs/comparison.yaml
```

Outputs go to `artifacts/comparison/run_<timestamp>/` (reports, predictions, manifest). M3 artifacts in `artifacts/baseline/` are preserved.

## Candidates
- **A:** M3 Isolation Forest (reference)
- **B:** Robust MAD baseline (training-fitted medians/MAD, higher = more anomalous)
- **C:** Local Outlier Factor in `novelty=True` mode (scaler fitted on training only)

## Principles
- Explicit feature allowlist only; preprocessing/thresholding fit on training/validation only
- Test labels never used for tuning/threshold/selection
- Missing/non-finite policies enforced; scores not treated as probabilities
- Synthetic results labeled as synthetic
- If no clear improvement, retain the simpler/reproducible baseline
