# Milestone 2 — Completion Report

## Objective
Implement only **Milestone 2** (Data Validation, Leakage Audit, Dataset Splits, and Experiment Readiness) in the repository created during Milestone 1. Work in the existing repository, keep changes consistent with existing style, add thorough tests, produce reproducible artifacts, and document everything. Do **not** start Milestone 3 (no Isolation Forest, no model training/scoring, no statistical detection code) and no AWS/cloud/API/UI code.

## Completed deliverables
- **Configuration:** `configs/data_readiness.yaml` with explicit feature-role patterns, eligible-features allowlist, data policy (non-negative/unit-interval/integer, missing-policy report-only), split config (random/time/group), and output paths. Validation rejects invalid roles/strategies/ratios/unknown policies.
- **Feature roles & leakage guard:** `src/anomaly_detector/roles.py` classifies columns (label/identifier/timestamp/scenario/future_derived/feature), records ambiguity and unknowns, enforces that eligibility is an **explicit allowlist** (never just "numeric"). Forbid label/identifier/timestamp/scenario/future-derived columns from the model feature matrix; detect violations when allow-listed but forbidden.
- **Dataset alignment:** `src/anomaly_detector/dataset.py` loads features/metadata/labels (CSV/Parquet via `load_table`), builds a stable sample key (default `host_id|service|window_start`), aligns rows (positions match), joins labels by key (reporting unmatched labels), and detects duplicate sample keys.
- **Audit:** `src/anomaly_detector/audit.py` is read-only (never mutates sources), profiles every column (dtype, missing, infinities, cardinality, constant/near-constant, min/max/mean/std), checks non-negative/unit-interval/integer policies, reports missing (never imputed), reports true infinities as errors, reports duplicate rows and duplicate sample keys, summarises labels. Writes `reports/data_quality_report.json` + `.md`.
- **Leakage:** `src/anomaly_detector/leakage.py` audits for leaked columns (label/identifier/timestamp/scenario/future-derived in features), cross-table label overlap, identifier-like high-cardinality (integer-only heuristic), perfect correlation with label, writes `reports/leakage_audit.json` + `.md`, and provides `assert_no_leakage()`.
- **Splits:** `src/anomaly_detector/splits.py` implements **random**, **time-aware** (chronological check), **group-aware** (LPT greedy, seeded tie-break, group disjointness, full coverage) with reproducibility, single-class and empty-split warnings.
- **Training-ready:** `src/anomaly_detector/training_ready.py` exports strictly separate `features.csv` (eligible features only, no labels/ids/timestamps), `labels.csv`, `metadata.csv`, `sample_index.csv`, and `manifest.json` (seed, row counts, feature list, leakage status, SHA-256 checksums). Verifies row alignment and sample keys on write/export.
- **Scripts:** `scripts/audit_dataset.py`, `scripts/export_training_ready.py`, `scripts/make_splits.py`, `scripts/prepare_experiments.py` (one-shot audit→export→splits). All use the same alignment + audit + leakage logic.
- **Tests:** 103 tests total (all pass, offline). Added 52 new tests covering readiness config validation, roles, alignment, audit, leakage, splits, training-ready export. All use small fixtures; no network access.
- **Docs & housekeeping:** `docs/DATA_READINESS.md`, README updated for Milestones 1–2, pyproject description updated, `.gitignore` now ignores `reports/` with `.gitkeep`, new modules wired in package layout.

## Results on current dataset (synthetic)
- 227 windows (5,367 input events), 13 eligible features, 256 label rows (45 anomaly). 
- Audit: **ready=True**, 0 errors. Warnings: missing `http_5xx_ratio` / `mean_latency_ms` / `p95_latency_ms` in windows lacking samples (missing≠zero, expected), and 6 duplicate feature rows. 29 label rows reference windows not materialized (expected, empty windows skipped).
- Leakage: **clean=True**, 0 errors. Weak-to-moderate feature/label correlations (e.g. latency ratios), no identifier/label/timestamp in features.
- Splits (seed=42): random 136/45/46; time 136/45/46 (validation has 0 positives — time-based asymmetry); group by `scenario_name` 113/60/54 (validation/test single-class warnings; tool warns, not hidden).
- Training-ready artifacts: `features.csv`, `labels.csv`, `metadata.csv`, `sample_index.csv`, `manifest.json` in `data/processed/training_ready/`; manifests + tables in `data/processed/splits/{random,time,group}/`.

## Verification
- `python3 -m pytest tests/ -q` → **103 passed** (4.8s, offline).
- `python3 scripts/generate_synthetic_data.py` and `python3 scripts/prepare_experiments.py` both run end-to-end successfully.
- Editable install in a clean venv with `--no-deps` confirmed imports work.
- Grep confirms no forbidden ML/model/training/framework or cloud/API code beyond standard library + pandas/numpy/pyyaml.
- The audit is strictly read-only; exports require leakage to be clean by default.

## Git
- 8 commits: config+roles+alignment → audit/leakage/splits/training_ready → scripts+tests (103 passed). Working tree clean after producing reports/docs/artifacts (all generated outputs are .gitignored).

## Known tradeoffs/notes
- Group-by-scenario is often degenerate for a small synthetic anomaly corpus (warnings surfaced). Use random/time depending on evaluation goal.
- Missingness reporting only (no imputation) — appropriate for this milestone; downstream decides imputation/handling.
- High-cardinality heuristic limited to integer-valued columns to avoid false positives on continuous latency/ratios.
- Alignment uses position + sample key validation (defensive, fails loudly on tampering/mismatched row counts).

All Milestone 2 requirements are satisfied. Milestone 3 must not start until data is trusted (and the split degenerate cases are understood).
