# Data Readiness (Milestone 2)

This document describes the Milestone 2 tooling: how the processed dataset is
profiled, audited for leakage, exported as a training-ready dataset, and split into
reproducible train/validation/test folds.

Scope: **data validation and experiment-readiness only.** No model is trained and no
anomaly score is computed here. That belongs to a later milestone.

---

## 1. Concepts

### Sample key

Features, labels and metadata are separate tables that must stay row-aligned. A
**stable sample key** is built from the configured key columns
(`data_policy.sample_key_columns`, default `host_id`, `service`, `window_start`) and
stored in `metadata.csv`/`labels.csv`. The training-ready export also writes an
explicit `sample_index.csv` (`row_index → sample_key`). Row *i* is the same sample in
every export; this is verified before writing.

### Roles

Every column is classified into exactly one primary role, in priority order:

| Role | Meaning | Model input? |
|---|---|---|
| `feature` | numeric predictor | yes, **if** allow-listed |
| `label` | ground truth | never |
| `identifier` | host/service/event/sample id | never |
| `timestamp` | timestamps / window bounds | never |
| `scenario` | scenario / generation metadata | never |
| `future_derived` | computed using future observations | never |

Roles come from ordered substring patterns plus optional per-column `explicit`
overrides. Matching is case-insensitive and the *first* matching rule wins as the
primary role; all matches are recorded so ambiguity is reportable.

### Eligibility is an allowlist, not a dtype test

A column is model-eligible **only if** it appears in
`feature_roles.eligible_features`. "It is numeric" is never sufficient. A column that
is allow-listed *and* also matches a forbidden role (label/identifier/timestamp/
scenario/future-derived) is a **leakage violation** and is reported, not silently
dropped.

---

## 2. Configuration (`configs/data_readiness.yaml`)

```yaml
feature_roles:
  explicit: {}                 # exact-name role overrides (highest priority)
  patterns:                    # ordered role rules, first match wins
    - role: label
      match: [is_anomaly, scenario_name, label, ...]
    - role: timestamp
      match: [window_start, window_end, ...]
    - role: identifier
      match: [host_id, service, sample_key, ...]
    - role: future_derived
      match: [future, next_, ...]
    - role: feature
      match: [count, ratio, mean_, p95_, ...]
  eligible_features: [...]     # the model-input allowlist
  unknown_role_policy: error   # error | warn | exclude

data_policy:
  non_negative_columns: [...]  # must be >= 0
  unit_interval_columns: [...] # must be within [0, 1]
  integer_columns: [...]       # whole numbers expected
  constant_max_unique: 1       # constant-feature threshold
  near_constant_ratio: 0.99    # near-constant threshold
  missing_value_action: report # report only — never imputed with zero
  sample_key_columns: [host_id, service, window_start]
  group_columns: [host_id, service, scenario_name]

split:
  strategy: random             # random | time | group
  seed: 42
  ratios: {train: 0.6, validation: 0.2, test: 0.2}
  group_column: scenario_name
  time_column: window_start
  label_column: is_anomaly

paths:
  reports_dir: reports
  splits_dir: data/processed/splits
  training_ready_dir: data/processed/training_ready
```

Invalid values (unknown role, ratios not summing to 1, unknown strategy, bad unknown
policy) raise a clear `ValueError` at load time.

---

## 3. Dataset audit

`anomaly_detector.audit.audit_dataset(aligned, config)`

Read-only. Produces a per-column profile (dtype, missingness, infinite values,
distinct count, dominant-value ratio, constant/near-constant flags, min/max/mean/std)
plus:

- duplicate feature rows;
- duplicate sample keys (error — alignment would be ambiguous);
- policy violations: negative values in non-negative columns, ratios outside `[0,1]`,
  non-integer counts, unknown roles, forbidden columns present in features;
- label distribution and missing-label counts.

Missing values are a **warning** (policy: report, never impute). True infinities are
an **error** (a model cannot consume them). The source tables are never modified.

Outputs: `reports/data_quality_report.json` and `.md`.

---

## 4. Leakage audit

`anomaly_detector.leakage.audit_leakage(aligned, config)`

Detects, and fails on:

- forbidden-role columns present in the feature table;
- allow-listed features that also carry a forbidden role;
- label column names overlapping the feature table;
- (warning) integer-valued, near-unique columns that look like identifiers;
- (warning) features whose absolute correlation with the label exceeds `0.999`.

`assert_no_leakage(report)` raises if any error-severity finding exists. The export
step calls the leakage audit and refuses to write on errors (`strict_leakage=True`);
`--allow-leakage` exists only for debugging and still records the problem.

Outputs: `reports/leakage_audit.json` and `.md`.

---

## 5. Splits

`anomaly_detector.splits.make_splits(metadata, config, labels=None)`

| Strategy | Method | Guarantee checked |
|---|---|---|
| `random` | seeded `numpy` permutation, sliced by ratio | same seed ⇒ identical assignment |
| `time` | sort by `time_column`, slice chronologically | max(train) ≤ min(validation) ≤ … ≤ min(test) |
| `group` | whole groups assigned to one split (size-desc LPT greedy) | no group appears in two splits; all groups covered |

Every strategy also checks full row coverage and that no row is assigned twice.
Warnings are emitted for empty splits and single-class splits (a split with only
positives or only negatives makes evaluation degenerate).

Group and time keys may live in the metadata **or** the aligned labels table (e.g.
`scenario_name`). When a key comes from labels it is used only to partition rows,
never as a model feature.

Outputs: `data/processed/splits/<strategy>/{train,validation,test}/…` plus
`split_manifest.json`.

---

## 6. Training-ready export

`anomaly_detector.training_ready.build_training_ready(...)` /
`write_training_ready(...)`

Writes, into `data/processed/training_ready/`:

| File | Contents |
|---|---|
| `features.csv` | eligible features only — no labels, ids or timestamps |
| `labels.csv` | ground truth, aligned by row |
| `metadata.csv` | identifiers/descriptive columns, aligned by row |
| `sample_index.csv` | `row_index → sample_key` |
| `manifest.json` | seed, row count, column lists, alignment method, leakage status, label distribution, SHA-256 checksums per file |

`verify_alignment` fails loudly if features/labels/metadata/index row counts differ
or if a recomputed sample key does not match the stored key.

---

## 7. Commands

```bash
# Everything at once (audit → export → random/time/group splits)
python scripts/prepare_experiments.py

# Or step by step
python scripts/audit_dataset.py
python scripts/export_training_ready.py
python scripts/make_splits.py --strategy random
python scripts/make_splits.py --strategy time
python scripts/make_splits.py --strategy group --group-column scenario_name
```

`audit_dataset.py` exits non-zero if the audit has errors or the leakage audit is not
clean, so it can gate CI.

---

## 8. Findings on the current synthetic dataset

Run against the Milestone 1 exports (5,367 events → 227 windows, 256 label rows,
45 anomalous windows):

- **Audit:** ready (0 errors). 4 warnings: missing `http_5xx_ratio`,
  `mean_latency_ms`, `p95_latency_ms` in windows with no HTTP/latency samples
  (expected — missing ≠ zero), and 6 duplicate feature rows (windows whose feature
  vectors coincide).
- **Leakage:** clean. 13 eligible features; 0 findings.
- **Alignment:** 227/227 windows have a label; 29 label rows reference windows that
  were not materialized (empty windows are skipped by design).
- **Splits:** `random` is balanced (136/45/46 rows). `time` places most anomalies in
  the test fold (validation has 0 positives) — a real property of scenario timing.
  `group` by `scenario_name` puts all anomalies in one fold (warned). These are
  surfaced, not hidden; pick the strategy per experiment.

---

## 9. Design decisions

- **Missing ≠ zero.** Missing measurements are reported and left as `NaN`; they are
  never imputed with zero and never silently dropped.
- **No silent coercion / dropping.** Invalid values are reported; exclusions must be
  explicit and auditable.
- **Allowlist-driven eligibility.** A numeric dtype is not a licence to enter the
  model matrix.
- **Read-only audits.** The audit copies inputs; the source CSVs are untouched.
- **Reproducibility.** Splits and exports are seeded and deterministic.
- **Separation.** Labels and identifiers never enter the feature matrix.

---

## 10. Limitations

- The dataset is synthetic; readiness checks validate *process*, not real-world
  detection performance.
- Missingness is reported but not yet handled downstream (imputation is a modelling
  decision for the next milestone).
- Group-by-scenario splits can be degenerate for anomaly detection; the tooling warns
  but does not choose for you.
- Correlation-based leakage is a heuristic (Pearson, numeric features only).
