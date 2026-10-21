# Milestone 3 — Isolation Forest Baseline

This document describes the first baseline anomaly detector: what it does, the exact
contract it implements, how to reproduce it, and — just as importantly — what it does
**not** prove.

> **Scope.** Milestone 3 covers **detection and scoring only**. There is no evidence
> correlation, no root-cause hypothesis, no incident narrative, no API, no dashboard
> and no cloud code. A flagged window is a *lead for investigation*, not proof of an
> attack. That reasoning layer belongs to a separate workstream.

---

## 1. Where it sits in the pipeline

```text
raw events
   │
   ▼  M1  preprocessing.py / features.py
feature windows + metadata + labels (kept separate)
   │
   ▼  M2  audit.py / leakage.py / splits.py / training_ready.py
data/processed/splits/<strategy>/{train,validation,test}/
   │       features.csv | labels.csv | metadata.csv
   ▼  M3  baseline_pipeline.py
IsolationForest → raw_score → threshold → is_anomaly
   │
   ├── model/model.joblib + model/manifest.json
   ├── predictions_<split>.csv
   └── reports/metrics.{json,md} + experiment_note.md
```

The baseline consumes **only** the Milestone 2 training-ready export. It never reads
raw logs and never re-derives features.

---

## 2. The model-input contract

The single most important rule: **a column reaches the estimator only because it is on
an explicit allowlist**, never because it happens to be numeric.

The allowlist lives in `configs/baseline.yaml` (`data.feature_columns`) and mirrors the
Milestone 2 eligible-feature list:

| Excluded from the model | Why |
|---|---|
| `is_anomaly` | ground truth — would be leakage |
| `scenario_name`, `description`, `intensity` | scenario metadata — would leak the label |
| `host_id`, `service`, `sample_key`, `event_id` | identifiers — not generalisable |
| `window_start`, `window_end`, `window_size_minutes` | timestamps / window geometry |

`baseline_config.py` **rejects** a config that puts any of those names into the
allowlist, so a mistake fails at load time rather than silently producing a
meaningless score.

### Schema stability

`model_contract.validate_feature_schema` enforces:

- every allowlisted column must be present — a missing column is an error;
- extra columns are tolerated (the feature table legitimately carries the sample key
  and metadata) but are never selected into the matrix;
- column **order** is normalised to the allowlist order, and the order is recorded in
  the manifest because it is part of the model's identity.

A scoring table whose columns differ from the trained schema fails with a clear
message; it is never coerced into shape.

---

## 3. Missing values and non-finite values

The Milestone 2 policy is **missing ≠ zero**, and it is preserved here.

| Policy | Behaviour |
|---|---|
| `median_impute` *(default)* | Missing values are replaced with the **training-split median** of that column. This is learned preprocessing, fitted on training rows only and applied unchanged to validation/test rows. It is never a zero-fill. |
| `drop` | Samples containing a missing value are excluded under a documented rule. The excluded sample keys are written to `excluded_<split>.csv` and counted in the metrics report — they are never silently counted as normal. |
| `error` | The run fails loudly with the affected columns and cell count. |

**Infinities (`±inf`) are always rejected** — they are never imputed, whatever the
policy — because an infinite value distorts every tree split it reaches. After the
policy is applied, the matrix is re-checked: `NaN` or `inf` reaching the estimator is
an internal error, not a warning.

A column whose values are *entirely* missing in the training rows has no usable
median; it is dropped from the model input, reported in the manifest
(`data_policy.imputation_statistics` / `training_data.dropped_columns`) and stays
dropped at scoring time.

> scikit-learn's `IsolationForest` silently accepts `NaN` in recent versions. The
> baseline does **not** rely on that: the missing-value policy is applied explicitly
> so the behaviour is version-independent and documented.

---

## 4. Training-data policy

| Policy | Rows the estimator is fitted on |
|---|---|
| `normal_only` *(default)* | Only training rows whose label is `False`. The model learns *normal* behaviour; labelled anomalies are excluded. |
| `all` | Every training row. |

Rows with an unknown label (`NaN`) are always excluded — an unlabelled window is not
evidence of normality.

The number of rows used, the number excluded and the label distribution are recorded
in the manifest (`training_data`).

**The estimator never sees validation or test rows.** This is enforced by the code
path (training reads only `train/`), and a regression test poisons the test split and
asserts that neither the model nor the threshold changes.

---

## 5. Score definition and threshold

### Score

| Property | Value |
|---|---|
| Name | `sklearn.ensemble.IsolationForest.score_samples` |
| Direction | **lower = more anomalous** |
| Flag rule | `is_anomaly = raw_score <= threshold` (inclusive at the boundary) |
| Is it a probability? | **No.** It is a ranking statistic. |

`decision_function` (positive = inlier) is also written to the predictions file for
consumers who prefer that convention, and `anomaly_score = -raw_score` is provided as
a convenience for "higher = more anomalous" readers.

### Threshold selection

The threshold is selected on the **validation** split only. Selecting it on the test
split raises `ModelContractError` — the guard is code, not a convention.

| Method | Rule |
|---|---|
| `validation_f1` *(default)* | The candidate threshold maximising F1 on the validation split. Ties break on recall, then on the lower (more conservative) threshold. |
| `validation_precision` | The most sensitive threshold (highest recall) that still meets `min_precision`. |
| `validation_recall` | The most precise threshold that still meets `min_recall`. |
| `train_quantile` | Operational heuristic: the `fallback_quantile` quantile of the **training** scores. Flags roughly that fraction of rows. Not a validated optimum. |

Candidate thresholds are every distinct validation score plus one below the minimum
("flag nothing") and one above the maximum ("flag everything"), so the search covers
both degenerate extremes.

### Documented fallback

If the requested method cannot be applied — labels missing, or the validation split
contains a single class — the **fallback** method (`train_quantile`) is used and the
result is marked `is_heuristic: true` with the reason recorded in the manifest. A
fallback threshold is never presented as if it had been validated.

---

## 6. Evaluation

Metrics are computed **per split**. Training metrics are never presented as test
performance, and a metric that cannot be computed is *skipped with a stated reason*
rather than reported as zero.

| Metric | Notes |
|---|---|
| `precision`, `recall`, `f1` | Computed from the confusion matrix; `0` when the denominator is zero. |
| `tp`, `fp`, `tn`, `fn` | Raw counts. |
| `false_positive_count`, `false_positive_rate` | Operational cost of alerting. |
| `prevalence` | Share of positive windows in the split. |
| `roc_auc`, `pr_auc` | Ranking metrics, computed on `-raw_score` so they read the intuitive way. **Skipped** when the split has a single class. |
| `alerts_per_host_hour` | Flagged windows ÷ observed host-hours (needs `window_size_minutes`). **Skipped** when the metadata is unavailable. |
| `detection_delay` | **Not computed.** The label contract carries no anomaly onset timestamp, and deriving one from the same features being modelled would leak the answer. The reason is recorded rather than the number. |

### Reference baselines

Two trivial references are reported for context (never as "the model"):

- `no_alert` — flag nothing (recall 0, F1 0).
- `train_quantile` — the label-free quantile heuristic.

### Synthetic-data caveat

Every metrics file states `data_nature: synthetic`. These numbers demonstrate that
the pipeline works end to end; they do **not** establish real-world detection
quality.

---

## 7. Artifacts

Written under `paths.output_dir` (default `artifacts/baseline/`):

| Path | Contents |
|---|---|
| `model/model.joblib` | The fitted estimator **plus its contract** (feature columns, imputation statistics, score definition, model config). |
| `model/manifest.json` | Model type + library version, UTC training timestamp, configuration, seed, ordered feature schema, data policy, training-data provenance, split info, score definition, threshold + selection method, git revision. |
| `predictions_<split>.csv` | Per sample: `sample_key`, `split`, window metadata, `raw_score`, `anomaly_score`, `decision_function`, `threshold`, `is_anomaly`, `true_label`. No raw log content, no secrets. |
| `excluded_<split>.csv` | Samples removed by the documented missing-value rule (only written when something was excluded). |
| `reports/metrics.json` | Machine-readable evaluation. |
| `reports/metrics.md` | Human-readable evaluation. |
| `experiment_note.md` | Limitations + the exact reproduction commands. |

Serialization uses **joblib** (the standard companion to scikit-learn). As with any
pickle-based format, `model.joblib` must only be loaded from a trusted source.

---

## 8. Commands

```bash
# Everything: train → score → evaluate → write artifacts
python scripts/run_baseline.py run

# Individual steps
python scripts/run_baseline.py train
python scripts/run_baseline.py score --split test
python scripts/run_baseline.py evaluate --split test

# With an explicit configuration
python scripts/run_baseline.py run --config configs/baseline.yaml
python scripts/run_baseline.py run --config configs/baseline.yaml --output-dir artifacts/baseline_v2
```

`--config` and `--output-dir` may be given before or after the subcommand. The command
exits with `0` on success and `2` on a handled error (missing split, schema mismatch,
missing model, …), printing the reason on stderr.

Full sequence from a clean checkout:

```bash
python scripts/generate_synthetic_data.py --config configs/default.yaml
python scripts/prepare_experiments.py
python scripts/run_baseline.py run
```

---

## 9. Configuration reference

`configs/baseline.yaml`:

| Key | Meaning |
|---|---|
| `model.n_estimators` | Number of trees (default 100). |
| `model.max_samples` | Samples per tree (default 256). scikit-learn clamps this to the training-set size when the set is smaller — that is expected behaviour, not a bug. |
| `model.contamination` | `"auto"` or a float in `(0, 0.5]`. |
| `model.random_state` | The seed — makes the run reproducible. |
| `model.n_jobs` | Parallelism for tree fitting. |
| `data.feature_columns` | The explicit model-input allowlist. |
| `data.missing_value_policy` | `median_impute` \| `drop` \| `error`. |
| `data.training_data_policy` | `normal_only` \| `all`. |
| `data.label_column` / `data.sample_key_column` | Column names used to join and label rows. |
| `data.metadata_columns` | Metadata copied into the predictions file (never modelled). |
| `threshold.method` | Threshold selection method (see §5). |
| `threshold.min_precision` / `min_recall` | Constraints for the constraint-based methods. |
| `threshold.fallback_method` / `fallback_quantile` | The documented fallback and its quantile. |
| `split.strategy` | Which `data/processed/splits/<strategy>/` to read. |
| `paths.*` | Input and output directories (relative to the repo root). |

Invalid configuration (an empty allowlist, a forbidden column in the allowlist, an
unknown policy, an out-of-range quantile) raises `ValueError` at load time.

---

## 10. Tests

`tests/test_baseline_config.py`, `tests/test_model_contract.py`,
`tests/test_baseline_model.py`, `tests/test_thresholding.py`,
`tests/test_evaluation.py` and `tests/test_baseline_pipeline.py` cover, among others:

- the config rejects forbidden columns, empty/duplicated allowlists and bad values;
- the schema check fails on a missing column and tolerates (but ignores) extras;
- imputation uses the training median and **never** zero; `drop` reports what it
  dropped; `error` fails loudly; infinities are always rejected;
- the score direction is "lower = more anomalous" and the threshold boundary is
  inclusive;
- the **test split is refused** as a threshold source; a single-class validation split
  falls back to the documented heuristic;
- metrics handle single-class splits, missing labels and no-flagged-samples without
  pretending a value was computed;
- poisoning the test split changes neither the model nor the threshold, while changing
  the validation labels *does* change the threshold;
- the CLI exits non-zero with a clear message when the model, a split or the
  predictions are missing.

All tests run **offline**.

---

## 11. Limitations and honest caveats

- **Synthetic data.** The evaluation uses synthetic windows with synthetic labels. The
  numbers validate the pipeline, not the approach.
- **No hyperparameter search.** `n_estimators`, `max_samples` and `contamination` are
  fixed configuration values. Tuning belongs to a later milestone.
- **Small data.** The training split has ~110 usable windows after the `normal_only`
  policy; Isolation Forest's `max_samples=256` is therefore clamped to the training
  size. Results will move as the dataset grows.
- **Single-class and degenerate folds.** The `time` and `group` split strategies can
  produce folds with no anomalies (Milestone 2 warns about this). The baseline reports
  such metrics as skipped rather than as zero.
- **The score is not a probability.** Do not present `raw_score` as a confidence.
- **Detection delay is not measured** (no onset timestamps in the label contract).
- **No reasoning.** No evidence correlation, no root cause, no explanation.
- **Joblib artifacts are pickle-based** — load them only from a trusted source.

---

## 12. Boundary with the next milestone

Milestone 3 delivers a reproducible, leakage-free baseline with a stable output
contract. The next milestone can add hyperparameter search, a statistical baseline
(e.g. robust z-score / EWMA on the same features) and a model comparison — all of
which must reuse the *same* contract: the same allowlist, the same missing-value
policy, the same score direction and the same train/validation/test discipline.
