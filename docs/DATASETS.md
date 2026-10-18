# Public Dataset Adapter Path and First-Dataset Decision

This document records the dataset-adapter decision for **Milestone 1**. It is a
documentation artifact only; no dataset is downloaded or committed.

> **Honesty note.** No public dataset was accessed or downloaded to produce this
> milestone. The adapter interface is implemented and exercised against a small
> **offline fixture** that mimics the Loghub HDFS raw-log format. The fixture
> lives in `tests/fixtures/hdfs_sample/` and is clearly labelled as synthetic.

## Interface

All adapters implement `anomaly_detector.dataset_adapters.base.BaseDatasetAdapter`:

```python
class BaseDatasetAdapter(ABC):
    name: str
    source: str
    license: str
    def can_load(self, path: str = None) -> bool: ...
    def load_data(self, path: str = None) -> Tuple[List[StructuredEvent], List[LabelInfo]]: ...
    def get_dataset_info(self) -> Dict[str, Any]: ...
```

Key contract:
- `load_data` returns **events and labels separately**, preserving the
  label/feature separation enforced throughout Milestone 1.
- `get_dataset_info` must document provenance, license, field mapping,
  label semantics, limitations and lossy conversions.
- Adapters must not fabricate provenance or claim data was downloaded.

## Implemented adapter: Loghub HDFS

- **Location:** `src/anomaly_detector/dataset_adapters/loghub_hdfs.py`
- **Source:** <https://github.com/logpai/loghub> (HDFS dataset)
- **Format:** `YYMMDD HHMMSS PID LEVEL COMPONENT: MESSAGE`
- **Assumed year:** 2008 (the raw format omits the year; configurable).

### Mapping to the common event format

| Common field | Source |
|---|---|
| `timestamp` | `YYMMDD HHMMSS` + assumed year, treated as UTC |
| `host_id` | constant (`hdfs-namenode` by default) |
| `service` | constant (`hdfs`) |
| `event_type` | `application_log` |
| `level` | raw level (normalized later by the validator) |
| `attributes.component` | log component |
| `attributes.block_id` | regex `blk_-?\d+` extracted from the message |
| `source_metadata.message` | free-text message (not used as a feature) |

### Labels

`anomaly_label.csv` maps `BlockId -> {Normal, Anomaly}`. These labels are
**block-level (instance-level), not time-window-level**. The adapter returns
them as separate `LabelInfo` rows; projecting them onto windows is a later-milestone
decision.

### Known limitations and leakage risks

- Raw timestamps have no year and no timezone (2008/UTC assumed).
- Block labels are instance-level; naive projection onto time windows risks
  **temporal leakage** across block boundaries.
- PID and the component hierarchy are flattened (lossy).
- Free-text `message` is retained only in `source_metadata`, never as a feature.

## Candidate datasets considered

| Dataset | Source | Labels | Why considered | Blocker / decision |
|---|---|---|---|---|
| **HDFS** (Loghub) | logpai/loghub | Block-level (`Normal`/`Anomaly`) | Very common log-anomaly benchmark; simple raw format | **Adapter implemented**; instance-level labels need careful window projection |
| **BGL** (Blue Gene/L) | logpai/loghub | Per-message alert labels | Has per-event timestamps and message-level labels | Not implemented in M1; message-level labels but large/heavy; deferred |
| **Thunderbird** | logpai/loghub | Per-message labels | Similar to BGL | Not implemented; very large; deferred |
| **OpenStack** | logpai/loghub | Instance-level (via `abnormal_label`) | Real cloud control-plane logs | Requires generating from source instances; heavier pipeline; deferred |
| **Windows / Linux / Apache / Zookeeper / HPC** | logpai/loghub | Mostly none or partial | Good unlabeled sources for real ingestion | No reliable labels for evaluation; useful later for ingestion testing |
| **CICIDS2017 / NSL-KDD** | UNB / KDD | Network attack labels | Well-known IDS datasets | **Not log data** (flow records); poor fit for the log-event contract |

## Exact blocker for a "verified first dataset"

There is **no network access assumed** in this environment and downloading a
multi-hundred-megabyte dataset is out of scope for a data-foundation milestone.
Rather than fabricate provenance, Milestone 1 provides:

1. the adapter interface,
2. one concrete adapter (Loghub HDFS) with an offline fixture test,
3. this documented candidate decision,
4. instructions to add a dataset later (below).

## How to add a public dataset later

```bash
# 1. Obtain the dataset from its official source and verify the license.
# 2. Convert it through the adapter (no downloads happen in this script):
python scripts/prepare_dataset.py --dataset loghub_hdfs --path /path/to/HDFS

# 3. Inspect the exported feature matrix, metadata, labels and quality report.
#    Outputs: data/processed/loghub_hdfs_*.csv, artifacts/dataset_adapter_status.json

# 4. Add a fixture + test mirroring tests/test_dataset_adapters.py.
```

To add a new dataset, subclass `BaseDatasetAdapter`, document its provenance in
`get_dataset_info`, register it in `scripts/prepare_dataset.py`, and add an
offline fixture test. Do **not** commit dataset files.
