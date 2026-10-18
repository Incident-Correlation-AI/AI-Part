"""Loghub HDFS dataset adapter.

Dataset
-------
* Name: **HDFS** (from the Loghub collection).
- Source: https://github.com/logpai/loghub (``HDFS/`` directory).
- Citation: He et al., "Drain: An Online Log Parsing Approach with Fixed Depth
  Tree", ICWS 2017; Xu et al., "Detecting Large-Scale System Problems by Mining
  Console Logs" (LogPal), 2009.
- License: the Loghub collection is distributed for research use; the HDFS data
  originates from a Hadoop benchmark. See the Loghub repository for the current
  license statement. **Always re-verify the license before redistribution.**

Raw log line format (``HDFS.log``)::

    081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating

which decomposes as::

    YYMMDD HHMMSS PID LEVEL COMPONENT: MESSAGE

Because the raw format has no year, this adapter assumes the dataset year
**2008** (Loghub documents the HDFS logs as collected in 2008). This assumption
is configurable and is recorded in :meth:`get_dataset_info`.

Labels
------
Loghub ships ``anomaly_label.csv`` with columns ``BlockId,Label`` where
``Label`` is ``Normal`` or ``Anomaly``. Labels are **block-level**, not
time-window-level. This adapter returns them as separate
:class:`~anomaly_detector.schemas.LabelInfo` rows keyed by block id in the
``description`` field with ``host_id``/``service`` left empty, because a block
does not map one-to-one to a window. Downstream code must decide how to project
block labels onto windows; doing so is explicitly out of scope for Milestone 1.

Known limitations / leakage risks
---------------------------------
* No per-event timestamps at second resolution beyond ``YYMMDD HHMMSS`` (no
  timezone information; treated as UTC).
* Block labels are instance-level. Projecting them onto windows risks temporal
  leakage if windows overlap block boundaries; that mapping must be handled
  carefully in a later milestone.
* ``content`` is free text; MessageId/EventTemplate are *not* used as features.
"""
import csv
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

from ..schemas import StructuredEvent, LabelInfo
from .base import BaseDatasetAdapter

# 081109 203615 148 INFO dfs.DataNode$PacketResponder: message
_HDFS_LINE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+(?P<component>[^:]+):\s?(?P<message>.*)$"
)
_BLOCK_ID = re.compile(r"(blk_-?\d+)")


class LoghubHDFSAdapter(BaseDatasetAdapter):
    name = "Loghub HDFS"
    source = "https://github.com/logpai/loghub"
    license = "Loghub research datasets - verify current terms at the source URL"

    def __init__(self, year: int = 2008, host_id: str = "hdfs-namenode",
                 service: str = "hdfs"):
        self.year = year
        self.host_id = host_id
        self.service = service

    # ------------------------------------------------------------------ #
    def can_load(self, path: str = None) -> bool:
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for _ in range(5):
                    line = fh.readline()
                    if not line:
                        break
                    if _HDFS_LINE.match(line.rstrip("\n")):
                        return True
                return False
        except OSError:
            return False

    # ------------------------------------------------------------------ #
    def _parse_line(self, line: str, index: int) -> Optional[StructuredEvent]:
        match = _HDFS_LINE.match(line.rstrip("\n"))
        if not match:
            return None
        date, time = match.group("date"), match.group("time")
        try:
            # ``date`` is YYMMDD; drop YY and prepend the assumed year.
            ts = datetime.strptime(f"{self.year:04d}{date[2:]} {time}", "%Y%m%d %H%M%S")
        except ValueError:
            return None
        ts = ts.replace(tzinfo=timezone.utc)
        level = match.group("level")
        component = match.group("component").strip()
        message = match.group("message")
        block_match = _BLOCK_ID.search(message)
        attributes: Dict[str, Any] = {"component": component}
        if block_match:
            attributes["block_id"] = block_match.group(1)
        return StructuredEvent(
            event_id=f"hdfs-{index:08d}",
            timestamp=ts,
            host_id=self.host_id,
            service=self.service,
            event_type="application_log",
            level=level,
            attributes=attributes,
            source_metadata={"component": component, "message": message, "dataset": "loghub_hdfs"},
        )

    # ------------------------------------------------------------------ #
    def load_data(self, path: str = None) -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Load HDFS raw log lines and (optionally) block labels.

        ``path`` may be either the raw ``.log`` file or a directory containing
        ``HDFS.log`` and optionally ``anomaly_label.csv``.
        """
        log_path, label_path = self._resolve_paths(path)
        events: List[StructuredEvent] = []
        if log_path and os.path.isfile(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if not line.strip():
                        continue
                    ev = self._parse_line(line, i)
                    if ev is not None:
                        events.append(ev)

        labels: List[LabelInfo] = []
        if label_path and os.path.isfile(label_path):
            labels = self._load_block_labels(label_path)
        return events, labels

    # ------------------------------------------------------------------ #
    def _resolve_paths(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        if path is None:
            return None, None
        if os.path.isdir(path):
            log = os.path.join(path, "HDFS.log")
            label = os.path.join(path, "anomaly_label.csv")
            return (log if os.path.isfile(log) else None,
                    label if os.path.isfile(label) else None)
        return path, None

    def _load_block_labels(self, label_path: str) -> List[LabelInfo]:
        labels: List[LabelInfo] = []
        with open(label_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                block_id = (row.get("BlockId") or "").strip()
                raw = (row.get("Label") or "").strip()
                if not block_id:
                    continue
                is_anomaly = raw.lower() == "anomaly"
                labels.append(LabelInfo(
                    window_start=datetime(1970, 1, 1, tzinfo=timezone.utc),
                    window_end=datetime(1970, 1, 1, tzinfo=timezone.utc),
                    host_id="",
                    service="",
                    is_anomaly=is_anomaly,
                    scenario_name="hdfs_block_label",
                    description=f"block_id={block_id};raw_label={raw}",
                ))
        return labels

    # ------------------------------------------------------------------ #
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "license": self.license,
            "format": "HDFS raw log: 'YYMMDD HHMMSS PID LEVEL COMPONENT: MESSAGE'",
            "year_assumption": self.year,
            "fields": ["timestamp", "pid", "level", "component", "message", "block_id (extracted)"],
            "label_semantics": (
                "anomaly_label.csv maps BlockId -> {'Normal','Anomaly'}; "
                "labels are block-level (instance-level), NOT time-window-level."
            ),
            "mapping": {
                "timestamp": "YYMMDD HHMMSS parsed with assumed year, treated as UTC",
                "level": "raw level normalized by the common validator",
                "component": "attributes.component",
                "block_id": "extracted from message via regex 'blk_-?\\d+'",
                "event_type": "application_log",
            },
            "limitations": [
                "No year in raw timestamps; 2008 assumed (configurable).",
                "No timezone in raw logs; UTC assumed.",
                "Block labels are instance-level and require careful projection "
                "onto windows to avoid leakage.",
                "Free-text message is retained only in source_metadata, not used as a feature.",
            ],
            "lossy_conversions": [
                "PID is not preserved as a field (only in message).",
                "Original component hierarchy flattened to a string.",
            ],
        }
