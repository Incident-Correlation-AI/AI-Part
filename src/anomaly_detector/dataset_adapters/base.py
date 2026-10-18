"""Minimal dataset-adapter interface for public log datasets.

All adapters must convert their source format into the common
:class:`~anomaly_detector.schemas.StructuredEvent` representation and return any
available ground-truth labels **separately** from events, so the label/feature
separation enforced in Milestone 1 is preserved end to end.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from ..schemas import StructuredEvent, LabelInfo


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Subclasses must not fabricate provenance. If a dataset cannot be accessed
    or its license/format cannot be verified, the adapter should record the
    exact blocker via :meth:`get_dataset_info`.
    """

    #: Human-readable dataset name.
    name: str = "unnamed"
    #: Official source URL.
    source: str = ""
    #: License / usage restrictions.
    license: str = ""

    @abstractmethod
    def can_load(self, path: str = None) -> bool:
        """Return True if the adapter can load data from ``path``."""

    @abstractmethod
    def load_data(self, path: str = None) -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Load and convert a dataset into common events and separate labels."""

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return documented metadata about the dataset and the mapping."""
