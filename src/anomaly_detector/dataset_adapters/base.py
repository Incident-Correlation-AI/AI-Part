"""Base adapter interface for public datasets."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from ..schemas import StructuredEvent, LabelInfo

class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""
    
    @abstractmethod
    def load_data(self, path: str = None) -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Load and convert dataset to common event format."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset metadata including license, source, etc."""
        pass
    
    @abstractmethod
    def can_load(self) -> bool:
        """Check if dataset is accessible."""
        pass
