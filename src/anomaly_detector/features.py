"""Feature extraction utilities."""
from typing import List, Dict, Any, Tuple
from .schemas import FeatureWindow, LabelInfo
import pandas as pd

def windows_to_feature_dataframe(windows: List[FeatureWindow]) -> pd.DataFrame:
    """Convert windows to feature dataframe (no metadata)."""
    if not windows:
        return pd.DataFrame()
    features = [w.to_features_dict() for w in windows]
    return pd.DataFrame(features)

def windows_to_metadata_dataframe(windows: List[FeatureWindow]) -> pd.DataFrame:
    """Convert windows to metadata dataframe."""
    if not windows:
        return pd.DataFrame()
    metadata = [w.to_metadata_dict() for w in windows]
    return pd.DataFrame(metadata)

def labels_to_dataframe(labels: List[LabelInfo]) -> pd.DataFrame:
    """Convert labels to dataframe."""
    if not labels:
        return pd.DataFrame()
    label_dicts = [l.to_dict() for l in labels]
    return pd.DataFrame(label_dicts)

def check_label_leakage(feature_df: pd.DataFrame) -> List[str]:
    """Check for label leakage in feature dataframe."""
    leakage_indicators = ["is_anomaly", "anomaly", "label", "scenario", "ground_truth", "true_label"]
    leaks = []
    for col in feature_df.columns:
        col_lower = str(col).lower()
        for indicator in leakage_indicators:
            if indicator in col_lower:
                leaks.append(col)
                break
    return leaks
