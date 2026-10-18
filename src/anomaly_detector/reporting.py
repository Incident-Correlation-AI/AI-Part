"""Data quality reporting."""
from typing import List, Dict, Any
from datetime import datetime
import json

from .schemas import FeatureWindow, LabelInfo
from .validation import ValidationResult

def generate_data_quality_report(validation_result: ValidationResult,
                                windows: List[FeatureWindow],
                                labels: List[LabelInfo] = None) -> Dict[str, Any]:
    """Generate comprehensive data quality report."""
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input": {
            "total_processed": validation_result.total_processed,
            "accepted_count": len(validation_result.accepted),
            "rejected_count": len(validation_result.rejected),
            "duplicated_count": len(validation_result.duplicated),
            "partially_usable_count": len(validation_result.partially_usable)
        },
        "rejection_reasons": {},
        "windows": {
            "count": len(windows),
            "hosts": set(w.host_id for w in windows),
            "services": set(w.service for w in windows)
        },
        "labels": {
            "count": len(labels) if labels else 0,
            "anomaly_count": sum(1 for l in (labels or []) if l.is_anomaly),
            "normal_count": sum(1 for l in (labels or []) if not l.is_anomaly),
            "scenario_distribution": {}
        },
        "features": {
            "missingness": {},
            "availability": {}
        }
    }
    
    # Count rejection reasons
    for rej in validation_result.rejected:
        for reason in rej.get("reasons", []):
            key = reason[:80]  # Truncate long reasons
            report["rejection_reasons"][key] = report["rejection_reasons"].get(key, 0) + 1
    
    # Window stats
    if windows:
        report["windows"]["num_hosts"] = len(report["windows"]["hosts"])
        report["windows"]["num_services"] = len(report["windows"]["services"])
        report["windows"]["time_range"] = {
            "start": min(w.metadata.window_start for w in windows).isoformat(),
            "end": max(w.metadata.window_end for w in windows).isoformat()
        }
        if windows[0].metadata.window_size_minutes:
            report["windows"]["window_size_minutes"] = windows[0].metadata.window_size_minutes
    else:
        report["windows"]["num_hosts"] = 0
        report["windows"]["num_services"] = 0
    
    # Label distribution
    if labels:
        for lbl in labels:
            sc = lbl.scenario_name
            report["labels"]["scenario_distribution"][sc] = report["labels"]["scenario_distribution"].get(sc, 0) + 1
    
    return report
