"""Preprocessing and window aggregation."""
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from .schemas import StructuredEvent, FeatureWindow, WindowMetadata

logger = logging.getLogger(__name__)

def aggregate_events_to_windows(events: List[StructuredEvent], 
                               window_size_minutes: int = 1,
                               allow_empty_windows: bool = False) -> List[FeatureWindow]:
    """
    Aggregate events into time windows per host/service pair.
    Do not invent observations for empty windows by default.
    """
    if not events:
        return []
    
    # Group events by (host_id, service)
    grouped = defaultdict(list)
    for ev in events:
        key = (ev.host_id, ev.service)
        grouped[key].append(ev)
    
    windows = []
    window_delta = timedelta(minutes=window_size_minutes)
    
    for (host_id, service), ev_list in grouped.items():
        # Sort events by timestamp
        ev_list.sort(key=lambda e: e.timestamp)
        
        if not ev_list:
            continue
        
        # Determine window boundaries
        start_time = ev_list[0].timestamp.replace(second=0, microsecond=0)
        # Align to window
        minute = start_time.minute
        aligned_minute = (minute // window_size_minutes) * window_size_minutes
        start_time = start_time.replace(minute=aligned_minute)
        
        end_time = ev_list[-1].timestamp
        # Create windows covering the range
        current_start = start_time
        while current_start < end_time + window_delta:
            current_end = current_start + window_delta
            # Collect events in this window
            window_events = []
            for ev in ev_list:
                if current_start <= ev.timestamp < current_end:
                    window_events.append(ev)
            # Create window if events exist or if empty windows allowed
            if window_events or allow_empty_windows:
                metadata = WindowMetadata(
                    window_start=current_start,
                    window_end=current_end,
                    host_id=host_id,
                    service=service,
                    window_size_minutes=window_size_minutes
                )
                windows.append(create_feature_window(metadata, window_events))
            current_start = current_end
            # Prevent infinite loop
            if current_start > end_time + timedelta(hours=1):  # safety
                break
    
    # Sort windows
    windows.sort(key=lambda w: (w.metadata.host_id, w.metadata.service, w.metadata.window_start))
    return windows

def create_feature_window(metadata: WindowMetadata, events: List[StructuredEvent]) -> FeatureWindow:
    """Create feature window from events."""
    fw = FeatureWindow(metadata=metadata)
    fw.event_count = len(events)
    
    if not events:
        return fw
    
    # Count various event types
    error_levels = {"ERROR", "FATAL", "CRITICAL"}
    fw.error_count = sum(1 for e in events if e.level in error_levels)
    fw.error_eligible_count = fw.event_count  # All events eligible for error ratio
    if fw.error_eligible_count > 0:
        fw.error_ratio = fw.error_count / fw.error_eligible_count
    else:
        fw.error_ratio = None
    
    # HTTP requests
    http_events = [e for e in events if e.event_type == "http_request" or "http" in e.event_type.lower()]
    fw.http_request_count = len(http_events)
    fw.http_request_eligible_count = fw.http_request_count
    
    # HTTP 5xx
    http_5xx = 0
    for e in http_events:
        status = e.attributes.get("status_code")
        if status is not None:
            try:
                sc = int(status)
                if sc >= 500 and sc < 600:
                    http_5xx += 1
            except Exception:
                pass
    fw.http_5xx_count = http_5xx
    if fw.http_request_eligible_count > 0:
        fw.http_5xx_ratio = fw.http_5xx_count / fw.http_request_eligible_count
    else:
        fw.http_5xx_ratio = None
    
    # Auth failures
    fw.auth_failure_count = sum(1 for e in events if e.event_type == "auth_failure" or "auth" in e.event_type)
    
    # Restarts
    fw.restart_count = sum(1 for e in events if e.event_type == "service_restart" or "restart" in e.event_type)
    
    # Latency stats
    latencies = []
    for e in events:
        lat = e.attributes.get("latency_ms")
        if lat is not None:
            try:
                lat_val = float(lat)
                if lat_val >= 0:  # Valid (non-negative)
                    latencies.append(lat_val)
            except Exception:
                pass
    fw.latency_sample_count = len(latencies)
    if latencies:
        fw.mean_latency_ms = sum(latencies) / len(latencies)
        # Calculate p95
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        if p95_idx >= len(sorted_lat):
            p95_idx = len(sorted_lat) - 1
        fw.p95_latency_ms = sorted_lat[p95_idx]
    else:
        fw.mean_latency_ms = None
        fw.p95_latency_ms = None
    
    return fw
