"""Reproducible synthetic data generation."""
import random
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Any
import logging

from .schemas import StructuredEvent, FeatureWindow, LabelInfo, WindowMetadata
from .preprocessing import aggregate_events_to_windows, create_feature_window

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
    
    def generate_stable_traffic(self, start_time: datetime, duration_hours: float = 2,
                              host_id: str = "web-01", service: str = "web-api",
                              events_per_minute: int = 30) -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate stable traffic with normal variation."""
        events = []
        labels = []
        current = start_time
        end = start_time + timedelta(hours=duration_hours)
        event_id = 0
        
        # Normal windows get normal label pattern - but labels are per window
        while current < end:
            # Add small variation
            eps = self.rng.uniform(-5, 5)
            n_events = max(1, int(events_per_minute + eps))
            for _ in range(n_events):
                latency = self.rng.gauss(50, 10)
                status = 200 if self.rng.random() < 0.98 else 404  # Mostly OK
                event = StructuredEvent(
                    event_id=f"evt-stable-{event_id:08d}",
                    timestamp=current + timedelta(seconds=self.rng.uniform(0, 59)),
                    host_id=host_id,
                    service=service,
                    event_type="http_request",
                    level="INFO",
                    attributes={"status_code": status, "latency_ms": max(10, latency)}
                )
                events.append(event)
                event_id += 1
            current += timedelta(minutes=1)
        
        # Create window labels
        w = start_time.replace(second=0, microsecond=0)
        while w < end:
            labels.append(LabelInfo(
                window_start=w,
                window_end=w + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=False,
                scenario_name="stable_traffic",
                description="Normal stable traffic with natural variation"
            ))
            w += timedelta(minutes=1)
        
        return events, labels
    
    def generate_http_5xx_surge(self, start_time: datetime, surge_duration_minutes: int = 10,
                              intensity: float = 0.5, host_id: str = "web-01",
                              service: str = "web-api") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate HTTP 5xx surge anomaly."""
        events = []
        labels = []
        current = start_time
        end = start_time + timedelta(minutes=surge_duration_minutes)
        event_id = 0
        
        while current < end:
            n_events = int(self.rng.uniform(20, 60))
            for _ in range(n_events):
                # High 5xx rate during surge
                if self.rng.random() < intensity:
                    status = 500 + self.rng.choice([0, 1, 2])  # 500, 501, 502
                    level = "ERROR"
                else:
                    status = 200
                    level = "INFO"
                latency = self.rng.gauss(200, 50)
                event = StructuredEvent(
                    event_id=f"evt-5xx-{event_id:08d}",
                    timestamp=current + timedelta(seconds=self.rng.uniform(0, 59)),
                    host_id=host_id,
                    service=service,
                    event_type="http_request",
                    level=level,
                    attributes={"status_code": status, "latency_ms": max(50, latency)}
                )
                events.append(event)
                event_id += 1
            current += timedelta(minutes=1)
        
        w = start_time
        while w < end:
            labels.append(LabelInfo(
                window_start=w,
                window_end=w + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=True,
                scenario_name="http_5xx_surge",
                intensity=intensity,
                description=f"HTTP 5xx surge with {intensity*100:.0f}% error rate"
            ))
            w += timedelta(minutes=1)
        
        return events, labels
    
    def generate_auth_failure_surge(self, start_time: datetime, surge_duration_minutes: int = 5,
                                   host_id: str = "web-01", service: str = "auth") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate authentication failure surge."""
        events = []
        labels = []
        current = start_time
        end = start_time + timedelta(minutes=surge_duration_minutes)
        event_id = 0
        
        while current < end:
            n_auth = int(self.rng.uniform(10, 30))
            for _ in range(n_auth):
                event = StructuredEvent(
                    event_id=f"evt-auth-{event_id:08d}",
                    timestamp=current + timedelta(seconds=self.rng.uniform(0, 59)),
                    host_id=host_id,
                    service=service,
                    event_type="auth_failure",
                    level="ERROR",
                    attributes={}
                )
                events.append(event)
                event_id += 1
            current += timedelta(minutes=1)
        
        w = start_time
        while w < end:
            labels.append(LabelInfo(
                window_start=w,
                window_end=w + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=True,
                scenario_name="auth_failure_surge",
                description="Authentication failure surge"
            ))
            w += timedelta(minutes=1)
        
        return events, labels

    def generate_latency_increase(self, start_time: datetime, duration_minutes: int = 15,
                                host_id: str = "web-01", service: str = "web-api") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate latency increase anomaly."""
        events = []
        labels = []
        current = start_time
        end = start_time + timedelta(minutes=duration_minutes)
        event_id = 0
        
        while current < end:
            n_events = int(self.rng.uniform(20, 40))
            for _ in range(n_events):
                latency = self.rng.gauss(500, 100)  # High latency
                status = 200 if self.rng.random() < 0.9 else 504
                event = StructuredEvent(
                    event_id=f"evt-lat-{event_id:08d}",
                    timestamp=current + timedelta(seconds=self.rng.uniform(0, 59)),
                    host_id=host_id,
                    service=service,
                    event_type="http_request",
                    level="WARNING" if latency > 400 else "INFO",
                    attributes={"status_code": status, "latency_ms": max(100, latency)}
                )
                events.append(event)
                event_id += 1
            current += timedelta(minutes=1)
        
        w = start_time
        while w < end:
            labels.append(LabelInfo(
                window_start=w,
                window_end=w + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=True,
                scenario_name="latency_increase",
                description="Abnormal latency increase"
            ))
            w += timedelta(minutes=1)
        
        return events, labels
    
    def generate_volume_spike(self, start_time: datetime, duration_minutes: int = 10,
                             host_id: str = "web-01", service: str = "web-api") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate abnormal event volume spike."""
        events = []
        labels = []
        current = start_time
        end = start_time + timedelta(minutes=duration_minutes)
        event_id = 0
        
        while current < end:
            n_events = int(self.rng.uniform(100, 200))  # High volume
            for _ in range(n_events):
                latency = self.rng.gauss(80, 20)
                status = 200 if self.rng.random() < 0.97 else 503
                event = StructuredEvent(
                    event_id=f"evt-vol-{event_id:08d}",
                    timestamp=current + timedelta(seconds=self.rng.uniform(0, 59)),
                    host_id=host_id,
                    service=service,
                    event_type="http_request",
                    level="INFO",
                    attributes={"status_code": status, "latency_ms": max(20, latency)}
                )
                events.append(event)
                event_id += 1
            current += timedelta(minutes=1)
        
        w = start_time
        while w < end:
            labels.append(LabelInfo(
                window_start=w,
                window_end=w + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=True,
                scenario_name="volume_spike",
                description="Abnormal event volume spike"
            ))
            w += timedelta(minutes=1)
        
        return events, labels
    
    def generate_restart_surge(self, start_time: datetime, count: int = 5,
                              host_id: str = "web-01", service: str = "web-api") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate repeated service restarts."""
        events = []
        labels = []
        event_id = 0
        for i in range(count):
            ts = start_time + timedelta(minutes=i)
            event = StructuredEvent(
                event_id=f"evt-restart-{event_id:08d}",
                timestamp=ts,
                host_id=host_id,
                service=service,
                event_type="service_restart",
                level="WARNING",
                attributes={"reason": "unexpected"}
            )
            events.append(event)
            event_id += 1
            labels.append(LabelInfo(
                window_start=ts.replace(second=0, microsecond=0),
                window_end=ts.replace(second=0, microsecond=0) + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=True,
                scenario_name="restart_surge",
                description=f"Repeated service restart #{i+1}"
            ))
        return events, labels

    def generate_occasional_errors(self, start_time: datetime, duration_hours: float = 1,
                               host_id: str = "web-01", service: str = "web-api") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate occasional isolated errors (normal)."""
        events = []
        labels = []
        current = start_time
        end = start_time + timedelta(hours=duration_hours)
        event_id = 0
        
        while current < end:
            n_events = int(self.rng.uniform(15, 25))
            for _ in range(n_events):
                # Rare errors are normal
                if self.rng.random() < 0.05:  # 5% error rate - occasional
                    status = self.rng.choice([404, 500])  # Some errors
                    level = "ERROR" if status >= 500 else "WARNING"
                else:
                    status = 200
                    level = "INFO"
                latency = self.rng.gauss(60, 15)
                event = StructuredEvent(
                    event_id=f"evt-iso-{event_id:08d}",
                    timestamp=current + timedelta(seconds=self.rng.uniform(0, 59)),
                    host_id=host_id,
                    service=service,
                    event_type="http_request",
                    level=level,
                    attributes={"status_code": status, "latency_ms": max(10, latency)}
                )
                events.append(event)
                event_id += 1
            current += timedelta(minutes=1)
        
        w = start_time
        while w < end:
            labels.append(LabelInfo(
                window_start=w,
                window_end=w + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=False,
                scenario_name="occasional_errors",
                description="Normal traffic with occasional isolated errors"
            ))
            w += timedelta(minutes=1)
        return events, labels
    
    def generate_low_traffic(self, start_time: datetime, duration_hours: float = 1,
                           host_id: str = "web-01", service: str = "web-api") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate low traffic periods (normal)."""
        events = []
        labels = []
        current = start_time
        end = start_time + timedelta(hours=duration_hours)
        event_id = 0
        
        while current < end:
            n_events = int(self.rng.uniform(0, 5))  # Very low
            for _ in range(n_events):
                latency = self.rng.gauss(40, 10)
                event = StructuredEvent(
                    event_id=f"evt-low-{event_id:08d}",
                    timestamp=current + timedelta(seconds=self.rng.uniform(0, 59)),
                    host_id=host_id,
                    service=service,
                    event_type="http_request",
                    level="INFO",
                    attributes={"status_code": 200, "latency_ms": max(10, latency)}
                )
                events.append(event)
                event_id += 1
            current += timedelta(minutes=1)
        
        w = start_time
        while w < end:
            labels.append(LabelInfo(
                window_start=w,
                window_end=w + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=False,
                scenario_name="low_traffic",
                description="Low traffic quiet period"
            ))
            w += timedelta(minutes=1)
        return events, labels
    
    def generate_missing_delayed(self, start_time: datetime, duration_minutes: int = 10,
                                host_id: str = "web-01", service: str = "web-api") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate scenario with sparse observations (normal)."""
        events = []
        labels = []
        current = start_time
        end = start_time + timedelta(minutes=duration_minutes)
        event_id = 0
        
        # Only generate events in some minutes
        while current < end:
            if self.rng.random() < 0.3:  # 30% of minutes have data
                n_events = int(self.rng.uniform(1, 3))
                for _ in range(n_events):
                    latency = self.rng.gauss(50, 10)
                    event = StructuredEvent(
                        event_id=f"evt-miss-{event_id:08d}",
                        timestamp=current + timedelta(seconds=self.rng.uniform(0, 59)),
                        host_id=host_id,
                        service=service,
                        event_type="http_request",
                        level="INFO",
                        attributes={"status_code": 200, "latency_ms": max(10, latency)}
                    )
                    events.append(event)
                    event_id += 1
            current += timedelta(minutes=1)
        
        w = start_time
        while w < end:
            labels.append(LabelInfo(
                window_start=w,
                window_end=w + timedelta(minutes=1),
                host_id=host_id,
                service=service,
                is_anomaly=False,
                scenario_name="missing_delayed",
                description="Missing or delayed observations"
            ))
            w += timedelta(minutes=1)
        return events, labels
    
    def generate_restart_normal(self, start_time: datetime, host_id: str = "web-01",
                               service: str = "web-api") -> Tuple[List[StructuredEvent], List[LabelInfo]]:
        """Generate normal occasional restart."""
        events = []
        labels = []
        event = StructuredEvent(
            event_id="evt-restart-norm-001",
            timestamp=start_time,
            host_id=host_id,
            service=service,
            event_type="service_restart",
            level="INFO",
            attributes={"reason": "scheduled"}
        )
        events.append(event)
        w = start_time.replace(second=0, microsecond=0)
        labels.append(LabelInfo(
            window_start=w,
            window_end=w + timedelta(minutes=1),
            host_id=host_id,
            service=service,
            is_anomaly=False,
            scenario_name="normal_restart",
            description="Normal scheduled restart"
        ))
        return events, labels

    def generate_full_synthetic_dataset(self, base_time: datetime = None) -> Dict[str, Any]:
        """Generate complete synthetic dataset with all scenarios."""
        if base_time is None:
            base_time = datetime(2026, 10, 18, 10, 0, 0, tzinfo=timezone.utc)
        
        all_events = []
        all_labels = []
        
        # Normal scenarios
        ev, lbl = self.generate_stable_traffic(base_time + timedelta(hours=0), duration_hours=1)
        all_events.extend(ev); all_labels.extend(lbl)
        
        ev, lbl = self.generate_occasional_errors(base_time + timedelta(hours=2), duration_hours=1)
        all_events.extend(ev); all_labels.extend(lbl)
        
        ev, lbl = self.generate_low_traffic(base_time + timedelta(hours=4), duration_hours=1)
        all_events.extend(ev); all_labels.extend(lbl)
        
        ev, lbl = self.generate_missing_delayed(base_time + timedelta(hours=6), duration_minutes=30)
        all_events.extend(ev); all_labels.extend(lbl)
        
        ev, lbl = self.generate_restart_normal(base_time + timedelta(hours=8))
        all_events.extend(ev); all_labels.extend(lbl)
        
        # Anomalous scenarios
        ev, lbl = self.generate_http_5xx_surge(base_time + timedelta(hours=1), surge_duration_minutes=10, intensity=0.8)
        all_events.extend(ev); all_labels.extend(lbl)
        
        ev, lbl = self.generate_auth_failure_surge(base_time + timedelta(hours=3), surge_duration_minutes=5)
        all_events.extend(ev); all_labels.extend(lbl)
        
        ev, lbl = self.generate_restart_surge(base_time + timedelta(hours=5), count=5)
        all_events.extend(ev); all_labels.extend(lbl)
        
        ev, lbl = self.generate_latency_increase(base_time + timedelta(hours=7), duration_minutes=15)
        all_events.extend(ev); all_labels.extend(lbl)
        
        ev, lbl = self.generate_volume_spike(base_time + timedelta(hours=9), duration_minutes=10)
        all_events.extend(ev); all_labels.extend(lbl)
        
        return {
            "events": all_events,
            "labels": all_labels,
            "scenarios": self._get_scenario_list(),
            "metadata": {
                "seed": self.seed,
                "base_time": base_time.isoformat(),
                "total_events": len(all_events),
                "total_labels": len(all_labels)
            }
        }
    
    def _get_scenario_list(self) -> List[str]:
        return [
            "stable_traffic", "occasional_errors", "low_traffic", 
            "missing_delayed", "normal_restart", "http_5xx_surge",
            "auth_failure_surge", "restart_surge", "latency_increase", "volume_spike"
        ]