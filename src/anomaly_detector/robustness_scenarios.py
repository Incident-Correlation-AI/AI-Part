"""Robustness scenario generators for Milestone 5.

All generators are deterministic with seeds. Scenario labels are NEVER placed
into model features - they stay in separate metadata for evaluation only.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

FEATURES = ["event_count","error_count","error_ratio","http_request_count","http_5xx_count","http_5xx_ratio",
            "auth_failure_count","restart_count","mean_latency_ms","p95_latency_ms","latency_sample_count",
            "http_request_eligible_count","error_eligible_count"]


@dataclass
class ScenarioCase:
    features: pd.DataFrame
    labels: pd.DataFrame  # is_anomaly + scenario metadata
    meta: pd.DataFrame
    notes: str


def _base_rows(rng, n, host="r-host", svc="r-svc"):
    starts = pd.date_range("2026-03-01", periods=n, freq="min", tz="UTC")
    meta = pd.DataFrame({
        "sample_key":[f"{host}|{svc}|{s.isoformat()}" for s in starts],
        "window_start":[s.isoformat() for s in starts],
        "window_end":[(s+pd.Timedelta(minutes=1)).isoformat() for s in starts],
        "host_id":[host]*n, "service":[svc]*n, "window_size_minutes":[1]*n
    })
    labels = pd.DataFrame({"sample_key":meta["sample_key"], "is_anomaly":[False]*n, "scenario":["normal_variation"]*n, "scenario_group":["normal"]*n})
    return meta, labels


def gen_normal_variation(seed=12345, n=40):
    rng = np.random.RandomState(seed)
    meta, labels = _base_rows(rng, n)
    feats = pd.DataFrame({c: rng.poisson(15, n).astype(float) for c in FEATURES})
    feats["error_count"]=rng.poisson(1,n).astype(float); feats["error_ratio"]=feats["error_count"]/feats["event_count"].clip(lower=1)
    feats["http_request_count"]=rng.poisson(15,n).astype(float); feats["http_5xx_count"]=rng.poisson(0,n).astype(float)
    feats["http_5xx_ratio"]=feats["http_5xx_count"]/feats["http_request_count"].clip(lower=1)
    feats["mean_latency_ms"]=rng.normal(55,8,n).clip(10); feats["p95_latency_ms"]=feats["mean_latency_ms"]*1.3
    feats["latency_sample_count"]=feats["http_request_count"]
    feats["http_request_eligible_count"]=feats["http_request_count"]; feats["error_eligible_count"]=feats["event_count"]
    return ScenarioCase(feats, labels, meta, "Ordinary traffic fluctuations")


def gen_service_anomalies(seed=12346, n=30):
    rng = np.random.RandomState(seed)
    meta, labels = _base_rows(rng, n)
    feats = pd.DataFrame({c: rng.poisson(15, n).astype(float) for c in FEATURES})
    # inject anomalies
    idx = rng.choice(n, size=max(1,n//3), replace=False)
    feats.loc[idx,"http_5xx_count"]=rng.poisson(25,len(idx)).astype(float)
    feats.loc[idx,"http_request_count"]=rng.poisson(30,len(idx)).astype(float)
    feats.loc[idx,"error_count"]=feats.loc[idx,"http_5xx_count"]
    feats.loc[idx,"error_ratio"]=0.8
    feats.loc[idx,"http_5xx_ratio"]=0.8
    feats.loc[idx,"mean_latency_ms"]=rng.normal(450,50,len(idx)).clip(50)
    feats.loc[idx,"p95_latency_ms"]=feats.loc[idx,"mean_latency_ms"]*1.6
    feats.loc[idx,"latency_sample_count"]=feats.loc[idx,"http_request_count"]
    feats.loc[idx,"http_request_eligible_count"]=feats.loc[idx,"http_request_count"]
    feats.loc[idx,"error_eligible_count"]=feats.loc[idx,"event_count"]
    labels["is_anomaly"] = False; labels.loc[idx,"is_anomaly"]=True
    labels["scenario"]="service_anomalies"; labels["scenario_group"]="anomaly"
    return ScenarioCase(feats, labels, meta, "HTTP 5xx spike / latency spike")


def gen_telemetry_faults(seed=12347, n=40):
    rng = np.random.RandomState(seed)
    meta, labels = _base_rows(rng, n)
    feats = pd.DataFrame({c: rng.poisson(15, n).astype(float) for c in FEATURES})
    feats["error_count"]=rng.poisson(1,n).astype(float); feats["error_ratio"]=feats["error_count"]/feats["event_count"].clip(lower=1)
    feats["http_request_count"]=rng.poisson(15,n).astype(float); feats["http_5xx_count"]=rng.poisson(0,n).astype(float)
    feats["http_5xx_ratio"]=0.0
    feats["mean_latency_ms"]=rng.normal(55,8,n).clip(10); feats["p95_latency_ms"]=feats["mean_latency_ms"]*1.3
    feats["latency_sample_count"]=feats["http_request_count"]
    feats["http_request_eligible_count"]=feats["http_request_count"]; feats["error_eligible_count"]=feats["event_count"]
    # inject faults: missing latency (NaN), invalid becomes missing per policy, zero-denom already handled
    idx = rng.choice(n, size=max(2,n//5), replace=False)
    feats.loc[idx[::2], "mean_latency_ms"] = np.nan
    feats.loc[idx[1::2], "latency_sample_count"] = 0  # no samples
    # mark indeterminate/benign fault - should not be treated as attack by label unless specified
    labels["scenario"]="telemetry_faults"; labels["scenario_group"]="fault"
    labels["is_anomaly"]=False
    labels["fault_type"] = "none"
    for i, ii in enumerate(idx):
        labels.at[ii, "fault_type"] = "missing_latency" if i%2==0 else "no_latency_samples"
    return ScenarioCase(feats, labels, meta, "Missing/delayed/invalid telemetry")


def gen_low_volume_boundaries(seed=12348, n=25):
    rng = np.random.RandomState(seed)
    meta, labels = _base_rows(rng, n)
    feats = pd.DataFrame({c: [0.0]*n for c in FEATURES})
    feats.loc[0:4, "event_count"]=1.0; feats.loc[0:4, "http_request_count"]=1.0; feats.loc[0:4, "http_request_eligible_count"]=1.0
    feats.loc[0:4, "latency_sample_count"]=1.0; feats.loc[0:4, "mean_latency_ms"]=50.0; feats.loc[0:4, "p95_latency_ms"]=50.0
    feats.loc[0:4, "error_eligible_count"]=1.0; feats.loc[5:9, "event_count"]=0.0  # empty-like
    feats.loc[10:14, "event_count"]=2.0; feats.loc[10:14, "http_request_count"]=0.0; feats.loc[10:14, "http_request_eligible_count"]=0.0
    feats["error_ratio"]=0.0; feats["http_5xx_ratio"]=0.0
    labels["scenario"]="low_volume_boundaries"; labels["scenario_group"]="boundary"
    labels["is_anomaly"]=False
    return ScenarioCase(feats, labels, meta, "Low volume, zero denominators, boundaries")


def gen_distribution_shifts(seed=12349, n=25):
    rng = np.random.RandomState(seed)
    meta, labels = _base_rows(rng, n)
    feats = pd.DataFrame({c: rng.poisson(40, n).astype(float) for c in FEATURES})  # shifted volume
    feats["error_count"]=rng.poisson(2,n).astype(float); feats["error_ratio"]=feats["error_count"]/feats["event_count"].clip(lower=1)
    feats["http_request_count"]=rng.poisson(40,n).astype(float); feats["http_5xx_count"]=rng.poisson(1,n).astype(float)
    feats["http_5xx_ratio"]=feats["http_5xx_count"]/feats["http_request_count"].clip(lower=1)
    feats["mean_latency_ms"]=rng.normal(95,15,n).clip(20); feats["p95_latency_ms"]=feats["mean_latency_ms"]*1.4
    feats["latency_sample_count"]=feats["http_request_count"]
    feats["http_request_eligible_count"]=feats["http_request_count"]; feats["error_eligible_count"]=feats["event_count"]
    labels["scenario"]="distribution_shifts"; labels["scenario_group"]="shift"
    labels["is_anomaly"]=False  # legitimate shift, not labeled attack
    return ScenarioCase(feats, labels, meta, "Legitimate distribution shift (higher volume/latency)")
