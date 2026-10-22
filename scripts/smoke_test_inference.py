#!/usr/bin/env python3
"""Milestone 6 smoke test for the ML inference interface.

Demonstrates the interface end to end without a web server:

1. load the selected model once,
2. score a feature window taken from the existing feature pipeline output,
3. print the structured result,
4. show how a downstream consumer branches on the explicit status.

Usage
-----
    python scripts/smoke_test_inference.py
    python scripts/smoke_test_inference.py --split test
    python scripts/smoke_test_inference.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd

from anomaly_detector.inference import (
    CONTRACT_VERSION,
    InferenceStatus,
    load_detector,
    make_window_id,
)
from anomaly_detector.inference.contract import FeatureWindowRequest


def _load_window(splits_dir: str, strategy: str, split: str, index: int):
    """Read one already-aggregated feature window from the M1/M2 split output."""
    base = os.path.join(splits_dir, strategy, split)
    features = pd.read_csv(os.path.join(base, "features.csv"))
    metadata = pd.read_csv(os.path.join(base, "metadata.csv"))
    row = features.iloc[index]
    meta = metadata.iloc[index]
    request = FeatureWindowRequest(
        window_id=str(meta.get("sample_key") or make_window_id(
            meta.get("host_id"), meta.get("service"), meta.get("window_start")
        )),
        window_start=datetime.fromisoformat(str(meta["window_start"])),
        window_end=datetime.fromisoformat(str(meta["window_end"])),
        features={c: row[c] for c in features.columns},
        host_id=str(meta.get("host_id", "")) or None,
        service_name=str(meta.get("service", "")) or None,
    )
    return request


def main() -> int:
    parser = argparse.ArgumentParser(description="ML inference interface smoke test")
    parser.add_argument("--model", default="artifacts/baseline/model/model.joblib")
    parser.add_argument("--manifest", default="artifacts/baseline/model/manifest.json")
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--strategy", default="random")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="score every row of the split")
    args = parser.parse_args()

    print("=" * 72)
    print("ML inference interface smoke test")
    print("=" * 72)

    detector = load_detector(args.model, args.manifest)
    contract = detector.contract()
    print(f"contract_version    : {CONTRACT_VERSION}")
    print(f"model               : {contract['model_name']} (version {contract['model_version']})")
    print(f"feature schema      : {contract['feature_schema_version']} "
          f"({len(contract['feature_columns'])} features)")
    print(f"score direction     : {contract['score_direction']} (not a probability)")
    print(f"threshold           : {contract['threshold']}  [{contract['threshold_rule']}]")
    print(f"threshold config    : {contract['threshold_config_version']}")
    print()

    if args.all:
        base = os.path.join(args.splits_dir, args.strategy, args.split)
        features = pd.read_csv(os.path.join(base, "features.csv"))
        metadata = pd.read_csv(os.path.join(base, "metadata.csv"))
        requests = []
        for i in range(len(features)):
            row = features.iloc[i]
            meta = metadata.iloc[i]
            requests.append(
                FeatureWindowRequest(
                    window_id=str(meta["sample_key"]),
                    window_start=datetime.fromisoformat(str(meta["window_start"])),
                    window_end=datetime.fromisoformat(str(meta["window_end"])),
                    features={c: row[c] for c in features.columns},
                    host_id=str(meta.get("host_id", "")) or None,
                    service_name=str(meta.get("service", "")) or None,
                )
            )
        results = detector.score_windows(requests)
        scored = [r for r in results if r.status is InferenceStatus.SCORED]
        flagged = [r for r in scored if r.is_anomaly]
        print(f"batch of {len(results)} window(s):")
        print(f"  scored             : {len(scored)}")
        print(f"  flagged as anomaly : {len(flagged)}")
        print(f"  not scored         : {len(results) - len(scored)}")
        print()
        print("first result:")
        print(json.dumps(results[0].to_dict(), indent=2))
        return 0

    request = _load_window(args.splits_dir, args.strategy, args.split, args.index)
    result = detector.score_window(request)
    print(json.dumps(result.to_dict(), indent=2))

    print()
    print("downstream branching example:")
    if result.status is InferenceStatus.SCORED:
        print("  -> scored: forward to the evidence-correlation boundary")
        print(f"     window_id={result.window_id} score={result.raw_score:.6f} "
              f"is_anomaly={result.is_anomaly}")
    else:
        print(f"  -> {result.status.value}: do NOT forward as normal "
              f"(reason_code={result.reason_code.value})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
