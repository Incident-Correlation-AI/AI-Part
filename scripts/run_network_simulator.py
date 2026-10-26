#!/usr/bin/env python3
"""Generate the deterministic, local-only network incident dataset.

The simulator emits canonical LAD event envelopes to ``events.jsonl`` and
writes evaluation labels to a **separate** ``labels.csv`` channel.  It opens no
sockets, contacts no host and performs no authentication; "auth" records are
synthetic log lines.

Examples
--------
    python scripts/run_network_simulator.py
    python scripts/run_network_simulator.py --config configs/network.yaml --runs 10 --seed 42
    python scripts/run_network_simulator.py --output-dir /tmp/network-sim
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.network.config import load_network_config
from anomaly_detector.network.simulator import SimulatorConfig, simulate


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local synthetic network incidents")
    parser.add_argument("--config", default="configs/network.yaml", help="Path to YAML config")
    parser.add_argument("--runs", type=int, default=None, help="Number of simulated runs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    parser.add_argument("--max-events", type=int, default=None, help="Hard event bound")
    args = parser.parse_args()

    config = load_network_config(args.config)
    simulator = SimulatorConfig(
        seed=args.seed if args.seed is not None else config.simulator.seed,
        run_count=args.runs if args.runs is not None else config.simulator.run_count,
        run_minutes=config.simulator.run_minutes,
        window_size_minutes=config.simulator.window_size_minutes,
        base_http_per_minute=config.simulator.base_http_per_minute,
        base_auth_per_minute=config.simulator.base_auth_per_minute,
        base_db_per_minute=config.simulator.base_db_per_minute,
        inject_malformed=config.simulator.inject_malformed,
        max_events=args.max_events if args.max_events is not None else config.simulator.max_events,
        start_time=config.simulator.start_time,
    )
    dataset = simulate(simulator)
    output_dir = args.output_dir or config.paths["synthetic_dir"]
    paths = dataset.write(output_dir)
    print(json.dumps({"summary": dataset.summary, "paths": paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
