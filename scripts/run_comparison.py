#!/usr/bin/env python3
"""Run Milestone 4 model comparison.

Pre-declares criteria; uses validation for selection/thresholding; test evaluated once.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.comparison_config import load_comparison_config
from anomaly_detector.comparison import run_comparison


def main():
    p = argparse.ArgumentParser(description="M4 model comparison")
    p.add_argument("--config", default="configs/comparison.yaml")
    p.add_argument("--out-root", default=None)
    args = p.parse_args()
    cfg = load_comparison_config(args.config)
    rep = run_comparison(cfg, args.out_root)
    print("M4 comparison complete")
    print(f"Recommended: {rep.get('recommended')}")
    print(f"Report: {rep['run_dir']}/reports/comparison_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
