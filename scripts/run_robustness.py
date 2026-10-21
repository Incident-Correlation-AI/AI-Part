#!/usr/bin/env python3
"""Run Milestone 5 robustness validation.

Uses the SELECTED M3 model without retraining or tuning.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from anomaly_detector.robustness import run_robustness


def main():
    p = argparse.ArgumentParser(description="M5 robustness validation")
    p.add_argument("--out-root", default=None)
    args = p.parse_args()
    rep = run_robustness(args.out_root)
    print("M5 robustness validation complete")
    print(f"Run dir: {rep['run_dir']}")
    print(f"Report: {rep['run_dir']}/reports/robustness_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
