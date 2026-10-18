#!/usr/bin/env python3
"""Generate synthetic data for testing and development."""
import argparse
import json
import os
from datetime import datetime, timezone

from anomaly_detector.synthetic_data import SyntheticDataGenerator
from anomaly_detector.validation import validate_and_normalize_events
from anomaly_detector.preprocessing import aggregate_events_to_windows
from anomaly_detector.features import windows_to_feature_dataframe, windows_to_metadata_dataframe, labels_to_dataframe
from anomaly_detector.reporting import generate_data_quality_report

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ML anomaly detector data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="./data/synthetic", help="Output directory")
    parser.add_argument("--window-size", type=int, default=1, help="Window size in minutes")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./data/processed", exist_ok=True)
    os.makedirs("./artifacts", exist_ok=True)
    
    print(f"Generating synthetic data with seed={args.seed}")
    gen = SyntheticDataGenerator(seed=args.seed)
    data = gen.generate_full_synthetic_dataset()
    
    # Save raw events
    events_dict = [e.to_dict() for e in data["events"]]
    with open(os.path.join(args.output_dir, "synthetic_events.json"), "w") as f:
        json.dump(events_dict, f, indent=2)
    
    # Save labels
    labels_dict = [l.to_dict() for l in data["labels"]]
    with open(os.path.join(args.output_dir, "synthetic_labels.json"), "w") as f:
        json.dump(labels_dict, f, indent=2)
    
    # Validate
    val_result = validate_and_normalize_events(events_dict)
    accepted_events = val_result.accepted
    
    # Aggregate to windows
    windows = aggregate_events_to_windows(accepted_events, window_size_minutes=args.window_size)
    
    # Generate features and metadata
    feat_df = windows_to_feature_dataframe(windows)
    meta_df = windows_to_metadata_dataframe(windows)
    lbl_df = labels_to_dataframe(data["labels"])
    
    # Save
    feat_df.to_csv(os.path.join("./data/processed", "features.csv"), index=False)
    meta_df.to_csv(os.path.join("./data/processed", "metadata.csv"), index=False)
    lbl_df.to_csv(os.path.join("./data/processed", "labels.csv"), index=False)
    
    # Generate report
    report = generate_data_quality_report(val_result, windows, data["labels"])
    with open(os.path.join("./artifacts", "data_quality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Generated {len(data['events'])} events")
    print(f"✓ Created {len(windows)} feature windows")
    print(f"✓ Features saved to data/processed/features.csv")
    print(f"✓ Report saved to artifacts/data_quality_report.json")
    print(f"✓ Scenarios: {', '.join(data['scenarios'])}")

if __name__ == "__main__":
    main()
