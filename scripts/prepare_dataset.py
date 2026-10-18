#!/usr/bin/env python3
"""Prepare public datasets (adapter path)."""
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Prepare public dataset using adapter")
    parser.add_argument("--dataset", default="none", help="Dataset name")
    parser.add_argument("--path", help="Path to dataset")
    args = parser.parse_args()
    
    # Document adapter path as per requirements
    info = {
        "dataset_name": args.dataset,
        "status": "no_adapter_implemented",
        "blocker": "No suitable public dataset automatically downloaded/verified in this environment",
        "adapter_interface": "src/anomaly_detector/dataset_adapters/base.py",
        "documentation": "See README.md for instructions on adding dataset adapters",
        "requirements": [
            "Verify license and usage restrictions",
            "Confirm field mapping to common event format",
            "Document label semantics",
            "Add tests with fixture data",
            "Do not commit dataset files"
        ]
    }
    
    os.makedirs("./artifacts", exist_ok=True)
    with open(os.path.join("./artifacts", "dataset_adapter_status.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print("Dataset adapter path documented.")
    print("See artifacts/dataset_adapter_status.json for details.")

if __name__ == "__main__":
    main()
