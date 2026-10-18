"""Pytest configuration and path setup."""
import os
import sys

# Add src directory to path so anomaly_detector package can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
