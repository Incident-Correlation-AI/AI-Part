"""Vendored upstream correlation implementation (see ``PROVENANCE.md``)."""
from .correlator import correlate_anomalies, parse_timestamp

#: Upstream commit the vendored file was taken from.
UPSTREAM_REVISION = "b155a05c00741cc5f129610d2d90af0d18ca48ed"
UPSTREAM_MODULE = "correlator.py"

__all__ = ["UPSTREAM_MODULE", "UPSTREAM_REVISION", "correlate_anomalies", "parse_timestamp"]
