
"""Vendored upstream evidence-correlation implementation.

Copied **verbatim** from
``kalmak889900/automated-log-anomaly-detector@b155a05c00741cc5f129610d2d90af0d18ca48ed``
(``correlator.py``).  Only this provenance header was added; the correlation
logic below is unchanged.  See ``PROVENANCE.md`` in this directory.

Upstream has no LLM.  This module is the actual evidence-correlation code that
the integration reuses; it produces candidate groups, not confirmed root
causes.
"""
from datetime import datetime, timezone


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp, including a UTC Z suffix."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")

    return parsed.astimezone(timezone.utc)


def correlate_anomalies(
    anomalies: list[dict],
    window_seconds: int = 120,
) -> list[list[dict]]:
    """
    Group anomalies by service and time proximity.

    This creates candidate groups, not confirmed root causes.
    """
    if window_seconds < 0:
        raise ValueError("window_seconds cannot be negative")

    groups = []

    # Ignore events without the required correlation fields.
    valid = []
    for anomaly in anomalies:
        if not anomaly.get("service") or not anomaly.get("timestamp"):
            continue

        valid.append((
            parse_timestamp(anomaly["timestamp"]),
            anomaly,
        ))

    valid.sort(key=lambda item: item[0])

    for timestamp, anomaly in valid:
        matching_group = None

        for group in groups:
            last_timestamp, last_anomaly = group[-1]

            same_service = (
                anomaly["service"] == last_anomaly["service"]
            )
            close_in_time = (
                (timestamp - last_timestamp).total_seconds()
                <= window_seconds
            )

            if same_service and close_in_time:
                matching_group = group
                break

        if matching_group is None:
            groups.append([(timestamp, anomaly)])
        else:
            matching_group.append((timestamp, anomaly))

    return [
        [anomaly for _, anomaly in group]
        for group in groups
    ]
