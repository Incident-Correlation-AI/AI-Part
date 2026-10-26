# Upstream provenance — evidence correlation

This directory contains one file copied from the upstream repository so the
integration does not depend on a developer's checkout or working directory.

| Field | Value |
|---|---|
| Repository | `kalmak889900/automated-log-anomaly-detector` |
| Commit | `b155a05c00741cc5f129610d2d90af0d18ca48ed` |
| Original file | `correlator.py` |
| Local file | `correlator.py` |
| Change | Added the module docstring/header only. Correlation logic is unchanged. |

The upstream repository is **private** and contains no licence file. It was
made accessible to this workstream and the file is vendored only as the
correlation reference implementation; no upstream secrets, configuration or
data are included. Redistribution rights for a public release were requested by
the project owner and should be confirmed with the upstream owner.

Upstream also contains `incident.py` and `incident_pipeline.py`. They are **not**
vendored because their incident shape (`affected_services`, free-form
`evidence`) does not satisfy the canonical LAD contract
(`LAD/contracts/schemas/incident.schema.json`), which requires
`evidence.supporting_event_ids`, `evidence.contradicting_event_ids`,
`hypotheses`, `diagnostic_checks` and `audit`. The local adapter reuses
`correlate_anomalies()` and maps its groups into the canonical incident shape.

Upstream has **no** LLM, prompt template, provider abstraction, redaction layer
or LLM tests. The integration uses the existing local, optional
`anomaly_detector.network.llm` adapter for explanations. This mismatch is
documented in `docs/ml_llm_integration_reconnaissance.md`.
