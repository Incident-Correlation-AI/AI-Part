# Contract fixture provenance

`incident.schema.json` is a byte-for-byte copy of the canonical LAD contract:

| Field | Value |
|---|---|
| Source | `Incident-Correlation-AI/LAD` → `contracts/schemas/incident.schema.json` |
| SHA-256 | `cfda58928d0b2dd7f5455a16e67744e1c863809375cc12a49fd254085b163f1f` |

It is copied into this repository's test fixtures only so the integration tests
can validate incident drafts against the real downstream schema without
requiring a sibling checkout. If the canonical schema changes, update this copy
and the hash together; do not edit it to make a test pass.
