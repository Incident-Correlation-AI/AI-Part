# Draft integration comment for Abel

**Status:** draft only — **not posted**. The addressee and target are not
unambiguous in the available context, so this file is the holding place until
they are confirmed.

Before posting, replace every placeholder below and confirm all three:

- [x] Addressee identity: `bella-247` (confirmed by the repository owner).
- [ ] Target: `[TARGET_ISSUE_OR_PR_URL]` is the correct thread.
- [ ] The comment does not disclose secrets, private data or synthetic metrics
      as real-world results.

---

## Comment body (draft)

Hi `bella-247` — the network/service-log extension is ready for an
integration review. It is **additive**: the existing `features-v1` path and its
artifacts are unchanged, and the contract is now `1.1.0`.

### What you can call

```python
from anomaly_detector.network import NetworkWindow, load_network_detector

detector = load_network_detector()      # artifacts/network/model/*
result = detector.score_window(window)  # same InferenceResult shape
```

- Schema: `network-features-v1` (39 columns). The allowlist is
  `detector.expected_feature_columns`.
- Statuses are unchanged: `scored`, `insufficient_data`, `invalid_input`,
  `model_unavailable`, `inference_error`.
- Three additive reason codes matter for network windows: `no_events`,
  `insufficient_coverage`, `telemetry_quality`.
- `is_anomaly` is the Isolation Forest decision
  (`raw_score <= threshold`). The transparent rule baseline only contributes
  `extra["network"]["baseline"]` evidence and `severity`; it does not rewrite
  the flag.

### What I need you to handle

1. **Never turn unknown into normal.** Empty/sparse/degraded windows return
   `insufficient_data` with `is_anomaly = None`. They must not become
   "no anomaly" events downstream.
2. **Use the existing forwarding rule.** `route_result()` / the integration
   harness already refuses non-`scored` results. Please do not re-implement the
   `if is_anomaly` check.
3. **Keep evidence separate from diagnosis.** Rule hits, `severity` and
   `telemetry_quality` are leads for correlation. The ML component does not
   select evidence event IDs, group incidents, propose hypotheses or execute
   checks.
4. **Do not explain an unscored window as a detection.** The optional adapter
   in `anomaly_detector.network.llm` is disabled by default, is grounded only in
   a bounded evidence bundle, and never changes a decision.
5. **Preserve unknown enum members.** If you serialise/deserialise
   `ReasonCode` or `EventType`, keep members you do not recognise instead of
   failing.

### Questions for you

1. Is the whole `InferenceResult` acceptable, or do you need a narrower DTO?
2. Is the incident schema's `evidence.supporting_event_ids` /
   `contradicting_event_ids` population purely your responsibility? The ML side
   can supply window IDs and rule evidence, but not event-ID selection.
3. Should network windows and existing `features-v1` windows share one sink, or
   stay on separate routes?
4. What sync/async and retry behaviour should the network path use? The current
   boundary is in-process and in-memory only.

### Honest scope note

All measured figures come from a deterministic local simulator that repeats one
90-minute scenario schedule. They validate the pipeline, not production
generalisation. The benign high-traffic scenario has a 0.25 false-positive rate
in the time test split, and the rule baseline beats the forest on the clearest
auth-burst pattern. Please read
`docs/ml_network_evaluation.md` §7 before quoting any number.

Full details: `docs/ml_network_integration.md`,
`docs/ml_network_features.md`, `docs/ml_network_evaluation.md`.

---

## Posting checklist (do not skip)

- [ ] Replace `[TARGET_ISSUE_OR_PR_URL]` with the correct thread URL.
- [ ] Confirm the repository and thread are the intended destination.
- [ ] Confirm no secrets or private data appear in the comment.
- [ ] Post only after explicit confirmation from the repository owner.
