# Python Testing

No Python test configuration exists yet. When introduced, prefer `pytest` unless the implementation establishes another standard.

- Mirror source paths under `tests/` and name files `test_<behavior>.py`.
- Use small synthetic sequences for unit tests; do not require the full MOT20 dataset for ordinary checks.
- Test empty frames, missing detections, occlusion gaps, duplicate detections, identity switches, sequence boundaries, and invalid result rows where relevant.
- Mark GPU, download, dataset, and long-running tests so the default suite stays fast and offline.
- Use fixed seeds for stochastic behavior and compare numerical output with explicit tolerances.
- For bug fixes, reproduce the failure with a test before changing behavior when practical.
- Validate exported tracking results structurally before running expensive metric evaluation.
