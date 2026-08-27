# Python and ML Safety

- Never log or commit tokens, private dataset locations, or credentials.
- Validate paths and configuration before starting expensive GPU jobs.
- Do not delete or overwrite datasets, checkpoints, embeddings, predictions, or evaluation results without explicit approval.
- Load pickled or PyTorch checkpoint formats only from trusted sources.
- Prefer argument-list subprocess calls; do not interpolate untrusted paths into shell strings.
- Make device, dtype, and tensor-shape assumptions explicit at module boundaries.
- Avoid silent exception swallowing in batch pipelines. Record failed sequences and report partial completion.
- Network downloads, external compute, and long-running training require clear scope and cost awareness.
