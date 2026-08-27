# Python Coding Style

These conventions apply when Python code is introduced. Defer to committed formatter, linter, and type-checker configuration once it exists.

- Use four spaces, explicit imports, and modern type hints.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Use `pathlib.Path` for filesystem paths.
- Keep command-line scripts thin; reusable logic belongs under `src/mot20/`.
- Put experiment tunables in versioned configuration rather than hardcoded code.
- Represent structured configuration with the project's selected validation approach; do not introduce Pydantic, Hydra, OmegaConf, or another framework before the project chooses it.
- Document public interfaces and non-obvious coordinate, shape, or identity conventions.
- Avoid hidden global state. Make random seeds and device selection explicit where reproducibility depends on them.
