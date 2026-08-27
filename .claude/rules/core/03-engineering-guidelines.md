# Engineering Guidelines

## Think Before Editing

- Distinguish verified facts, assumptions, and open decisions.
- Read the relevant implementation, caller, configuration, test, and data contract.
- If a choice changes public behavior, experiment validity, or data safety, surface it before acting.

## Keep Changes Small

- Implement only the requested behavior.
- Match established patterns before introducing abstractions.
- Avoid adjacent refactors and speculative configuration.
- Remove only artifacts made obsolete by the current change.

## Protect Evidence and Data

- Treat datasets, weights, embeddings, predictions, and result directories as valuable records.
- Never invent metrics or claim unrun verification.
- Preserve identities, frame indices, coordinate conventions, and sequence boundaries.

## Verify Outcomes

- Define the smallest check that proves the claim.
- Run it fresh and read its exit status and output.
- Report failures, skipped checks, and environmental limitations explicitly.
