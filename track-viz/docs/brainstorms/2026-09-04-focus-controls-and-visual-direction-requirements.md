# Focus Controls and Visual Direction Requirements

Date: 2026-09-04

Status: User feedback captured; implementation and detailed event semantics remain pending approval

Reference: `https://www.beautifului.dev/`

## Accepted Product Feedback

1. Replace the separate Focus/Context mode control and Context-track count with
   one numeric control named **Number of nearby tracks**.
   - Default: `0`.
   - `0` means focus-only rendering.
   - Any positive value enables nearby-track context and requests that many
     competitors, up to the existing safe hard cap.
2. Abrupt displacement, scale change, and close interaction are not yet trusted
   as correct user-facing review tools. Do not treat the present automated tests
   as proof that their product semantics are right. Characterize the algorithms,
   thresholds, grouping, anchors, navigation, disabled states, and representative
   real-track behavior before proposing corrections.
3. Replace the large trajectory fieldset with one compact control named
   **Show future trajectory**. Past-through-current remains the default; enabling
   the control adds visually distinct future evidence.
4. Remove the visible banner saying that a source is local test-adapted
   development material. This is a UI decision only: benchmark-separation and
   provenance requirements remain authoritative in maintained experiment and
   export records.

## Current Visual Problem

The current viewer uses a ruled-paper background, strong horizontal rules,
square controls, serif headings, and many controls placed directly on broad
bands. The result feels like a rule book with overlays instead of a focused
visual inspection workspace. The immediate priority remains usability, but a
contained visual refresh is acceptable if it does not destabilize interaction,
frame identity, accessibility, performance, or the canvas.

## Beautiful UI Review

The reference site is a gallery of compact AI-interface primitives rather than
a complete tracker-review application. Useful qualities include:

- a restrained neutral surface hierarchy instead of page-wide decorative lines
- hairline borders, shallow elevation, moderate corner radii, and deliberate
  interior spacing
- compact segmented controls, chips, status indicators, and secondary actions
- one strong accent reserved for active or important state
- clear typography hierarchy with small labels and readable content text
- responsive cards that retain their hierarchy at narrow widths
- loading, table, filter, selection-action, and diff patterns that could later
  support annotation workflows

The chat, prompt, thinking, and agent-specific primitives are not relevant to the
current tracker viewer and should not be imported merely for visual consistency.

The site offers copy/view-code interactions and an MIT license. Its deployed
implementation appears to use Next.js and Tailwind-style utility classes, while
Track-Viz uses React/Vite and a single plain-CSS design layer. Direct component
copying would therefore require adaptation and would introduce unnecessary
framework carrying cost. Reusing its visual principles and reimplementing only
the few applicable patterns with the existing stack is the preferred approach.

## Feasibility Assessment

### Recommended: contained visual refresh

A pure visual skin is a small-to-medium change if the existing React structure,
accessible names, control grouping, and behavioral classes are preserved. Most
of the current appearance is centralized in `track-viz/web/src/styles.css`, so
the following can change without rewriting the application:

- remove the ruled-paper page background
- introduce a compact neutral token palette for page, panel, inset, border,
  primary text, secondary text, accent, warning, and focus states
- restyle existing sections as quiet panels instead of rule-separated bands
- simplify typography and reduce uppercase labels
- use compact rounded controls, consistent heights, restrained shadows, and
  clearer active/disabled states
- visually prioritize the frame viewport, timeline, and focused-track evidence
  over source metadata and optional diagnostics

This CSS-first approach can preserve the current DOM and most browser tests while
making the viewer substantially calmer and more modern. The accepted compact
control changes are separate small behavior/markup edits.

Changing information hierarchy, regrouping controls into new cards, or altering
desktop/narrow responsive order is a medium change rather than a style-only
change. Those decisions affect markup, accessibility relationships, geometry
assertions, screenshots, and browser selectors and must be planned and verified
accordingly.

### Not recommended now: structural redesign

A new sidebar/dashboard layout, movable inspectors, resizable panes, or a full
component-framework migration would be a medium-to-large project. It would touch
responsive geometry, keyboard order, layout-stability assertions, screenshots,
and annotation architecture. That work should wait until the annotation workflow
and panel requirements are understood.

## Recommended Direction

Proceed in this order:

1. Implement the three compact control changes and remove the visible policy
   banner while retaining policy metadata and documentation.
2. Run a dedicated event-semantics investigation before changing the three event
   detectors.
3. Apply a style-only Beautiful-UI-inspired visual refresh that preserves the
   current application structure.
4. Reconsider structural layout only when annotation actions, queues, and review
   decisions have concrete requirements.

## Non-Goals for the First Visual Pass

- no Next.js or Tailwind migration
- no wholesale copying of unrelated AI-agent components
- no canvas, tracking, frame, or source-identity behavior changes
- no new sidebar or multipane annotation workspace
- no claim that a visual refresh validates event semantics

## Open Decision

Before a style implementation plan is approved, choose whether the primary
inspection workspace should be dark by default, light with dark panels, or
support both modes. The Beautiful UI reference supports both, but its dark mode
is the stronger fit for image review and overlay contrast.
