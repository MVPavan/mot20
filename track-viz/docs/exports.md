# Viewer Track Exports

Track exports are read-only derived artifacts. They never write beneath a configured image or annotation source. Final videos and metadata are published under `track-viz/artifacts/exports/<export-id>/`, where the export ID hashes the source/result hashes, inclusive frame range, rendering parameters, tool versions, and context selection.

## Focused API Export

`POST /api/sequences/{source_key}/exports` performs a synchronous export of at most 300 inclusive frames. The JSON body requires the current `source_hash`, a usable positive `track_id`, `start_frame`, and `end_frame`. Optional `context_count` and `trace_length` values are bounded.

The request must carry an `Origin` equal to the server's configured application origin. `track-viz/scripts/run_viewer.py` derives that origin from the bind host and port; `--app-origin` overrides it when a proxy changes the browser-visible origin. Development mode accepts the configured Vite origin for POST CORS.

Requests fail with typed errors for a missing or stale source hash, an annotation result changed after startup, an invalid Origin, unavailable track capability, an invalid range, or a range above 300 frames. Larger work is directed to the offline command.

## Offline Per-Track Export

Use the fixed configured source key and an explicit safety bound:

```bash
.venv/bin/python track-viz/scripts/export_track_video.py mot20-01-gt 72 \
  --source-hash 89fd0196d67a5eb6011a470dc2a49b02255403b49e8848031cdf99add8a36d9c \
  --start-frame 404 --end-frame 429 --max-frames 40
```

Omitted frame bounds use the selected track's first and last observations. `--max-frames` may exceed the interactive 300-frame cap, defaults to 3,000, and cannot exceed the hard 100,000-frame safety ceiling.

## Rendering And Metadata

Exports use display-clamped `xyxy` geometry, matching browser hit regions and focus overlays. Supervision 0.27 renders focal `BoxAnnotator` boxes, smart-positioned selected `LabelAnnotator` labels, context `BoxCornerAnnotator` corners, and bounded `TraceAnnotator` traces with `ColorLookup.TRACK`. Colors use the shared `fnv1a32-hsv-integer-v1` sequence/track contract.

Each `metadata.json` records:

- annotation/result SHA-256 and every included image SHA-256
- per-frame source observation row hashes
- sequence, inclusive frames, track, context IDs, and render parameters
- dimensions, frame rate, codec, output SHA-256, and tool/library versions
- incoming producer, detector, checkpoint, tracker, post-processing, adaptation, and notes provenance
- MOT policy classification, including local test-adapted development material

Publication uses a temporary sibling directory followed by atomic rename. An identical valid export returns the existing artifact. An incomplete or different collision is rejected and never overwritten; a failed render removes its temporary output.

## Task 7 Evidence

On 2026-09-02, source `mot20-01-gt` at result hash `89fd0196d67a5eb6011a470dc2a49b02255403b49e8848031cdf99add8a36d9c` was exercised for continuous track 72:

- focused clip, frames 404-413: `track-viz/artifacts/exports/22efb554a3a175369ac9e366793e62f4a6670d0104a2037636157bf367513bec/`
- offline track video, frames 404-429: `track-viz/artifacts/exports/cb52b1df27017703b0682fe37dc85b26583417b40e478676b5b5af8db88bcdf1/`
- decoded first/middle/last readback: `track-viz/artifacts/exports/verification/task7-final-contract-readback.jpg`

Full decode observed 10 and 26 frames respectively at 1920x1080, 25 fps. Video hashes matched sidecars and focal/context overlays remained readable at first, middle, and last frames. The exercised local codec is MP4V in an MP4 container; no broader codec support is claimed.

The offline per-track video can replace the continuous lifespan-overview portion of review for a short track. The focused clip is useful for sharing a bounded interval. Neither replaces browser ambiguity selection, exact source-row inspection, or navigation around long gaps and optional events.