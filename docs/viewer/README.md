# MOT20 Viewer

This is a loopback-only, read-only browser for exact MOT20 frames and configured
observations. `configs/viewer.toml` is the source catalog authority. Derived crop
caches, browser reports, screenshots, and exports stay below
`artifacts/viewer/`; configured source files are never written.

## Setup

The committed runtime requires Python 3.12. The final 2026-09-02 acceptance used
Python 3.12.3, Node 24.20.0, and npm 11.19.0. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
npm --prefix web ci
PLAYWRIGHT_BROWSERS_PATH="$PWD/web/.playwright" npm --prefix web exec playwright install chromium
```

The local dataset must provide every path named in `configs/viewer.toml`.
Each `[[sources]]` entry contains a stable key, sequence name, repo-relative
`seqinfo`, image directory, annotation file, and adapter. Optional provenance
records producer, detector, checkpoint, tracker, post-processing, adaptation
iterations, and notes.

`mot_gt_9` reads MOT ground-truth rows as
`frame,id,x,y,width,height,mark,class,visibility`. `mot_result_10` preserves all
ten result fields as `frame,id,x,y,width,height,score,opaque,opaque,opaque`.
Frames are one-based. Browser geometry is display-clamped `xyxy`; raw `xywh`,
source-row index, row hash, source hash, and identity fields remain available for
audit.

## Run And Verify

Build the production frontend and start the loopback application:

```bash
make build
make run
```

Open `http://127.0.0.1:8000`. A different loopback port is explicit:

```bash
.venv/bin/python scripts/run_viewer.py --host 127.0.0.1 --port 8010
```

Stable aggregate gates and focused commands are:

```bash
make test          # backend, frontend, and CVAT unit tests
make lint          # Ruff, mypy, and strict TypeScript
make build         # production frontend
make e2e           # deterministic desktop and narrow browser acceptance
make smoke-local   # configured local-source contracts
make pip-check
make acceptance    # all offline gates above
make e2e-real      # production server and real local-data journeys/measurements
```

`make e2e-real` writes ignored screenshots, traces, and the measured report; it
requires configured MOT20 files and takes several minutes. See
`docs/viewer/performance.md` for the accepted environment and values.

## Controls And Capabilities

- Choose a source, enter an exact one-based frame, use the scrubber, or move by
  `-10`, `-1`, play/pause, `+1`, and `+10`.
- With canvas focus, Left/Right moves one frame, Shift+Left/Right moves ten,
  Space toggles playback, `B` temporarily reveals all current-frame boxes, and
  Escape clears selection or exits Focus.
- Hover ranks containing observations. Wheel cycles ambiguous candidates.
  Click pins the current candidates; keyboard or pointer selection confirms one.
- Tracked sources allow exact ID search, Focus, previous/next observation and
  gap navigation, timeline markers, deterministic filmstrip samples, and a
  restrained Context overlay of at most eight competitors.
- Sentinel-only or unusable IDs disable Follow and track tools. The UI reports
  missing provenance, stale source hashes, unavailable files, and capability
  reasons rather than synthesizing identity.

The image bitmap cache accepts capacities 100 through 200 and defaults to 150.
It prefetches four frames in the current direction, aborts superseded requests,
closes stale/evicted bitmaps, and validates immutable frame ETags. Crop caches
are write-once derived files below `artifacts/viewer/cache/`.

## Policy And Limitations

The language in `docs/MOTPolicy.md` is mandatory: MOT20 test-derived review and
adaptation are local development, not clean held-out benchmark evaluation and
not directly comparable to official leaderboard results. Preserve detector,
tracker, checkpoint, post-processing, review, and adaptation provenance.

- The configured MOT20-06 and MOT20-08 JOCO files currently contain only
  sentinel ID `-1`. Corrected tracked exports are absent, so real Stage 1
  tracked-result behavior is not validated on 06/08. Their accepted real
  evidence is detection-only ambiguity; tracked 06/08 evidence is synthetic
  fixture coverage only.
- Meaningful low-confidence navigation is covered by synthetic varying-score
  fixtures only. MOT20-01 is ground truth with no tracker-score confidence
  signal.
- The accepted real MOT20-01 Focus/Context journey uses track 72, frames
  404-429, which has no internal gaps. Gap journeys are therefore synthetic
  fixture evidence, not a real MOT20-01 gap claim.
- Browser exports have no dedicated UI control. The bounded POST API and offline
  CLI are documented in `docs/viewer/exports.md`.
- The locally exercised export codec is MP4V in an MP4 container. No broader
  browser or platform codec compatibility is claimed.

Source-integrity manifests and performance reports live under
`artifacts/viewer/verification/`; browser screenshots and traces live under
`web/test-results/` and `web/playwright-report/`. These are ignored derived
records and may contain local paths or machine details.