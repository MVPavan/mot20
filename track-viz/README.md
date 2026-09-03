# MOT20 Viewer

This is a loopback-only, read-only browser for exact MOT20 frames and a
user-selected prediction or ground-truth file. The page takes the server-local
image directory and annotation file explicitly. Derived crop caches, browser
reports, screenshots, and exports stay below
`track-viz/artifacts/`; configured source files are never written.

For a code-oriented architecture map and a short new-session reading order, see
`track-viz/docs/HANDOFF.md`.

## Setup

The committed runtime requires Python 3.12. The final 2026-09-02 acceptance used
Python 3.12.3, Node 24.20.0, and npm 11.19.0. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './track-viz[dev]'
npm --prefix track-viz/web ci
PLAYWRIGHT_BROWSERS_PATH="$PWD/track-viz/web/.playwright" npm --prefix track-viz/web exec playwright install chromium
```

The image path must be a directory of one-based, six-digit JPEG frames such as
`img1/000001.jpg`, with `seqinfo.ini` beside that image directory. The annotation
path may be a 9-column MOT ground-truth file or a 10-column MOT prediction file;
the viewer infers the matching parser. Repository-relative and absolute paths
are accepted.

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

Open `http://127.0.0.1:8000`, then enter the image folder and annotation file as
paths on the machine running the server. Suggestions are read from that server
filesystem, not from the browser machine. Each Browse control starts at the
server filesystem root, shows complete wrapping paths, and supports parent and
child directory navigation. Direct absolute or repository-relative text entry
remains available. Applying a source replaces the active source without
restarting the process.

For a background server with explicit lifecycle commands, use the Track-Viz
service script from the repository root. The port defaults to 8000, the default
configuration is `track-viz/configs/viewer.toml`, and `start` builds the current
frontend before launching the backend:

```bash
./track-viz/scripts/manage_viewer.sh start
./track-viz/scripts/manage_viewer.sh status
./track-viz/scripts/manage_viewer.sh stop

# Select another port.
./track-viz/scripts/manage_viewer.sh start --port 8004
./track-viz/scripts/manage_viewer.sh status --port 8004
./track-viz/scripts/manage_viewer.sh restart --port 8004
./track-viz/scripts/manage_viewer.sh stop --port 8004
```

Runtime PID and log files are ignored under `track-viz/artifacts/service/`.
Stopping is ownership-aware: the script stops a Track-Viz `run_viewer.py`
process on the selected port, but refuses to terminate an unrelated listener.
Use `--config PATH` to select a different viewer configuration.

A scripted launch may preselect the same two paths, and a different loopback
port is explicit:

```bash
.venv/bin/python track-viz/scripts/run_viewer.py \
  --images /path/to/MOT20-01/img1 \
  --annotations /path/to/predictions/MOT20-01.txt \
  --host 127.0.0.1 \
  --port 8010
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
`track-viz/docs/performance.md` for the accepted environment and values.

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

### Focus Review Controls

Focus keeps its review structure mounted while optional-event data refreshes.
Each threshold is an independent local decimal draft: valid values commit after
300 ms of inactivity, or immediately with Enter or blur. Empty, incomplete, or
out-of-range drafts send no request; blur restores the last accepted value and
announces the reason. Checkbox changes are independent and the newest combined
settings win if requests overlap.

Abrupt displacement, scale change, and close interaction are shown as separate
families. Their counts distinguish backend raw matches from the UI's contiguous
activity episodes, and their Previous/Next controls navigate episode anchors.
Low-confidence observations are separate exact-frame controls; they are never
merged into heuristic-family navigation. A disabled control stays visible and
states whether its family is off, lacks matches, or has no destination.

The sequence-wide timeline shows observation runs, explicit missing-frame gaps,
endpoints, activity/low-confidence evidence, and the current-frame playhead.
Clicking the rail seeks its exact rounded one-based frame. With rail focus,
Arrow keys seek one frame, Shift+Arrow seeks ten, and Home/End seek sequence
bounds. The crop filmstrip remains the focused visual evidence.

Trajectory defaults to past-through-current observations. Complete track adds
future evidence with a distinct dashed treatment. Both modes break at missing
observations: no line is inferred through a gap. Image, observation, Context,
and event-refresh statuses occupy overlays or reserved status space, so they do
not move the viewport, timeline, filmstrip, or lower controls.

The image bitmap cache accepts capacities 100 through 200 and defaults to 150.
Forward playback prefetches stable 12-frame batches through three workers,
deduplicates foreground and prefetch requests, and advances only after the
current bitmap is decoded. Slow frames therefore pause the playback clock
instead of being skipped. The cache closes stale or evicted bitmaps and
validates immutable frame ETags. Crop caches are write-once derived files below
`track-viz/artifacts/cache/`.

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
- Raw event density is not an activity-navigation verdict. In particular, a
  zero displacement/scale threshold can match every valid transition and a zero
  proximity threshold can match touching boxes; review raw-match counts and
  episode anchors separately. Real track-72 observations are read-only and are
  recorded as observations, never deterministic fixture expectations.
- Browser exports have no dedicated UI control. The bounded POST API and offline
  CLI are documented in `track-viz/docs/exports.md`.
- The locally exercised export codec is MP4V in an MP4 container. No broader
  browser or platform codec compatibility is claimed.

Source-integrity manifests and performance reports live under
`track-viz/artifacts/verification/`; browser screenshots and traces live under
`track-viz/web/test-results/` and `track-viz/web/playwright-report/`. These are
ignored derived records and may contain local paths or machine details.
