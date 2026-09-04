# Viewer Release Evidence

Latest pointer/cache measurement: 2026-09-04 by `make e2e-real`. The
machine-readable authority is
`track-viz/artifacts/verification/browser-performance.json`; source manifests and
their comparison are in the same ignored directory.

## Environment

| Item | Observed value |
| --- | --- |
| CPU | AMD Ryzen Threadripper 7960X 24-Cores, 48 logical CPUs |
| RAM | 134,541,451,264 bytes |
| OS | Linux 6.8.0-137-generic x64 |
| Browser | Chromium 151.0.7922.34 |
| Display / viewport | 1440 x 1000 |
| Device-pixel ratio | 1 |
| Browser user agent | Playwright Desktop Chrome profile, Windows 10 token |

The OS row describes the host. The user-agent row records Playwright's browser
profile and must not be interpreted as the host OS.

## Pointer Latency

The densest configured MOT20-06/08 frame is `mot20-06-joco` frame 522 with 161
observations. Each sample starts at the native `pointermove` `event.timeStamp`
before coordinate conversion and hit/rank work, passes through selection
hysteresis/state update, and ends after the dependent overlay
`requestAnimationFrame` finishes drawing. The point `(1426.23, 378.825)`
intersects five observations. After 120 warm-up samples:

| Run | Samples | p50 | p95 | Required p95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 16.6 ms | 16.8 ms | < 50 ms |
| 2 | 1,000 | 16.6 ms | 16.8 ms | < 50 ms |
| 3 | 1,000 | 16.6 ms | 16.8 ms | < 50 ms |

Every run passed. No profiling or product optimization loop was required.

## Seek And Cache

Thirty cold frames were spaced five frames apart, beyond the four-frame
directional prefetch window. The same decoded frames were then visited in
reverse for warm-cache observations. The origin defines no pass threshold for
seek latency.

| Measurement | p50 | p95 |
| --- | ---: | ---: |
| Cold seek to completed image draw | 26.0 ms | 35.6 ms |
| Warm seek to completed image draw | 14.1 ms | 14.4 ms |

Compared with the 2026-09-03 accepted evidence, pointer p50/p95 is unchanged,
cold-seek p50 is 1.6 ms higher while p95 is 3.0 ms lower, and warm-seek p50 is
unchanged while p95 is 0.3 ms lower. The visual/control foundation therefore
stays within the prior interaction envelope; seek values remain observations,
not pass thresholds.

The 500-frame real scrub observed:

- configured cache bounds 100 through 200, default and active capacity 150
- bitmap-count ceiling 150
- explicit bitmap-release counter ceiling 538
- observed Chromium `usedJSHeapSize` ceiling 10,000,000 bytes
- strong frame ETag
  `"14dbf2e3bfe33afe6d48980dcf4f69c445b896c566308e4f6553b8b978643152"`
  and conditional response status 304

Focused tests additionally prove least-recently-used eviction, close-on-clear,
close after stale decode, directional prefetch, and superseded observation
request cancellation. Heap and seek values are observed ceilings/budgets, not
new pass thresholds.

## Browser And Accessibility

Deterministic Playwright acceptance passed 45 desktop/narrow tests with 13
intentional real-data/project skips. The axe
scan ran on Explore, pinned chooser, and Focus in both viewports: six scans,
zero serious violations, and zero critical violations. No exceptions are
recorded.

Real production-server acceptance passed seven tests with one intentional narrow
performance skip: MOT20-06 and MOT20-08 seek/hover/cycle/pin/confirm with Follow
disabled, and MOT20-01 track 72 search/Focus/timeline/filmstrip/Context/export
on desktop and narrow. Checks include nonblank canvas pixels, overlay hit
alignment, screenshots, horizontal/text/control bounds, and no accepted
console, failed-request, or HTTP-error events.

The focused export was hash-verified and decoded as 10 frames, 1920 x 1080,
25 fps, MP4V. MP4V is the only locally claimed codec.

The 2026-09-03 read-only track-72 Focus observation at frame 404 and
displacement threshold 0.02 found 17 raw matches. Its next raw displacement
frame and next grouped activity anchor were both 409. This is observational
local-data evidence, not a deterministic fixture assertion; the durable ignored
record is `artifacts/verification/mot20-01-track-72-focus-observation.json`.

## Source Integrity And Limits

SHA-256 manifests cover every configured `seqinfo`, annotation, and enumerated
image: 2,249 unique files before and after full acceptance. The manifests are
equal. Their canonical aggregate digest is
`50bcbdb30e1fadfd7f51f59adafa0a514f4f947bcd7421982e9f0b99a338ae66`.

Corrected tracked MOT20-06/08 exports remain absent. Real tracked evidence is
therefore MOT20-01 ground truth only; 06/08 tracked-result, varying-confidence,
and gap journeys remain synthetic-only as detailed in `track-viz/README.md`.
