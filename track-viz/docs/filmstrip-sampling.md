# Filmstrip Sampling

The track filmstrip is deterministic and contains at most 64 observations. Its
current observation is identified by source-row index, not by frame alone.

## Algorithm

1. Sort track observations by `(frame, source-row index)`.
2. Reject the request if the current source-row index is not in that exact
   sequence-local track.
3. If the track has at most 64 observations, return all observations in sorted
   order.
4. Otherwise, pin the first, current, and last positions. A current observation
   at an endpoint occupies one position rather than two.
5. Divide the unpinned positions into an earlier side, excluding the first and
   current positions, and a later side, excluding the current and last
   positions.
6. Give one remaining slot to each nonempty side when slots are available.
7. Allocate all other slots in proportion to each side's remaining capacity.
   The earlier side receives
   `floor(extra slots * earlier capacity / total capacity)`; the later side
   receives the remainder up to its capacity. Any capacity-limited remainder
   is assigned to the earlier side and then the later side.
8. Select `k` positions from each side of length `n` as follows:
   - `k = 0`: select none.
   - `k = 1`: select position `floor((n - 1) / 2)` on that side.
   - `1 < k < n`: for sample number `j` from `0` through `k - 1`, select side
     position `floor(j * (n - 1) / (k - 1))`.
   - `k >= n`: select the whole side.
9. Return the union of pinned and side positions in chronological order.

This rule always retains the current observation and both track endpoints. It
also retains representative earlier and later evidence whenever those sides
exist. Integer arithmetic makes the selected source rows reproducible across
runs.

## Dependency Compatibility

The first Supervision integration pins `supervision==0.27.0`, `numpy==2.2.6`,
and retains `pillow==11.3.0`. This combination installs on the project's Python
3.12 runtime, satisfies Supervision's Python `>=3.9` requirement, imports
successfully, and passes `pip check`. Supervision is limited to normalized
detection collections and offline adapters; interactive rendering remains in
the browser.