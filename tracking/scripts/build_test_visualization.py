#!/usr/bin/env python3
"""Render a self-contained HTML comparison of detection and tracking output.

Built for the MOT20 test split, where nothing can be scored locally: with no
ground truth, looking at the boxes is the only inspection available. The page
therefore optimises for comparing result sets against EACH OTHER on identical
frames, not for reporting a metric.

Annotation is done by `supervision`, so the drawing conventions are the library's
rather than this repository's. Two of its behaviours are load-bearing here:

- ``ColorLookup.TRACK`` colours a box by its ``tracker_id``. The same identity
  therefore keeps one colour across frames AND across result sets, which makes an
  identity switch read as a colour change on a body that did not move. That is
  the failure mode the test submissions actually differ on, so it needs to be
  visible rather than inferred.
- ``BoxCornerAnnotator`` draws corner brackets instead of a closed rectangle.
  Boxes that DTI invented (``postprocess_tracks.py`` writes confidence ``-1`` for
  them) are drawn that way, so manufactured boxes are distinguishable at a glance
  from boxes the detector actually produced.

Box coordinates are used exactly as stored. Only the background image is
downscaled, and the boxes are scaled with it by the same factor, so nothing is
re-derived or rounded into place.

The page is offline and self-contained: no network, no server, no build step.
Run it with `.venv`, which is where `supervision` and `cv2` are installed.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tracking" / "src"))

from mot20_tracking.sequences import read_split  # noqa: E402

INTERPOLATED_CONFIDENCE = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        required=True,
        metavar="LABEL=KIND=DIR",
        help="a result set to render, e.g. "
        "'arm E19 raw=track=artifacts/tracking/tracks/<combination>/test'. "
        "KIND is 'track' (10-column, coloured by identity) or 'det' "
        "(9- or 10-column detections, single colour, no identity)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequences", nargs="*", default=None, help="default: all in the split")
    parser.add_argument("--frames-per-sequence", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=1100)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="hide boxes below this confidence. Interpolated rows carry -1 and are "
        "always kept, since hiding them would conceal the thing being compared",
    )
    parser.add_argument("--labels", action="store_true", help="draw identity labels on tracks")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_set(raw: str) -> tuple[str, str, Path]:
    label, _, rest = raw.partition("=")
    kind, _, directory = rest.partition("=")
    if not label or kind not in {"track", "det"} or not directory:
        raise SystemExit(f"bad --set {raw!r}; expected LABEL=track|det=DIR")
    return label, kind, REPO_ROOT / directory


def find_rows_file(directory: Path, sequence: str) -> Path | None:
    """Locate one sequence's rows under either artifact layout.

    Tracks and submission exports are flat ``<sequence>.txt``; L1 detections are
    ``<sequence>/det.txt``. Both layouts exist in this repository.
    """
    for candidate in (directory / f"{sequence}.txt", directory / sequence / "det.txt"):
        if candidate.exists():
            return candidate
    return None


def load_rows(path: Path, frames: set[int], min_score: float) -> dict[int, np.ndarray]:
    """Return ``{frame: rows}`` restricted to the requested frames."""
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.size == 0:
        return {}
    raw = raw[np.isin(raw[:, 0].astype(np.int64), list(frames))]
    if raw.size == 0:
        return {}
    score = raw[:, 6] if raw.shape[1] > 6 else np.ones(len(raw))
    # -1 marks a DTI-interpolated box. It is not a low-confidence detection and
    # must not be removed by a confidence threshold.
    keep = (score >= min_score) | (score == INTERPOLATED_CONFIDENCE)
    raw = raw[keep]
    return {int(f): raw[raw[:, 0].astype(np.int64) == f] for f in np.unique(raw[:, 0].astype(np.int64))}


def to_detections(rows: np.ndarray, scale: float, kind: str) -> tuple[sv.Detections, sv.Detections]:
    """Split one frame's rows into (real, interpolated) supervision Detections.

    ``xywh`` is converted to ``xyxy`` and multiplied by the background's downscale
    factor so boxes land on the resized image without being re-derived.
    """
    if rows.size == 0:
        empty = sv.Detections.empty()
        return empty, empty
    xyxy = np.stack(
        [rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], axis=1
    ) * scale
    confidence = rows[:, 6].astype(np.float32) if rows.shape[1] > 6 else np.ones(len(rows), np.float32)
    identity = rows[:, 1].astype(int)
    # Detections carry no identity; give supervision a stable per-row id so
    # ColorLookup.TRACK does not colour every box the same by accident.
    tracker_id = identity if kind == "track" else np.zeros(len(rows), dtype=int)

    def build(mask: np.ndarray) -> sv.Detections:
        if not mask.any():
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=xyxy[mask].astype(np.float32),
            confidence=confidence[mask],
            class_id=np.zeros(int(mask.sum()), dtype=int),
            tracker_id=tracker_id[mask],
        )

    interpolated = confidence == INTERPOLATED_CONFIDENCE
    return build(~interpolated), build(interpolated)


def annotate(
    image: np.ndarray, rows: np.ndarray, scale: float, kind: str, color, draw_labels: bool
) -> tuple[np.ndarray, int, int]:
    """Draw one result set's boxes onto a frame with supervision."""
    real, interpolated = to_detections(rows, scale, kind)
    lookup = sv.ColorLookup.TRACK if kind == "track" else sv.ColorLookup.INDEX
    box = sv.BoxAnnotator(color=color, color_lookup=lookup, thickness=2)
    # Corner brackets, not a closed rectangle, so DTI's manufactured boxes are
    # distinguishable from boxes the detector actually produced.
    corner = sv.BoxCornerAnnotator(color=color, color_lookup=lookup, thickness=2, corner_length=9)
    scene = image
    if len(real):
        scene = box.annotate(scene=scene, detections=real)
    if len(interpolated):
        scene = corner.annotate(scene=scene, detections=interpolated)
    if draw_labels and kind == "track" and len(real):
        label = sv.LabelAnnotator(
            color=color, color_lookup=lookup, text_scale=0.35, text_thickness=1, text_padding=2
        )
        scene = label.annotate(
            scene=scene, detections=real, labels=[str(i) for i in real.tracker_id]
        )
    return scene, len(real), len(interpolated)


def sample_frames(length: int, count: int) -> list[int]:
    """Evenly spaced frames, endpoints included, without duplicates."""
    if count >= length:
        return list(range(1, length + 1))
    return sorted({int(round(v)) for v in np.linspace(1, length, count)})


PAGE = """<!DOCTYPE html>
<html lang="en"><meta charset="utf-8">
<title>MOT20 {split} — detection and tracking comparison</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; background:#111418; color:#e6e6e6;
        font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
 header {{ position:sticky; top:0; z-index:10; background:#161a20; padding:12px 18px;
           border-bottom:1px solid #2a3038; }}
 h1 {{ font-size:16px; margin:0 0 4px; font-weight:600; }}
 .note {{ color:#93a1b0; font-size:12px; max-width:96ch; }}
 .controls {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-top:10px;
              font-size:13px; }}
 .controls label {{ display:flex; gap:6px; align-items:center; cursor:pointer; }}
 main {{ padding:18px; }}
 .seq {{ margin-bottom:28px; }}
 .seq h2 {{ font-size:14px; font-weight:600; margin:0 0 10px; color:#cbd5e1; }}
 .row {{ display:grid; gap:12px; margin-bottom:14px;
         grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); }}
 figure {{ margin:0; background:#0d1014; border:1px solid #232a33; border-radius:6px;
           overflow:hidden; }}
 figcaption {{ padding:6px 10px; font-size:12px; color:#93a1b0;
               border-bottom:1px solid #232a33; display:flex; justify-content:space-between;
               gap:10px; }}
 figcaption b {{ color:#dbe4ee; font-weight:600; }}
 img {{ display:block; width:100%; height:auto; }}
 .hint {{ color:#7c8b9a; font-size:11px; margin-top:2px; }}
</style>
<header>
 <h1>MOT20 {split} — detection and tracking comparison</h1>
 <div class="note">{note}</div>
 <div class="controls" id="controls"></div>
</header>
<main id="main"></main>
<script>
const DATA = {data};
const SETS = {sets};
function render() {{
  const main = document.getElementById('main');
  main.innerHTML = '';
  for (const seq of DATA.sequences) {{
    const sec = document.createElement('section');
    sec.className = 'seq';
    sec.innerHTML = `<h2>${{seq.name}} — ${{seq.width}}×${{seq.height}}, ${{seq.length}} frames</h2>`;
    for (const fr of seq.frames) {{
      const row = document.createElement('div'); row.className = 'row';
      let any = false;
      for (const set of SETS) {{
        const cb = document.getElementById('set-' + set.key);
        if (!cb || !cb.checked) continue;
        const panel = fr.sets[set.key];
        if (!panel) continue;
        any = true;
        const fig = document.createElement('figure');
        fig.innerHTML =
          `<figcaption><span><b>${{set.label}}</b> — frame ${{fr.n}}</span>` +
          `<span>${{panel.n}} boxes` +
          (panel.i ? ` · <b>${{panel.i}}</b> interpolated` : '') + `</span></figcaption>` +
          `<img loading="lazy" src="data:image/jpeg;base64,${{panel.img}}" alt="">`;
        row.appendChild(fig);
      }}
      if (any) sec.appendChild(row);
    }}
    main.appendChild(sec);
  }}
}}
const c = document.getElementById('controls');
for (const s of SETS) {{
  const l = document.createElement('label');
  l.innerHTML = `<input type="checkbox" id="set-${{s.key}}" checked><span>${{s.label}}</span>`;
  c.appendChild(l);
}}
c.addEventListener('change', render);
render();
</script>
</html>
"""


def main() -> None:
    args = parse_args()
    sets = [parse_set(raw) for raw in args.sets]
    for label, _, directory in sets:
        if not directory.is_dir():
            raise SystemExit(f"{label}: not a directory: {directory}")

    sequences = read_split(args.split, repo_root=REPO_ROOT)
    if args.sequences:
        wanted = set(args.sequences)
        sequences = [s for s in sequences if s.name in wanted]
        missing = wanted - {s.name for s in sequences}
        if missing:
            raise SystemExit(f"sequences not in split {args.split}: {sorted(missing)}")

    palette = sv.ColorPalette.DEFAULT
    single = [sv.Color.from_hex(h) for h in ("#4cc9f0", "#f72585", "#ffd166", "#06d6a0")]
    set_meta = [
        {"key": f"s{i}", "label": label, "kind": kind}
        for i, (label, kind, _) in enumerate(sets)
    ]

    payload = []
    for sequence in sequences:
        frames = sample_frames(sequence.length, args.frames_per_sequence)
        frame_set = set(frames)
        loaded = []
        for (label, kind, directory) in sets:
            path = find_rows_file(directory, sequence.name)
            if path is None:
                print(f"  {sequence.name}: {label}: no rows under {directory}", flush=True)
                loaded.append({})
                continue
            loaded.append(load_rows(path, frame_set, args.min_score))

        entries = []
        for number in frames:
            image = cv2.imread(str(sequence.frame_path(number)))
            if image is None:
                raise SystemExit(f"could not read {sequence.frame_path(number)}")
            height, width = image.shape[:2]
            scale = min(1.0, args.max_width / width)
            if scale < 1.0:
                image = cv2.resize(
                    image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
                )
            panels = {}
            for index, (meta, (label, kind, _)) in enumerate(zip(set_meta, sets)):
                rows = loaded[index].get(number, np.empty((0, 10)))
                color = palette if kind == "track" else single[index % len(single)]
                scene, n_real, n_interp = annotate(
                    image.copy(), rows, scale, kind, color, args.labels
                )
                ok, buffer = cv2.imencode(
                    ".jpg", scene, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
                )
                if not ok:
                    raise SystemExit(f"JPEG encode failed for {sequence.name} frame {number}")
                panels[meta["key"]] = {
                    "img": base64.b64encode(buffer.tobytes()).decode("ascii"),
                    "n": n_real + n_interp,
                    "i": n_interp,
                }
            entries.append({"n": number, "sets": panels})
        payload.append(
            {
                "name": sequence.name,
                "width": sequence.width,
                "height": sequence.height,
                "length": sequence.length,
                "frames": entries,
            }
        )
        print(f"  {sequence.name}: {len(frames)} frames x {len(sets)} sets rendered", flush=True)

    note = (
        f"Annotated with supervision {sv.__version__}. {len(set_meta)} result sets over "
        f"{len(payload)} sequences, {args.frames_per_sequence} evenly spaced frames each. "
        f"Track boxes are coloured by identity (ColorLookup.TRACK), so one person keeps one "
        f"colour across frames and across sets, and a colour change on a stationary body is an "
        f"identity switch. Corner-bracket boxes were invented by DTI interpolation "
        f"(confidence -1), not detected. MOT20 {args.split} has no public ground truth, so "
        f"nothing shown here is scored."
    )
    page = PAGE.format(
        split=html.escape(args.split),
        note=html.escape(note),
        data=json.dumps({"sequences": payload}, separators=(",", ":")),
        sets=json.dumps(set_meta, separators=(",", ":")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"\nwrote {args.output} ({args.output.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
