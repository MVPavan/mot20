"""MOT20 sequence metadata."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

SPLIT_ROOTS = {
    "val_half": Path("datasets/val_half"),
    "train": Path("datasets/MOT20/train"),
    "test": Path("datasets/MOT20/test"),
}


@dataclass(frozen=True)
class Sequence:
    """One MOT20 sequence as described by its ``seqinfo.ini``."""

    name: str
    root: Path
    image_dir: Path
    length: int
    width: int
    height: int
    frame_rate: int
    image_ext: str

    def frame_path(self, frame_id: int) -> Path:
        """Path to a 1-based frame, using MOT20's six-digit naming."""
        if not 1 <= frame_id <= self.length:
            raise ValueError(f"{self.name}: frame {frame_id} outside 1..{self.length}")
        return self.image_dir / f"{frame_id:06d}{self.image_ext}"


def read_sequence(sequence_root: Path) -> Sequence:
    """Parse ``seqinfo.ini`` for one sequence directory."""
    sequence_root = Path(sequence_root)
    config_path = sequence_root / "seqinfo.ini"
    parser = configparser.ConfigParser()
    if not parser.read(config_path) or "Sequence" not in parser:
        raise ValueError(f"invalid MOT20 sequence metadata: {config_path}")
    values = parser["Sequence"]
    return Sequence(
        name=values["name"],
        root=sequence_root,
        image_dir=sequence_root / values["imDir"],
        length=int(values["seqLength"]),
        width=int(values["imWidth"]),
        height=int(values["imHeight"]),
        frame_rate=int(values["frameRate"]),
        image_ext=values["imExt"],
    )


def read_split(split: str, repo_root: Path | None = None) -> list[Sequence]:
    """Return every sequence of a split, ordered by name."""
    if split not in SPLIT_ROOTS:
        raise ValueError(f"unknown split {split!r}; expected one of {tuple(SPLIT_ROOTS)}")
    root = Path(repo_root or ".") / SPLIT_ROOTS[split]
    if not root.is_dir():
        raise FileNotFoundError(f"split directory not found: {root}")
    directories = sorted(path for path in root.iterdir() if (path / "seqinfo.ini").is_file())
    if not directories:
        raise FileNotFoundError(f"no MOT20 sequences under {root}")
    return [read_sequence(path) for path in directories]
