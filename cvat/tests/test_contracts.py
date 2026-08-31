from __future__ import annotations

import configparser
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot20_cvat.contracts import (  # noqa: E402
    build_task_plan,
    build_tracks,
    parse_mot_rows,
    read_sequences,
    validate_task_plan,
)


class Mot20CvatContractsTest(unittest.TestCase):
    def make_dataset(self) -> Path:
        directory = Path(tempfile.mkdtemp()) / "test"
        for name, length in (("MOT20-04", 5), ("MOT20-06", 3)):
            sequence = directory / name
            (sequence / "img1").mkdir(parents=True)
            config = configparser.ConfigParser()
            config["Sequence"] = {
                "name": name,
                "imDir": "img1",
                "seqLength": str(length),
                "imWidth": "1920",
                "imHeight": "1080",
                "imExt": ".jpg",
            }
            with (sequence / "seqinfo.ini").open("w") as stream:
                config.write(stream)
            for frame in range(1, length + 1):
                (sequence / "img1" / f"{frame:06d}.jpg").touch()
        return directory

    def test_task_plan_is_complete_balanced_and_deterministic(self) -> None:
        root = self.make_dataset()
        sequences = read_sequences(root)
        first = build_task_plan(sequences, ["alice", "bob"], max_images_per_task=2)
        second = build_task_plan(sequences, ["alice", "bob"], max_images_per_task=2)

        self.assertEqual(first, second)
        validate_task_plan(first, sequences)
        loads = {name: 0 for name in ("alice", "bob")}
        for assignment in first["assignments"]:
            loads[assignment["assignee"]] += assignment["stop_frame"] - assignment["start_frame"] + 1
        self.assertLessEqual(max(loads.values()) - min(loads.values()), 2)

    def test_mot_rows_become_task_local_tracks(self) -> None:
        rows = parse_mot_rows("2,17,10,20,30,40,1,1,0.9\n4,17,12,21,30,40,1,1,0.8\n")
        tracks = build_tracks(rows, label_id=9, start_frame=2, stop_frame=4)

        self.assertEqual(len(tracks), 1)
        self.assertEqual([shape["frame"] for shape in tracks[0]["shapes"]], [0, 2])
        self.assertEqual(tracks[0]["label_id"], 9)
        self.assertEqual(tracks[0]["shapes"][0]["points"], [10.0, 20.0, 40.0, 60.0])

    def test_invalid_image_contract_is_rejected(self) -> None:
        root = self.make_dataset()
        (root / "MOT20-04" / "img1" / "000003.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "image count"):
            read_sequences(root)


if __name__ == "__main__":
    unittest.main()
