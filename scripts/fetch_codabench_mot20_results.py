#!/usr/bin/env python3
"""Fetch the public MOT20 Codabench leaderboard and detailed results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://www.codabench.org"
COMPETITION_ID = 10050
PHASE_ID = 16383
LEADERBOARD_URL = (
    f"{BASE_URL}/api/phases/{PHASE_ID}/get_leaderboard/?page=1&page_size=500"
)
DETAIL_API_TEMPLATE = f"{BASE_URL}/api/submissions/{{submission_id}}/get_detail_result/"


class ResultsTableParser(HTMLParser):
    """Extract the first HTML table while preserving displayed cell values."""

    def __init__(self) -> None:
        super().__init__()
        self.headers: list[str] = []
        self.rows: list[list[str]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table" and not self._in_table:
            self._in_table = True
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_row and tag in {"th", "td"}:
            self._in_cell = True
            self._cell_text = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in_cell and tag in {"th", "td"}:
            self._current_row.append(" ".join("".join(self._cell_text).split()))
            self._in_cell = False
        elif self._in_table and self._in_row and tag == "tr":
            if self._current_row:
                if not self.headers:
                    self.headers = self._current_row
                else:
                    self.rows.append(self._current_row)
            self._in_row = False
        elif self._in_table and tag == "table":
            self._in_table = False


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "mot20-codabench-results-fetcher/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def fetch_detail(submission_id: int) -> dict[str, Any]:
    detail_api_url = DETAIL_API_TEMPLATE.format(submission_id=submission_id)
    detail_url = fetch_json(detail_api_url)
    html = fetch_bytes(detail_url).decode("utf-8")
    parser = ResultsTableParser()
    parser.feed(html)
    if not parser.headers or not parser.rows:
        raise ValueError(f"No detailed results table found for submission {submission_id}")
    return {
        "api_url": detail_api_url,
        "signed_url": detail_url,
        "columns": parser.headers,
        "rows": [dict(zip(parser.headers, row, strict=False)) for row in parser.rows],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect() -> dict[str, Any]:
    leaderboard = fetch_json(LEADERBOARD_URL)
    submissions = leaderboard.get("submissions", [])
    if not isinstance(submissions, list) or not submissions:
        raise ValueError("Leaderboard response did not contain submissions")

    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rank, submission in enumerate(submissions, start=1):
        submission_id = int(submission["id"])
        entry = {"rank": rank, **submission}
        try:
            entry["detailed_result"] = fetch_detail(submission_id)
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as error:
            entry["detailed_result"] = None
            failures.append({"id": submission_id, "error": str(error)})
        entries.append(entry)
        print(f"{rank}/{len(submissions)}: {submission_id}", file=sys.stderr)

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "competition": {"id": COMPETITION_ID, "title": leaderboard.get("title")},
        "phase": {"id": PHASE_ID, "title": leaderboard.get("title")},
        "source": {
            "leaderboard_api": LEADERBOARD_URL,
            "count_reported": leaderboard.get("count"),
        },
        "entry_count": len(entries),
        "failed_detail_count": len(failures),
        "failures": failures,
        "entries": entries,
    }


def write_exports(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.fromisoformat(payload["fetched_at"])
    stem = f"codabench_mot20_{fetched_at.strftime('%Y%m%dT%H%M%SZ')}"
    json_path = output_dir / f"{stem}.json"
    summary_path = output_dir / f"{stem}_leaderboard.csv"
    detail_path = output_dir / f"{stem}_detailed.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for entry in payload["entries"]:
        fact_sheet = entry.get("fact_sheet_answers", {})
        scores = {score["column_key"]: score["score"] for score in entry.get("scores", [])}
        summary_rows.append(
            {
                "rank": entry["rank"],
                "submission_id": entry["id"],
                "participant": entry.get("owner"),
                "created_when": entry.get("created_when"),
                "method_name": fact_sheet.get("name"),
                "paper": fact_sheet.get("paper"),
                "speed": fact_sheet.get("speed"),
                "venue": fact_sheet.get("venue"),
                "detections": fact_sheet.get("detections"),
                "online_tracker": fact_sheet.get("online_tracker"),
                **scores,
            }
        )
        detailed = entry.get("detailed_result")
        if detailed:
            for row in detailed["rows"]:
                detail_rows.append(
                    {
                        "rank": entry["rank"],
                        "submission_id": entry["id"],
                        "participant": entry.get("owner"),
                        **row,
                    }
                )

    summary_fields = list(summary_rows[0]) if summary_rows else []
    score_fields = [
        "HOTA",
        "MOTA",
        "IDF1",
        "MT",
        "ML",
        "CLR_FP",
        "CLR_FN",
        "CLR_Re",
        "CLR_Pr",
        "AssA",
        "DetA",
        "AssRe",
        "AssPr",
        "DetRe",
        "DetPr",
        "LocA",
        "FAR",
        "IDSW",
        "Frag",
    ]
    summary_fields = [field for field in summary_fields if field not in score_fields] + score_fields
    detail_fields = ["rank", "submission_id", "participant"] + (
        payload["entries"][0]["detailed_result"]["columns"]
        if payload["entries"] and payload["entries"][0].get("detailed_result")
        else []
    )
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(detail_path, detail_rows, detail_fields)
    print(json_path)
    print(summary_path)
    print(detail_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/codabench_mot20"),
        help="Directory for generated exports (default: results/codabench_mot20)",
    )
    args = parser.parse_args()
    try:
        payload = collect()
        write_exports(args.output_dir, payload)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, OSError) as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())