from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests


def task_data_form() -> dict[str, str]:
    """Return the scalar fields for CVAT's multipart task-data request."""
    return {
        "image_quality": "100",
        "sorting_method": "lexicographical",
        "use_zip_chunks": "true",
    }


def shared_file_fields(server_files: list[str]) -> dict[str, tuple[None, str]]:
    """Encode shared-file paths as indexed multipart text fields for CVAT."""
    if not server_files or any(not isinstance(path, str) or not path for path in server_files):
        raise ValueError("server_files must be a non-empty list of paths")
    return {f"server_files[{index}]": (None, path) for index, path in enumerate(server_files)}


class CvatClient:
    """Small CVAT REST client with conservative idempotency checks."""

    def __init__(self, url: str, username: str, password: str, *, timeout: tuple[float, float] = (10.0, 120.0)) -> None:
        self.base = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        response = self.session.post(f"{self.base}/api/auth/login", json={"username": username, "password": password}, timeout=timeout)
        self._check(response, "login")
        self.session.headers["Authorization"] = f"Token {response.json()['key']}"

    def _check(self, response: requests.Response, action: str) -> requests.Response:
        if response.status_code >= 400:
            raise RuntimeError(f"{action}: HTTP {response.status_code} {response.text[:500]}")
        return response

    def _find(self, resource: str, name: str, **params: Any) -> dict[str, Any] | None:
        page = 1
        seen = 0
        while True:
            body = self._check(self.session.get(f"{self.base}/api/{resource}", params={**params, "search": name, "page": page, "page_size": 100}, timeout=self.timeout), f"search {resource}").json()
            items, count = body.get("results"), body.get("count")
            if not isinstance(items, list) or not isinstance(count, int):
                raise RuntimeError(f"search {resource}: malformed response")
            match = next((item for item in items if isinstance(item, dict) and (item.get("name") == name or item.get("username") == name)), None)
            if match:
                return match
            seen += len(items)
            if seen >= count:
                return None
            if not items:
                raise RuntimeError(f"search {resource}: incomplete pagination")
            page += 1

    def find_user(self, username: str) -> dict[str, Any] | None:
        return self._find("users", username)

    def ensure_project(self, project: dict[str, Any]) -> dict[str, Any]:
        name = project.get("name")
        labels = project.get("labels")
        if not isinstance(name, str) or not name or not isinstance(labels, list) or not labels:
            raise ValueError("project config must contain a name and labels")
        found = self._find("projects", name)
        if not found:
            found = self._check(self.session.post(f"{self.base}/api/projects", json={"name": name, "labels": labels}, timeout=self.timeout), "create project").json()
        full = self._check(self.session.get(f"{self.base}/api/projects/{full_id(found)}", timeout=self.timeout), "get project").json()
        actual = full.get("labels")
        if isinstance(actual, dict):
            actual = self._list("labels", project_id=full_id(found))
        expected_contract = sorted((item.get("name"), item.get("type", "rectangle")) for item in labels if isinstance(item, dict))
        actual_contract = sorted((item.get("name"), item.get("type", "rectangle")) for item in actual if isinstance(item, dict)) if isinstance(actual, list) else []
        if expected_contract != actual_contract:
            raise RuntimeError(f"existing project label schema differs: {name}")
        return full

    def _list(self, resource: str, **params: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        page = 1
        while True:
            body = self._check(self.session.get(f"{self.base}/api/{resource}", params={**params, "page": page, "page_size": 100}, timeout=self.timeout), f"list {resource}").json()
            items, count = body.get("results"), body.get("count")
            if not isinstance(items, list) or not isinstance(count, int):
                raise RuntimeError(f"list {resource}: malformed response")
            result.extend(item for item in items if isinstance(item, dict))
            if len(result) >= count:
                return result
            if not items:
                raise RuntimeError(f"list {resource}: incomplete pagination")
            page += 1

    def ensure_task(self, *, task_name: str, project_id: int, image_names: list[str], server_files: list[str], assignee_id: int) -> tuple[int, str]:
        existing = self._find("tasks", task_name, project_id=project_id)
        if existing:
            task_id = full_id(existing)
            self._verify_task(task_id, task_name, project_id, image_names, assignee_id)
            return task_id, "existing"
        task = self._check(self.session.post(f"{self.base}/api/tasks", json={"name": task_name, "project_id": project_id, "segment_size": len(image_names)}, timeout=self.timeout), f"create task {task_name}").json()
        task_id = full_id(task)
        try:
            self._check(self.session.post(f"{self.base}/api/tasks/{task_id}/data", data=task_data_form(), files=shared_file_fields(server_files), headers={"Upload-Start": "true", "Upload-Finish": "true"}, timeout=self.timeout), f"upload task {task_name} images")
            self._wait_for_data(task_id)
            jobs = self._list("jobs", task_id=task_id)
            if len(jobs) != 1:
                raise RuntimeError(f"new task must have exactly one job: {task_name}")
            self._check(self.session.patch(f"{self.base}/api/jobs/{full_id(jobs[0])}", json={"assignee": assignee_id}, timeout=self.timeout), f"assign task {task_name}")
            self._verify_task(task_id, task_name, project_id, image_names, assignee_id)
        except Exception:
            raise RuntimeError(f"task {task_name} was created but not fully provisioned; inspect it before retrying") from None
        return task_id, "created"

    def _wait_for_data(self, task_id: int) -> None:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            state = self._check(self.session.get(f"{self.base}/api/tasks/{task_id}/status", timeout=self.timeout), "task status").json().get("state")
            if state == "Finished":
                return
            if state == "Failed":
                raise RuntimeError(f"task data ingestion failed: {task_id}")
            time.sleep(2)
        raise RuntimeError(f"task data ingestion timed out: {task_id}")

    def _verify_task(self, task_id: int, task_name: str, project_id: int, image_names: list[str], assignee_id: int) -> None:
        task = self._check(self.session.get(f"{self.base}/api/tasks/{task_id}", timeout=self.timeout), "get task").json()
        metadata = self._check(self.session.get(f"{self.base}/api/tasks/{task_id}/data/meta", timeout=self.timeout), "get task data metadata").json()
        jobs = self._list("jobs", task_id=task_id)
        names = [Path(str(frame.get("name", ""))).name for frame in metadata.get("frames", []) if isinstance(frame, dict)]
        assignee = jobs[0].get("assignee") if len(jobs) == 1 else None
        actual_assignee = assignee.get("id") if isinstance(assignee, dict) else assignee
        if task.get("name") != task_name or task.get("project_id") != project_id or task.get("size") != len(image_names) or names != image_names or len(jobs) != 1 or actual_assignee != assignee_id:
            raise RuntimeError(f"existing task is unsafe to reuse: {task_name}")

    def get_task(self, task_name: str, project_id: int) -> dict[str, Any]:
        task = self._find("tasks", task_name, project_id=project_id)
        if not task:
            raise ValueError(f"CVAT task does not exist: {task_name}")
        return self._check(self.session.get(f"{self.base}/api/tasks/{full_id(task)}", timeout=self.timeout), "get task").json()

    def task_frame_names(self, task_id: int) -> list[str]:
        metadata = self._check(self.session.get(f"{self.base}/api/tasks/{task_id}/data/meta", timeout=self.timeout), "get task data metadata").json()
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not all(isinstance(frame, dict) for frame in frames):
            raise RuntimeError(f"task has malformed frame metadata: {task_id}")
        return [Path(str(frame.get("name", ""))).name for frame in frames]

    def project_labels(self, project_id: int) -> list[dict[str, Any]]:
        return self._list("labels", project_id=project_id)

    def replace_annotations(self, task_id: int, payload: dict[str, Any]) -> None:
        self._check(self.session.put(f"{self.base}/api/tasks/{task_id}/annotations", json=payload, timeout=self.timeout), "replace task annotations")

    def annotations(self, task_id: int) -> dict[str, Any]:
        return self._check(self.session.get(f"{self.base}/api/tasks/{task_id}/annotations", timeout=self.timeout), "get task annotations").json()


def full_id(item: dict[str, Any]) -> int:
    try:
        return int(item["id"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("CVAT response has no valid id") from error
