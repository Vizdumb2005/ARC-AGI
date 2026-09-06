"""Kaggle-safe data discovery and exact submission generation."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from verification import validate_submission
except ImportError:
    from hybrid_verification import validate_submission


CHALLENGE_SUFFIX = "_challenges.json"


def discover_json_files(root: str = "/kaggle/input") -> list:
    found = []
    if not os.path.isdir(root):
        return found
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".json"):
                found.append(os.path.join(base, name))
    return sorted(found)


def select_test_challenges(paths: Iterable[str]) -> str:
    paths = list(paths)
    preferred = [p for p in paths if os.path.basename(p) == "arc-agi_test_challenges.json"]
    if preferred:
        return preferred[0]
    fallbacks = [p for p in paths if "test" in os.path.basename(p).lower() and "challenge" in os.path.basename(p).lower()]
    if len(fallbacks) != 1:
        raise FileNotFoundError("could not uniquely identify the test challenge JSON")
    return fallbacks[0]


def load_tasks(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("challenge file must contain a non-empty task object")
    for task_id, task in tasks.items():
        if not isinstance(task_id, str) or not isinstance(task, dict):
            raise ValueError("invalid task record")
        if not isinstance(task.get("train"), list) or not isinstance(task.get("test"), list):
            raise ValueError(f"{task_id}: expected train and test lists")
    return tasks


def make_submission(tasks: Dict[str, Any], portfolio, time_budget: float = 20.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    submission: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    for task_id, task in tasks.items():
        first, second, metadata = portfolio.solve(task, time_budget=time_budget)
        rows = []
        for index in range(len(task.get("test", []))):
            rows.append({
                "attempt_1": first.outputs[index],
                "attempt_2": second.outputs[index],
            })
        submission[task_id] = rows
        diagnostics[task_id] = metadata
    ok, reason = validate_submission(submission, tasks)
    if not ok:
        raise ValueError(f"submission validation failed: {reason}")
    return submission, diagnostics


def write_submission(submission: Dict[str, Any], path: str = "/kaggle/working/submission.json") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(submission, handle, separators=(",", ":"))
    with open(path, "r", encoding="utf-8") as handle:
        reparsed = json.load(handle)
    if reparsed != submission:
        raise ValueError("submission changed after JSON round-trip")
    return path
