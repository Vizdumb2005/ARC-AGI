"""Exact verification and submission validation for the hybrid solver."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_grid(grid: Any, max_size: int = 30) -> Tuple[bool, str]:
    if not isinstance(grid, list) or not grid:
        return False, "grid must be a non-empty list"
    if len(grid) > max_size:
        return False, f"more than {max_size} rows"
    width = None
    for row in grid:
        if not isinstance(row, list) or not row:
            return False, "rows must be non-empty lists"
        width = len(row) if width is None else width
        if len(row) != width:
            return False, "grid is not rectangular"
        if len(row) > max_size:
            return False, f"more than {max_size} columns"
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > 9 for x in row):
            return False, "cells must be integers in 0..9"
    return True, "ok"


def validate_grid_list(outputs: Any, count: int, name: str) -> Tuple[bool, str]:
    if not isinstance(outputs, list) or len(outputs) != count:
        return False, f"{name} must contain {count} grids"
    for i, grid in enumerate(outputs):
        ok, reason = validate_grid(grid)
        if not ok:
            return False, f"{name}[{i}]: {reason}"
    return True, "ok"


def verify_training_predictions(task: Dict[str, Any], predictions: Any) -> Tuple[bool, str]:
    train = task.get("train", [])
    ok, reason = validate_grid_list(predictions, len(train), "train_predictions")
    if not ok:
        return False, reason
    for i, pair in enumerate(train):
        if predictions[i] != pair.get("output"):
            return False, f"training pair {i} mismatch"
    return True, "all training pairs match"


def verify_model_payload(task: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[bool, str, List[Any]]:
    if not isinstance(payload, dict):
        return False, "proposal must be an object", []
    train_outputs = payload.get("train_predictions", payload.get("train_outputs"))
    test_outputs = payload.get("test_predictions", payload.get("test_outputs"))
    ok, reason = verify_training_predictions(task, train_outputs)
    if not ok:
        return False, reason, []
    ok, reason = validate_grid_list(test_outputs, len(task.get("test", [])), "test_predictions")
    if not ok:
        return False, reason, []
    return True, "verified", test_outputs


def validate_submission(submission: Dict[str, Any], tasks: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(submission, dict):
        return False, "submission must be an object"
    if set(submission) != set(tasks):
        return False, "task IDs do not exactly match challenge IDs"
    for task_id, task in tasks.items():
        rows = submission[task_id]
        expected = len(task.get("test", []))
        if not isinstance(rows, list) or len(rows) != expected:
            return False, f"{task_id}: expected {expected} prediction rows"
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"attempt_1", "attempt_2"}:
                return False, f"{task_id}[{index}]: invalid attempt keys"
            for key in ("attempt_1", "attempt_2"):
                ok, reason = validate_grid(row[key])
                if not ok:
                    return False, f"{task_id}[{index}].{key}: {reason}"
    return True, "valid submission"
