"""Optional offline vision/model proposal adapter.

The solver never requires this module's optional backend. A backend is injected
as a local callable so the Kaggle notebook can use a packaged model without
allowing network access or coupling the core solver to one ML framework.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from verification import verify_model_payload
except ImportError:  # direct execution from the repository root
    from hybrid_verification import verify_model_payload


class ProposalError(Exception):
    pass


def grid_to_text(grid: Any) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def build_prompt(task: Dict[str, Any]) -> str:
    """Build a strict prompt containing only the task demonstrations/inputs."""
    blocks = [
        "You are an offline ARC grid transformation proposer.",
        "Infer one deterministic transformation from the training pairs.",
        "Return JSON only with keys train_predictions, test_predictions, program.",
        "Every prediction must be a rectangular integer grid with cells 0 through 9.",
        "Do not use outside information, hidden labels, or stochastic guesses.",
        "The verifier will reject any training mismatch.",
        "",
        "TRAINING PAIRS:",
    ]
    for index, pair in enumerate(task.get("train", [])):
        blocks.extend([
            f"PAIR {index} INPUT:", grid_to_text(pair["input"]),
            f"PAIR {index} OUTPUT:", grid_to_text(pair["output"]), "",
        ])
    blocks.append("TEST INPUTS:")
    for index, pair in enumerate(task.get("test", [])):
        blocks.extend([f"TEST {index}:", grid_to_text(pair["input"]), ""])
    blocks.append('JSON schema: {"train_predictions": [[...]], "test_predictions": [[...]], "program": "..."}')
    return "\n".join(blocks)


def parse_json_response(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ProposalError("model backend must return a dict or JSON string")
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProposalError(f"model response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProposalError("model JSON must be an object")
    return payload


class OfflineVisionProposer:
    """Framework-neutral adapter for a locally packaged model.

    `generate` must be a local callable. It may wrap a Transformers, ONNX,
    TensorRT, or custom inference implementation, but it must not make network
    requests. The adapter only accepts a proposal after exact train verification.
    """

    def __init__(self, generate: Callable[[str], Any], timeout_seconds: float = 45.0):
        self.generate = generate
        self.timeout_seconds = timeout_seconds

    def propose(self, task: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        start = time.time()
        try:
            raw = self.generate(build_prompt(task))
            if time.time() - start > self.timeout_seconds:
                return None, "model proposal timed out"
            payload = parse_json_response(raw)
            ok, reason, _ = verify_model_payload(task, payload)
            if not ok:
                return None, reason
            return payload, "verified model proposal"
        except Exception as exc:
            return None, f"model proposal unavailable: {exc}"


class DisabledVisionProposer:
    def propose(self, task: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        return None, "offline model backend disabled"
