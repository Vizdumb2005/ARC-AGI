"""Hybrid candidate portfolio: symbolic search first, optional offline model second."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from verification import validate_grid
except ImportError:
    from hybrid_verification import validate_grid


@dataclass
class Candidate:
    source: str
    outputs: List[Any]
    program: Any = None
    mdl_cost: int = 10**9
    verified: bool = False
    confidence: float = 0.0

    def key(self) -> str:
        return repr(self.outputs)


class HybridPortfolio:
    """Collect verified candidates and return two safe attempts.

    The injected `symbolic_solver` should return the existing search.SearchResult
    shape. The optional `vision_proposer` must expose `propose(task)` and return
    `(payload, reason)`. No model is required for the portfolio to operate.
    """

    def __init__(self, symbolic_solver=None, vision_proposer=None):
        self.symbolic_solver = symbolic_solver
        self.vision_proposer = vision_proposer

    def _symbolic_candidate(self, task: Dict[str, Any], time_budget: float) -> Optional[Candidate]:
        if self.symbolic_solver is None:
            return None
        try:
            result = self.symbolic_solver(task, time_budget=time_budget)
        except Exception:
            return None
        if not getattr(result, "is_valid", lambda: False)():
            return None
        outputs = getattr(result, "output", None)
        if not isinstance(outputs, list):
            return None
        return Candidate(
            source=getattr(result, "engine", None) or "symbolic",
            outputs=outputs,
            program=getattr(result, "program", None),
            mdl_cost=int(getattr(result, "mdl_cost", 10**9) or 10**9),
            verified=True,
            confidence=float(getattr(result, "confidence", 1.0) or 1.0),
        )

    def _vision_candidate(self, task: Dict[str, Any]) -> Optional[Candidate]:
        if self.vision_proposer is None:
            return None
        try:
            payload, _reason = self.vision_proposer.propose(task)
        except Exception:
            return None
        if not payload:
            return None
        outputs = payload.get("test_predictions", payload.get("test_outputs"))
        if not isinstance(outputs, list):
            return None
        if any(not validate_grid(grid)[0] for grid in outputs):
            return None
        program = payload.get("program", "model_proposal")
        return Candidate(
            source="offline_vision",
            outputs=outputs,
            program=program,
            mdl_cost=max(1, len(repr(program))),
            verified=True,
            confidence=float(payload.get("confidence", 0.5) or 0.5),
        )

    def solve(self, task: Dict[str, Any], time_budget: float = 20.0) -> Tuple[Candidate, Candidate, Dict[str, Any]]:
        candidates: List[Candidate] = []
        symbolic = self._symbolic_candidate(task, time_budget)
        if symbolic:
            candidates.append(symbolic)
        vision = self._vision_candidate(task)
        if vision:
            candidates.append(vision)

        unique: List[Candidate] = []
        seen = set()
        for candidate in candidates:
            if candidate.key() not in seen:
                unique.append(candidate)
                seen.add(candidate.key())

        # Verified candidates outrank any fallback. MDL is a tie-breaker only
        # after exact training verification; source order is deterministic.
        unique.sort(key=lambda c: (not c.verified, c.mdl_cost, c.source))
        test_inputs = [pair.get("input") for pair in task.get("test", [])]
        fallback = Candidate(
            source="identity_fallback",
            outputs=copy.deepcopy(test_inputs),
            program="identity_fallback",
            mdl_cost=10**9,
            verified=False,
            confidence=0.0,
        )
        if not unique:
            unique = [fallback]
        first = unique[0]
        second = unique[1] if len(unique) > 1 else first
        metadata = {
            "sources": [first.source, second.source],
            "verified": [first.verified, second.verified],
            "distinct_attempts": first.key() != second.key(),
        }
        return first, second, metadata
