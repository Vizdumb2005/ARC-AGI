"""ARC-AGI-2 Kaggle entrypoint for the hybrid portfolio.

The default path is fully offline and uses the repository's symbolic solver.
A packaged local model can be connected by replacing `build_vision_proposer`;
no network access is performed here.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from portfolio import HybridPortfolio
from submission import discover_json_files, load_tasks, make_submission, select_test_challenges, write_submission
from vision_proposer import DisabledVisionProposer


def build_vision_proposer():
    """Return the optional local model adapter.

    Keep disabled until a packaged, offline backend is explicitly supplied.
    A backend must return a strict JSON proposal accepted by verification.py.
    """
    return DisabledVisionProposer()


def run(input_root: str = "/kaggle/input", output_path: str = "/kaggle/working/submission.json", time_budget: float = 20.0):
    from search import solve_task

    challenge_path = select_test_challenges(discover_json_files(input_root))
    tasks = load_tasks(challenge_path)
    portfolio = HybridPortfolio(symbolic_solver=solve_task, vision_proposer=build_vision_proposer())
    submission, diagnostics = make_submission(tasks, portfolio, time_budget=time_budget)
    write_submission(submission, output_path)
    return {"challenge_path": challenge_path, "output_path": output_path, "task_count": len(tasks), "diagnostics": diagnostics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/submission.json")
    parser.add_argument("--time-budget", type=float, default=20.0)
    args = parser.parse_args()
    result = run(args.input_root, args.output, args.time_budget)
    print("Wrote", result["output_path"], "for", result["task_count"], "tasks")


if __name__ == "__main__":
    main()
