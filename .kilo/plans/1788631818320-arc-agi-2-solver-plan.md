# Implementation Plan: ARC-AGI-2 Solver

## Environment
- Python 3.10.12, standard library only (no numpy/torch/scipy)
- pip unavailable
- /kaggle/input and /kaggle/working will exist at runtime
- Internet available for research, not for Kaggle execution

## Data Source
- Primary: /kaggle/input/arc-prize-2026-arc-agi-2/ (challenge, no test outputs)
- Dev fallback: GitHub arcprize/ARC-AGI-2 (1000 train + 120 eval tasks)
- Format: {"train": [{"input": [[..]], "output": [[..]]}], "test": [{"input": [[..]]}]}
- Grid: integers 0-9, 1x1 to 30x30, task IDs 8-hex strings

## Architecture: 3-Engine Portfolio
1. Symbolic DSL Search (pure Python, beam search, ~30 primitives)
2. Object-Centric MDL Parser (multi-parse, MDL ranking)
3. Heuristic Neural Proposal (pattern-based fallback for no-torch)
+ Exact Verification (all train pairs must match)
- attempt_1: shortest verified program (lowest MDL)
- attempt_2: alternative from different engine or variant

## Implementation Modules
substrate.py, diagnostics.py, representations.py, primitives.py,
grammar.py, search.py, verification.py, adaptation.py, portfolio.py,
submission.py, main.py

## Phased Execution
A: Substrate (grid ops, connected components, symmetry)
B: Diagnostics (7-dim feature vector per train pair)
C: Representations (6 views in parallel)
D: Primitives (6 groups: geometric, color, spatial, selection, arithmetic, composition)
E: Search (9 strategies benchmarked on dev set)
F: Verification (exact pixel match on all train pairs)
G: Adaptation (schemes B, D, F: reweighting, parse mutation, self-consistency)
H: Portfolio (3 engines -> ranked candidates -> 2 attempts)
I: Submission (schema validation + re-parse)

## Risk Mitigation
- No numpy: pure Python with array module
- 12h budget: 30s/task hard limit, ~100 tasks
- No torch: heuristic fallback for neural engine
- Overfitting: verify ALL train pairs, cross-validate, no test labels

## Key Decisions
1. Pure Python only
2. 3-engine portfolio (research best: 37.5% Pass@1, 45% Pass@2)
3. Multi-parse segmentation
4. MDL ranking
5. Adaptation: B, D, F only
6. No synthetic data (no benefit without torch)
7. Interventions: dev validation only

## Validation
- Benchmark on 100 evaluation tasks locally
- Cross-validate adaptation (hold out 1 demo per task)
- Verify no data leakage

## Artifacts (to /kaggle/working/)
research_register.md, assumptions_and_priors.md, tool_inventory.md,
experiment_log.json, failure_taxonomy.md, architecture_decision.md,
final_validation_report.md, submission.json
