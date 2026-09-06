# Implementation Plan: ARC-AGI-2 Solver

## Environment
- Python 3.10.12, standard library only (no numpy/torch/scipy)
- pip unavailable
- /kaggle/input and /kaggle/working will exist at runtime
- Internet available for research, not for Kaggle execution

## Data Source
- Primary: /kaggle/input/arc-prize-2026-arc-agi-2/ (challenge, no test outputs)
- Dev fallback: GitHub arcprize/ARC-AGI-2 for provenance and local development; do not conflate its splits with Kaggle submission packaging
- Kaggle package: six JSON files; the visible test file is a development placeholder and is replaced by 240 unseen rerun tasks
- Submission schema: top-level task IDs; ordered prediction objects with `attempt_1` and `attempt_2`; exact grid equality
- Format: {"train": [{"input": [[..]], "output": [[..]]}], "test": [{"input": [[..]]}]}
- Grid: integers 0-9, 1x1 to 30x30, task IDs 8-hex strings

## Architecture: Verified Multi-Route Portfolio
1. Symbolic DSL Search (pure Python, bounded search over registered primitives)
2. Object-Centric MDL Parser (multi-parse, MDL ranking)
3. Pattern/diagnostic proposal route (must remain deterministic and optional)
+ Exact Verification (all train pairs must match); neural proposals are not a runtime dependency
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
- 12h budget: 30s/task hard limit; engineer for the 240-task Kaggle rerun package
- No torch: heuristic fallback for neural engine
- Overfitting: verify ALL train pairs, cross-validate, no test labels

## Key Decisions
1. Pure Python only
2. Verified multi-route portfolio; no benchmark percentage is assumed until reproduced on a declared split
3. Multi-parse segmentation
4. MDL ranking
5. Adaptation: B, D, F only
6. No synthetic data (no benefit without torch)
7. Interventions: dev validation only

## Validation
- Benchmark the visible placeholder against evaluation solutions without using those labels in test inference
- Cross-validate adaptation with one held-out demonstration per task
- Validate exact submission schema and re-parse `submission.json`
- Do not claim a competition score until the actual rerun file has been used

## Artifacts (to /kaggle/working/)
research_register.md, assumptions_and_priors.md, tool_inventory.md,
experiment_log.json, failure_taxonomy.md, architecture_decision.md,
final_validation_report.md, submission.json

## Branch inspection corrections

The current branch contains `substrate.py`, `diagnostics.py`, `representations.py`, `primitives.py`, `inference.py`, `search.py`, plus the new hybrid `verification.py`, `vision_proposer.py`, `portfolio.py`, `submission.py`, and `main.py`. `grammar.py` and `adaptation.py` remain optional future modules; they should be added only after baseline validation.

Before Kaggle execution, remove hard-coded `/tmp/kilo/arc_data` demo paths, establish stable imports, add a data-file discovery/selection layer, and keep solution files isolated from test inference. The tracked `__pycache__` files should not be expanded and should eventually be removed from version control.

See `docs/arc_agi2/RESEARCH_GUIDANCE.md`, `docs/arc_agi2/EXPERIMENT_MATRIX.md`, and `docs/arc_agi2/CLAIM_AUDIT.md`.


## Hybrid model route

The optional offline vision route is proposal-only: it emits structured candidates, then exact verification rejects any candidate that fails a training pair. The symbolic route remains the default fallback. No remote API calls or mandatory ML dependencies are allowed.
