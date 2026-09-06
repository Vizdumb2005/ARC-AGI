# ARC-AGI-2 Research Guidance for the Coding Agent

Target repository: `Vizdumb2005/ARC-AGI`
Target branch: `kilo/jaunty-tiger-lbd`

## Repository inspection findings

The branch already contains a substantial pure-Python solver core:

- `solver/src/substrate.py` — grid operations, diagnostics helpers, components, symmetry, translation, traces.
- `solver/src/diagnostics.py` — seven-dimension input/output diagnostic signatures.
- `solver/src/representations.py` — raw pixels, connected components, color masks, bounding boxes, relational graph, and multi-parse forest.
- `solver/src/primitives.py` — registered deterministic primitives and MDL costs.
- `solver/src/inference.py` — direct inference and all-training-pair verification.
- `solver/src/search.py` — single primitive, composition, object-centric, and pattern search.

The tree currently does not contain the plan-listed `main.py`, `submission.py`, `verification.py`, `adaptation.py`, `portfolio.py`, or `grammar.py`. It also contains tracked `__pycache__` files, which should not be expanded and should eventually be removed from version control.

## Highest-priority implementation work

### P0: Kaggle execution entrypoint

Add a small, deterministic entrypoint that:

1. Discovers JSON files under `/kaggle/input`.
2. Selects only `arc-agi_test_challenges.json` (or the closest `*test*challenges*.json` fallback).
3. Never merges training, evaluation, or solution files into the submission task set.
4. Uses the visible test placeholder for local development only; Kaggle swaps it for 240 unseen rerun tasks.
5. Runs the existing search/inference engines per task.
6. Writes exactly `/kaggle/working/submission.json`.
7. Preserves task-ID and test-input order.
8. Creates exactly one object per test input with `attempt_1` and `attempt_2`.
9. Reopens and validates the output JSON.

The competition Data page says the rerun contains 240 unseen tasks, usually one test input per task and a small number with two. Do not hard-code 100, 120, or 400 as the Kaggle submission count.

### P0: Exact validation and portfolio output

Add a dedicated validator that checks:

- Exact task-ID equality.
- Correct number of prediction objects per task.
- Both attempt keys on every object.
- Non-empty rectangular grids.
- Dimensions no larger than 30×30.
- Integer cells in `0..9`.
- JSON reparse after writing.

Collect verified candidates from the existing engines. `attempt_1` should be the lowest-MDL candidate that fits every training pair. `attempt_2` should be a distinct verified candidate from another representation/engine when available. Never use a random second guess and never use hidden test labels.

### P0: Import/runtime cleanup

The current modules use top-level imports such as `from substrate import *` and demo harness paths such as `/tmp/kilo/arc_data/...`. Make the Kaggle entrypoint establish a stable `solver/src` import path or convert imports to package-safe imports. Replace hard-coded demo paths with CLI/configurable paths. Keep the runtime standard-library-only unless an installed dependency is explicitly discovered and tested.

## Research principles to preserve

- Raw pixels, color masks, 4-connected components, 8-connected components, bounding boxes, and relational graphs remain competing parses; none is ground truth.
- Exact verification is mandatory: reject any program that fails one available training output.
- Use diagnostic signatures to route search, but benchmark against an unguided baseline.
- Use MDL/complexity as a tie-breaker, not as proof that the shortest explanation is correct.
- Adaptation may reweight primitives, mutate parses, increase search depth, use self-consistency, or repair localized errors, but it must use demonstrations only and be evaluated with leave-one-demonstration-out validation.
- Synthetic data, LPN/neural proposals, RE-ARC, and BARC are optional experiments, not default dependencies.

## Evidence boundaries

The uploaded strategy PDF contains useful architecture hypotheses, but its numerical tables (search accuracy, intervention retention, synthetic-data gains, memory, and runtime) were supplied without reproducible logs or code. Treat them as unverified and do not put their percentages in reports or comments as measured results.

The official ARC-AGI-2 structure is 1,000 public training tasks and 120 public evaluation tasks, plus 120 semi-private and 120 private evaluation tasks. H-ARC’s 1,729-person study and 400+400 task counts concern original ARC, not ARC-AGI-2. See `research/UPLOADED_CLAIM_AUDIT.md`.

## Recommended validation sequence

1. Run substrate and representation self-tests.
2. Load the visible placeholder test challenges and matching evaluation solutions separately.
3. Produce reconnaissance: task count, train/test counts, dimensions, colors, size changes, and parse statistics.
4. Run single-primitive and bounded-composition search.
5. Run leave-one-demonstration-out evaluation.
6. Add portfolio ranking and exact submission validation.
7. Measure runtime and memory on the visible data.
8. Only then consider broader primitives, parse mutation, or optional neural proposals.

Do not claim a competition score until the notebook has run with the actual Kaggle rerun file. Do not click or automate Kaggle submission from code.
