# ARC-AGI-2 Experiment Matrix

Use this matrix to turn solver changes into reproducible evidence. Record one row per run in `experiment_log.csv`; do not report unsupported values from the uploaded strategy document.

## Required baselines

| ID | Change | Data | Metrics | Acceptance rule |
|---|---|---|---|---|
| B0 | Existing solver, current limits | visible evaluation challenges + solutions | task exact-match, test-input exact-match, timeout, memory | establishes current baseline |
| B1 | Single primitives only | same | same | isolates primitive coverage |
| B2 | B1 + bounded two-step composition | same | same | improvement must survive all training-pair verification |
| B3 | B2 + object-centric route | same | same | compare per diagnostic family, not only aggregate |
| B4 | B3 + multi-parse MDL ranking | same | same | MDL only breaks verified ties |
| B5 | B4 + adaptation | same, leave-one-demo-out | held-out demo accuracy, exact-match, search cost | no use of held-out output during candidate selection |

## Per-task measurements

Record:

- task ID and split
- train-pair count and test-input count
- input/output dimensions
- background hypothesis and whether it is stable across pairs
- diagnostic signature
- selected representation and MDL cost
- candidate engine and program length
- number of evaluated candidates
- search time and timeout status
- whether all training pairs verify
- whether attempt 2 is genuinely distinct
- failure category when unsolved

## Required ablations

1. Diagnostic routing versus all-engine routing.
2. 4-connected components versus 8-connected components.
3. Fixed background `0` versus inferred/contextual background.
4. MDL ranking versus first verified candidate.
5. One attempt versus two distinct verified attempts.
6. Adaptation enabled versus disabled.

## Evaluation hygiene

- Keep evaluation solutions separate from challenge inputs and never import them from a Kaggle submission runtime.
- The visible Kaggle test file is a development placeholder; actual rerun test tasks are unseen.
- Use exact grid equality, not approximate similarity.
- Report denominators and task-level versus test-input-level metrics separately.
- Preserve the full run manifest: code revision, data-file names, limits, seed (if any), and output path.
