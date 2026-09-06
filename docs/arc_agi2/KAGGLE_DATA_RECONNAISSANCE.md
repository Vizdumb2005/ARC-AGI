# Kaggle ARC-AGI-2 Data Reconnaissance

Date: 2026-09-06
Source: attached Kaggle competition Data page
URL: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data

## Confirmed file inventory

The competition Data page lists six JSON files:

1. `arc-agi_training_challenges.json`
2. `arc-agi_training_solutions.json`
3. `arc-agi_evaluation_challenges.json`
4. `arc-agi_evaluation_solutions.json`
5. `arc-agi_test_challenges.json`
6. `sample_submission.json`

The displayed bundle is approximately 6.91 MB. The visible evaluation-challenges file is approximately 984.68 kB; the visible test-challenges placeholder is approximately 1.02 MB.

## Schema observed

Task objects use:

```json
{
  "task_id": {
    "train": [
      {"input": [[0, ...]], "output": [[0, ...]]}
    ],
    "test": [
      {"input": [[0, ...]]}
    ]
  }
}
```

The Data page preview showed task IDs as top-level keys, `train` and `test` lists, and integer grid cells. It states that exact output-grid matching is required and that two trials are allowed per test input.

## Critical rerun behavior

Kaggle explicitly states:

- The visible `arc-agi_test_challenges.json` is a placeholder using evaluation tasks during development.
- When the notebook is submitted for rerun, that file is swapped with actual test challenges.
- The rerun leaderboard uses **240 unseen tasks**.
- Most tasks have one test input; a small number have two test inputs.
- The notebook must predict each test output in order.

## Engineering consequences

- Development code should select only `arc-agi_test_challenges.json`.
- It must not merge evaluation, training, or solution files into the submission task set.
- The visible placeholder permits local evaluation against `arc-agi_evaluation_solutions.json`, but those labels must never be used during the real test rerun.
- The final submission must preserve every test task ID and test-input ordering.
- The eventual repository entrypoint must implement this selection behavior.

## Not yet measured

The live Data page did not expose complete task-level distributions in the browser preview. The notebook must measure, from the mounted JSON:

- Exact task count in the visible placeholder.
- Train-pair count distribution.
- Test-input count distribution.
- Input/output dimensions.
- Color usage.
- Dimension changes.
- Component and relation diagnostics.
- Exact local holdout performance.

No uploaded-document percentages were substituted for these measurements.
