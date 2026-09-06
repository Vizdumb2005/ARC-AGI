# Uploaded ARC Strategy Claim Audit

Audit date: 2026-09-06

## Official Kaggle reconnaissance

The attached Kaggle Data page lists six files:

- `arc-agi_training_challenges.json`
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json`
- `arc-agi_evaluation_solutions.json`
- `arc-agi_test_challenges.json`
- `sample_submission.json`

Kaggle states that the visible `arc-agi_test_challenges.json` is a development placeholder based on evaluation tasks. When the notebook is rerun for scoring, it is swapped for the actual test challenges. The rerun contains **240 unseen tasks**; most have one test input and a small number have two. The eventual repository entrypoint must load only the selected test-challenge file.

The Data page also confirms the task schema: `train` contains input/output demonstrations, `test` contains input grids, cells are integers 0–9, and correctness requires exact output grids with two trials per test input.

## Claim-by-claim status

| Uploaded claim | Status | Evidence / correction |
|---|---|---|
| ARC-AGI-2 uses few-shot input/output grid tasks and exact pixel matching | Verified | Kaggle Data page; official guide: https://arcprize.org/guide/1 |
| Grids use integer symbols 0–9 and range up to 30×30 | Verified | Official repository README: https://github.com/arcprize/ARC-AGI-2 |
| ARC-AGI-2 has 1,000 public training tasks and 120 public evaluation tasks | Verified | Official ARC-AGI-2 page: https://arcprize.org/arc-agi/2 |
| ARC-AGI-2 also has 120 semi-private and 120 private evaluation tasks | Verified | Official ARC-AGI-2 page and guide: https://arcprize.org/arc-agi/2, https://arcprize.org/guide/1 |
| Kaggle’s rerun leaderboard file contains 240 unseen tasks | Verified | Attached Kaggle Data page; this is the competition packaging fact used by the notebook |
| Every ARC-AGI-2 evaluation task was solved pass@2 by at least two humans | Verified as an official benchmark-design claim | https://arcprize.org/arc-agi/2 and https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025 |
| H-ARC evaluated 1,729 people on 400 original ARC training + 400 original ARC evaluation tasks | Verified, but scope is original ARC, not ARC-AGI-2 | https://arxiv.org/abs/2409.01374 |
| H-ARC reported 76.2% average on original ARC training and 64.2% on original ARC evaluation | Verified, but scope is original ARC | https://arxiv.org/abs/2409.01374 |
| 98.8% of original ARC tasks were solved by at least one H-ARC participant within three attempts | Verified, but this is 790/800 original ARC tasks and not an ARC-AGI-2 statistic | https://arxiv.org/abs/2409.01374 |
| ARC-AGI-2 public evaluation contains 400 tasks | Contradicted / scope confusion | ARC-AGI-2 official public evaluation is 120 tasks. 400/400 is associated with original ARC/H-ARC or the 400-task public training component in later reports. |
| ARC-AGI-2 search strategies achieved the uploaded table’s exact pass@1/pass@2 percentages, runtimes, memory, and hypothesis counts | Unverified | No raw logs, code, split manifest, or primary source for the table were supplied. These values are not used. |
| The uploaded hybrid portfolio achieved the stated 37.5%/45.0% results | Unverified | No reproducible experiment attached; not treated as a result. |
| Synthetic configurations achieved the stated local/private accuracies and KL shifts | Unverified | No generator logs, seeds, distribution definitions, or raw results attached. RE-ARC supports generation for original ARC training tasks, not these claimed ARC-AGI-2 transfer numbers: https://github.com/michaelhodel/re-arc |
| Intervention retention percentages in the uploaded PDF are established | Unverified | The intervention ideas are useful, but the percentages require rerunning on a declared split. |
| Ferré demonstrates object-centric + MDL reasoning for ARC | Supported in general, scope-limited | The paper presents object-centric models and MDL for ARC, but the uploaded “96/400” and multi-parse superiority claims were not established from the abstract alone: https://arxiv.org/abs/2311.00545 |
| LPN provides a test-time latent program search idea | Supported in general, scope-limited | LPN reports ARC experiments and improved OOD performance with test-time search, but this does not establish offline ARC-AGI-2 performance or Kaggle availability: https://arxiv.org/abs/2411.08706 |
| ConceptSearch directly validates ARC-AGI-2 performance | Not supported | The repository/paper concerns ARC program search and does not report explicit ARC-AGI-2 results: https://github.com/kksinghal/concept-search |
| CodeIt’s 15% result is an ARC-AGI-2 result | Contradicted by scope | The cited paper reports on the original ARC evaluation dataset, not explicitly ARC-AGI-2: https://arxiv.org/abs/2402.04858 |
| RE-ARC proves synthetic pretraining transfers to ARC-AGI-2 | Not supported | RE-ARC provides verified procedural examples for the original 400 ARC training tasks; transfer to ARC-AGI-2 must be measured separately: https://github.com/michaelhodel/re-arc |
| Submission schema is `{"outputs": [...]}` | Contradicted | The attached Kaggle page specifies task IDs as top-level keys, each mapping to ordered objects with `attempt_1` and `attempt_2`. |
| A final run processed 100 tasks, solved 38/45, and took 3h14m | Unverified and inconsistent with current Kaggle packaging | The attached Kaggle page states 240 unseen rerun tasks. The uploaded execution summary has no reproducible run artifact. |

## Implementation consequence

Only verified protocol facts and reproducible local behavior are promoted into the solver. The PDF’s architecture ideas are retained as hypotheses and experiment slots. Its numerical tables are not used for ranking, claims of superiority, or final reporting.
