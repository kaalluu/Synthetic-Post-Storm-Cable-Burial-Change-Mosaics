# Post-Storm Subsea Cable Burial Change Audit

This is an Eris-ready paired-image change-audit challenge and dataset generator.

Agents classify structured audit items from synthetic paired side-scan sonar mosaics. Each image contains a baseline cable-corridor survey and a post-storm repeat-pass survey. The hidden renderer trace determines labels, while the public data exposes only paired mosaics, audit prompts, choices, and limited metadata.

## Why this task

The accepted examples emphasize domain-specific traps, not generic modeling:

- Notebook upvote prediction tests leakage removal, log targets, mixed feature handling, and target skew.
- Intraday liquidity forecasting tests temporal causality, regime shifts, outliers, and missingness.
- Aneurysm volume prediction tests segmentation-first reasoning, metadata-aware volume calculation, small-data augmentation, and medically appropriate preprocessing.

This task follows the same pattern in a current-friendly format:

- Primary artifact is non-tabular image data.
- Labels come from a deterministic synthetic trace, not LLM output.
- The scorer rewards robust performance across audit item type, visibility, seabed texture, scale, and hidden OOD axes.
- Rubrics are specific to repeat-pass sonar change detection, scale-aware burial compliance, and cable intervention triage.

## Files

| File | Purpose |
|---|---|
| `generate_raw.py` | Deterministically generates paired sonar mosaics and trace-derived audit rows. |
| `prepare.py` | Builds `public/` and `private/` splits from `raw/`. |
| `grade.py` | Scores submissions with robust grouped accuracy. |
| `problem.md` | Challenge prompt for solvers. |
| `dataset_description_eris_upload.md` | Dataset description for creator submission. |
| `rubrics.yaml` | Task-specific rubric criteria. |
| `config.yaml` | Eris scoring metadata. |
| `reference_solution.py` | Lightweight baseline using panel-difference features and question type. |

## Local build

```powershell
python .\generate_raw.py
python .\prepare.py
python .\reference_solution.py
```

The reference solution writes `working/submission.csv`.
