# Dataset Description

## Overview

This dataset contains synthetic paired side-scan sonar mosaics for post-storm subsea cable burial audits. Each PNG image has two aligned panels: a baseline survey on the left and a post-storm repeat-pass survey on the right. The rows in `questions.csv` describe structured audit items about what changed between the two panels and what operational response is appropriate.

The dataset is designed to test paired visual change detection, scale-aware cable exposure measurement, and intervention-priority reasoning. Labels are generated from a deterministic renderer trace, but the prepared public data exposes only the paired images, audit prompts, answer choices, and limited metadata.

No public sonar dataset, proprietary survey imagery, LLM-generated labels, or external image assets are used.

## Generation Process

`generate_raw.py` creates:

- Baseline and post-storm side-scan sonar panels with matched corridor geometry.
- Seabed styles: flat, rippled, rocky, and vegetated.
- Visibility conditions: clear, speckled, low gain, multipath, and dropout.
- Post-storm changes in exposed cable length, risk sector, anchor-drag scars, and crossing hazards.
- Operational labels for burial compliance and intervention priority.
- Hidden robustness groups for OOD conditions and scale settings.

The generated raw data contains trace-derived labels. `prepare.py` then creates public/private splits, obfuscates identifiers and image filenames, lightly perturbs prepared images, and writes private scoring groups.

## File Structure

```text
raw/
  images/
    scene_0000.png
    ...
  questions.csv
  images.csv

public/
  train.csv
  test.csv
  sample_submission.csv
  images/
    s_<hashed_id>.png
    ...

private/
  answers.csv
```

## Public Files

### `train.csv`

Labeled training audit rows. Columns:

| Column | Type | Description |
|---|---:|---|
| `question_id` | string | Unique audit row identifier. |
| `scene_id` | string | Paired-survey mosaic identifier. |
| `image_path` | string | Relative path under `public/`. |
| `question_type` | string | Audit item type. |
| `difficulty` | string | Easy, medium, or hard rendering condition. |
| `visibility` | string | Post-storm sonar visibility condition. |
| `seabed_texture` | string | Dominant seabed style. |
| `range_setting_m` | float | Approximate transect length for each panel. |
| `meters_per_pixel` | float | Scale factor for physical length estimates. |
| `question` | string | Audit prompt. |
| `choice_a` | string | Choice A text. |
| `choice_b` | string | Choice B text. |
| `choice_c` | string | Choice C text. |
| `choice_d` | string | Choice D text. |
| `answer_label` | string | Correct label: A, B, C, or D. |

### `test.csv`

Unlabeled test audit rows with the same columns as `train.csv` except `answer_label`.

### `sample_submission.csv`

Required submission schema:

| Column | Type | Description |
|---|---:|---|
| `question_id` | string | Question identifier from `test.csv`. |
| `answer_label` | string | Predicted answer label: A, B, C, or D. |

## Private Files

### `private/answers.csv`

Private labels and scoring groups:

| Column | Type | Description |
|---|---:|---|
| `question_id` | string | Test question identifier. |
| `answer_label` | string | Correct answer label. |
| `question_type` | string | Audit item group for worst-group scoring. |
| `difficulty` | string | Difficulty group for worst-group scoring. |
| `visibility` | string | Visibility group for worst-group scoring. |
| `seabed_texture` | string | Texture group for worst-group scoring. |
| `ood_axis` | string | Hidden OOD axis for worst-group scoring. |
| `length_scale` | string | Hidden scale group for worst-group scoring. |

## Notes

- The primary artifact is a paired image mosaic, not a standalone single-frame object-recognition sample.
- Solvers should split each image into baseline and post-storm panels and compare corresponding corridor regions.
- `meters_per_pixel` matters for `exposure_delta_bin` and `burial_compliance`; raw pixel length is not enough.
- Multiple audit rows can reference the same paired mosaic, so validation should be grouped by `scene_id`.
