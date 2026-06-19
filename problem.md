# Post-Storm Subsea Cable Burial Change Audit

## Overview

Subsea power and communication cables are surveyed repeatedly after storms, fishing activity, and anchor-drag incidents. Operators do not only need to identify objects in a single sonar frame; they must compare a baseline survey with a later post-storm survey, decide what changed, and prioritize the correct maintenance response.

Your task is to classify structured audit items from paired synthetic side-scan sonar mosaics. Each image contains two aligned panels:

- left panel: baseline cable-corridor survey
- right panel: post-storm repeat-pass survey

Each row asks for one audit decision about the paired image. The correct label is one of `A`, `B`, `C`, or `D`. Solving the task requires comparing the two panels, using the scale metadata, and separating true cable exposure changes from sonar speckle, seabed ripples, shadows, scars, and crossing clutter.

This is a paired change-detection and intervention-priority challenge, not a generic sonar captioning or open-ended VQA task. The private split includes scale shifts, low-gain repeat passes, diagonal cable corridors, clutter decoys, and partial sensor dropout.

## Evaluation

Submissions are scored using robust grouped accuracy:

`score = 0.45 * overall_accuracy + 0.20 * worst_accuracy_by_question_type + 0.10 * worst_accuracy_by_difficulty + 0.08 * worst_accuracy_by_visibility + 0.07 * worst_accuracy_by_seabed_texture + 0.07 * worst_accuracy_by_ood_axis + 0.03 * worst_accuracy_by_length_scale`

Scores are maximized. The minimum fallback score is `1e-9` and the maximum score is `1.0`. The grouped terms penalize methods that work only on easy, clear, in-distribution repeat passes or ignore one audit decision type.

## Dataset

The prepared public dataset has the following files:

- `train.csv`: labeled training rows. Each row is one audit item about one paired sonar mosaic and includes `answer_label`.
- `test.csv`: unlabeled test rows. It has the same feature columns as `train.csv`, but does not include `answer_label`.
- `sample_submission.csv`: an example CSV showing the exact submission columns.
- `images/`: PNG paired sonar mosaics referenced by the `image_path` column.

### Columns

The public CSV columns are:

- `question_id` (string): opaque row identifier. Use this in `submission.csv`.
- `scene_id` (string): opaque paired-survey scene identifier. Several audit rows may reference the same mosaic.
- `image_path` (string): relative path to the paired PNG mosaic under the public dataset directory.
- `question_type` (string): audit decision being tested.
- `difficulty` (string): rendering difficulty, one of `easy`, `medium`, or `hard`.
- `visibility` (string): post-storm sonar visibility condition such as clear, speckled, low gain, multipath, or dropout.
- `seabed_texture` (string): dominant seabed style such as flat, rippled, rocky, or vegetated.
- `range_setting_m` (float): approximate displayed transect length in meters for each panel.
- `meters_per_pixel` (float): scale factor for converting pixel length to physical length.
- `question` (string): audit prompt.
- `choice_a` (string): answer choice for label `A`.
- `choice_b` (string): answer choice for label `B`.
- `choice_c` (string): answer choice for label `C`.
- `choice_d` (string): answer choice for label `D`.
- `answer_label` (string): correct answer label. This column appears only in `train.csv`.

### Audit Item Types

The task includes these six audit item types:

- `exposure_delta_bin`: compare both panels and estimate the new exposed-cable length category: stable, minor under 18 m, moderate 18-45 m, or severe over 45 m.
- `new_risk_sector`: identify whether the left, middle, or right third of the corridor has the largest post-storm increase in exposure risk, or whether no meaningful increase is visible.
- `new_scar_count`: count new anchor-drag scars touching the cable in the post-storm panel: 0, 1, 2, or 3 or more.
- `new_crossing_threat`: identify the newly appearing crossing threat: none, rigid pipeline, rope/net loop, or debris pile.
- `burial_compliance`: classify post-storm burial status as within specification, marginal, non-compliant, or indeterminate/resurvey.
- `intervention_priority`: choose the operational response: no action, monitor, targeted inspection, or immediate reburial work order.

## Modeling Considerations

Strong solutions should compare corresponding regions across the left and right panels rather than treating the image as a single static sonar scene. The `meters_per_pixel` metadata is important for length-change and compliance items because the same pixel-length exposure can represent different physical distances.

Useful approaches include panel splitting, local contrast normalization, line/shadow features, change maps between baseline and post-storm panels, and validation grouped by `scene_id`. Several rows can reference the same paired image, so random row splits leak scene-level information.

Do not use external sonar datasets, pretrained models, internet access, private answer labels, or generator trace columns.

## Submission

Submit a CSV file named `submission.csv` with exactly these columns:

- `question_id`: row identifier copied exactly from `test.csv`.
- `answer_label`: predicted answer label. Must be one of `A`, `B`, `C`, or `D`.

Example `submission.csv`:

```csv
question_id,answer_label
q_13a8c4e51b091a,B
q_6c51d3aab0d802,D
q_785fdd19d34162,A
```

Requirements:

- Must contain exactly one row for every row in `test.csv`.
- Must include a header row.
- Must not include duplicate or missing `question_id` values.
- `answer_label` must be one of `A`, `B`, `C`, or `D`.
