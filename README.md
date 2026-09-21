# RetinaStage

Computer Vision coursework: diabetic retinopathy detection and
five-grade classification using transfer learning.

## Coursework
- Course: BSc (Hons) Computer Science
- Module: Computer Vision
- Batch: BSCCOMP24.2P

## Current status
Dataset preparation is in progress. No model has been trained yet.

## Dataset
APTOS 2019 Blindness Detection:
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Original images and labels are stored locally in data/raw/.
Dataset files are excluded from this repository.

## Initial filename audit
The following results were obtained by running the filename check locally:

| Check | Result |
|---|---:|
| CSV records | 3662 |
| Image files | 3662 |
| Missing images | 0 |
| Unlisted images | 0 |
| Repeated image identifiers | 0 |

This checks filename correspondence only. Image readability,
duplicate image content and image quality are not yet verified.

## Planned implementation
- Reproducible dataset auditing and split creation
- Image preprocessing and training augmentation
- CNN transfer learning for five-grade classification
- Evaluation and error analysis
- Ordinal loss experiment
- Image quality checks, confidence calibration and Grad-CAM
- Prototype demonstration

## Running the project
Execution instructions and dependencies will be added with the code.

## Results and deliverables
Training results, Colab notebooks, the final report and demonstration
video will be linked here when available.

## Intended use
Educational research prototype; not validated for clinical use.

See PROGRESS.md for the development checklist.

## Image integrity audit

All 3,662 labelled files were decoded successfully using Pillow.

| Audit check | Result |
|---|---:|
| Successfully decoded images | 3,662 |
| Unreadable images | 0 |
| RGB images | 3,662 |
| Distinct image dimensions | 17 |
| Identical-image groups | 123 |
| Images in identical-image groups | 251 |
| Redundant copies | 128 |
| Conflicting-label duplicate groups | 30 |
| Images in conflicting-label groups | 62 |

The audit identified identical retinal images stored under different
identifiers. Thirty duplicate groups contain conflicting diagnosis labels.

Planned handling:

- Preserve the complete original dataset in `data/raw`.
- Exclude all conflicting-label groups from supervised modelling.
- Retain one deterministic representative from each same-label duplicate group.
- Record every inclusion and exclusion decision in a preparation manifest.
- Create dataset splits only after duplicate handling, preventing identical
  images from crossing training, validation and test partitions.

The audit does not establish that the remaining medical labels are clinically
correct. It identifies technical duplication and direct conflicts in the
provided labels.

## Modelling-record preparation

The original dataset remains unchanged. A preparation manifest records the
decision for every original image.

| Decision | Records |
|---|---:|
| Included for modelling | 3,504 |
| Excluded redundant same-label copies | 96 |
| Excluded conflicting-label duplicates | 62 |
| Original records | 3,662 |

Eligible class distribution:

| Grade | Class | Eligible records |
|---|---|---:|
| 0 | No DR | 1,796 |
| 1 | Mild | 338 |
| 2 | Moderate | 922 |
| 3 | Severe | 177 |
| 4 | Proliferative DR | 271 |

The manifest retains one deterministic representative from each identical
same-label group and excludes every member of an identical-image group when
its labels conflict. Raw files are preserved and are never overwritten.
