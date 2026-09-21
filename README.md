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

## Image-quality audit

Technical quality measurements were calculated for all 3,504 eligible images
using a standardised 512-pixel representation.

| Quality result | Count |
|---|---:|
| Successfully measured | 3,504 |
| Images with one or more review flags | 449 |
| Low-brightness flags | 88 |
| High-brightness flags | 88 |
| Low-contrast flags | 88 |
| Low-sharpness flags | 176 |
| Low-retinal-coverage flags | 88 |

The limits are dataset-derived percentiles for manual review. They are not
clinical gradability thresholds. Visual review found exposure, contrast,
sharpness, colour and framing variation, but did not justify automatic
exclusion of the flagged records.

The lowest retinal-coverage examples were predominantly Grade 0 images with
smaller retinal circles and larger black borders. This suggests a possible
acquisition-related shortcut. Retinal-field cropping, aspect-ratio-preserving
padding and resizing will therefore be evaluated against ordinary full-image
resizing.

Quality measurements will also support input warnings, robustness analysis
and error analysis after model training.

## Perceptual near-duplicate review

A 256-bit perceptual-hash screen was applied after retinal-field cropping to
the 3,504 records retained by the exact-duplicate audit.

| Result | Count |
|---|---:|
| Candidate pairs | 11 |
| Unique candidate images | 19 |
| Connected near-duplicate groups | 9 |
| Same-label redundant images excluded | 2 |
| Conflicting-label near-duplicates excluded | 15 |
| Final eligible modelling records | 3,487 |

All candidate pairs were inspected side by side. Confirmed pairs shared the
same retinal structure and differed through small image transformations or
processing differences. Connected pairs were handled as groups.

Every member of a conflicting-label group was excluded. For a same-label
group, one deterministic representative was retained. No original file or
source label was changed.

## Frozen dataset splits

The 3,487 eligible records were stratified by disease grade using the fixed
random seed `20260921`.

| Split | Records | Purpose |
|---|---:|---|
| Training | 2,267 | Model parameter learning |
| Validation | 523 | Model and hyperparameter selection |
| Calibration | 174 | Confidence calibration |
| Test | 523 | Final evaluation after all choices are frozen |

The split manifest SHA-256 fingerprint is:

`b2024cba2260e0d3470a37f744f341db3a45451ffe2490ae351dbaf6da5ee211`

Repeated execution produced the same fingerprint and class counts. The test
partition must remain unused until preprocessing, architecture, loss and
threshold choices have been completed.

The source data does not provide explicit patient identifiers. Therefore,
patient-level separation cannot be confirmed, although exact and reviewed
near-duplicate images were handled before splitting.

## Preprocessing pipeline

The reusable preprocessing module implements:

1. Retinal-field estimation using thresholding and connected components.
2. Cropping with a safety margin.
3. Aspect-ratio-preserving square padding.
4. Resizing to 224 × 224 pixels.
5. Optional CLAHE on the LAB lightness channel.
6. Optional mild unsharp masking.

Visual review across all five disease grades confirmed that retinal-field
cropping preserved the visible retinal content while reducing acquisition-
related framing differences.

Three configurations will be compared experimentally:

| ID | Configuration |
|---|---|
| P0 | Retinal crop, square padding and resizing |
| P1 | P0 plus CLAHE |
| P2 | P1 plus mild unsharp masking |

CLAHE improved visibility in darker examples but also amplified texture and
changed image appearance. Unsharp masking provided limited additional visual
benefit and sometimes strengthened noise. Final selection will therefore use
validation-set evidence rather than visual preference.
