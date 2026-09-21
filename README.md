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
