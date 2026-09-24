# RetinaStage

Computer Vision coursework: diabetic retinopathy detection and
five-grade classification using transfer learning.

## Coursework
- Course: BSc (Hons) Computer Science
- Module: Computer Vision
- Batch: BSCCOMP24.2P

## Current status

The end-to-end experimental workflow is complete: auditable data preparation,
frozen splits, EfficientNetB0 transfer learning, ordinary and ordinal
fine-tuning, validation-based checkpoint selection, calibration, a locked
one-time test evaluation, Grad-CAM, quantitative error analysis, and
validation-only robustness analysis. The prototype application API wraps the
frozen selected model, calibration policy and technical-quality checks. A
React/TypeScript interface now provides image upload, calibrated probability
display, quality warnings and auditable review information.

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

This filename check was followed by the integrity, duplicate, and quality
audits documented below.

## Implemented workflow

- Reproducible image auditing, duplicate handling, and stratified splits
- Retinal-field cropping, aspect-ratio-preserving padding, and augmentation
- EfficientNetB0 frozen-backbone baseline and selective fine-tuning
- Hybrid classification/ordinal loss for distance-aware grade prediction
- Accuracy, macro/weighted F1, per-class recall, QWK, and grade MAE
- Temperature scaling and a calibration-only uncertainty threshold
- Locked one-time final-test evaluation
- Grad-CAM examples and saved-prediction error analysis

## Running the project

Use Python 3.12 in a virtual environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the original APTOS images in `data/raw/train_images/`. The split
manifest is tracked, but source images, checkpoints, and generated artifacts
are intentionally excluded from Git.

```bash
python scripts/train_baseline.py
python scripts/train_finetuned.py
python scripts/train_ordinal.py

python scripts/evaluate_checkpoint.py \
  artifacts/training/ordinal_stage2/best_macro_f1_model.keras \
  --output-directory artifacts/evaluation/ordinal_validation

python scripts/calibrate_model.py
python scripts/evaluate_final_test.py
python scripts/generate_gradcam.py
python scripts/analyze_test_errors.py
python scripts/evaluate_robustness.py
```

`evaluate_final_test.py` creates a completion marker and refuses a second run.
This protects the test split from repeated inspection and tuning. The current
coursework test has already been completed, so do not rerun it for further
model decisions. See `docs/EXPERIMENT_PROTOCOL.md` for the full separation of
train, validation, calibration, and test roles. Run the automated numerical
checks with:

```bash
python -m unittest discover -s tests -v
```

The robustness command uses validation images only. It applies fixed,
deterministic brightness, contrast, blur, noise, and JPEG perturbations to the
selected model without retraining or changing the calibration policy.

## Running the application API

Restore the selected model and calibration summary to the default artifact
locations documented in `docs/APPLICATION_ARCHITECTURE.md`, then run:

```bash
python -m pip install -r requirements-app.txt
PYTHONPATH=src uvicorn app.api.main:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`. The
prediction endpoint applies the frozen preprocessing, temperature scaling,
confidence threshold and dataset-derived technical-quality review limits. It
does not retrain the model or modify any experimental parameters.

## Running the web interface

With the API running, install and start the Vite application in a second
terminal:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Set `VITE_API_BASE_URL` in `frontend/.env.local` if the API uses a different
port, such as `http://127.0.0.1:8001`. Open `http://127.0.0.1:5173` to use the
interface. It presents the five calibrated probabilities, technical-quality
flags, confidence policy, human-review decision and traceability hashes without
storing the uploaded image. Frontend checks are available through:

```bash
npm test
npm run build
```

## Results and deliverables

| Validation experiment | Accuracy | Macro F1 | QWK | Grade MAE |
|---|---:|---:|---:|---:|
| Frozen baseline | 0.7533 | 0.5749 | 0.8402 | 0.3231 |
| Cross-entropy fine-tuning | **0.7935** | 0.5652 | 0.8406 | **0.2830** |
| Hybrid ordinal, selected | 0.7839 | **0.5891** | **0.8430** | 0.2945 |

The hybrid ordinal checkpoint was selected by validation macro F1, before the
test split was opened. On the 523-record locked test split it obtained 0.7667
accuracy, 0.5833 macro F1, 0.8348 quadratic weighted kappa, and 0.3212 grade
MAE. A total of 92.73% of predictions were within one grade of the reference.

After temperature scaling, test NLL changed from 0.6085 to 0.5995 and ECE from
0.0595 to 0.0564. The fixed uncertainty policy covered 77.25% of test records
at 83.42% selective accuracy. See
`results/model_evaluation/experiment_summary.json` for the complete tracked
summary and explicit limitations.

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

Three configurations were defined for controlled comparison:

| ID | Configuration |
|---|---|
| P0 | Retinal crop, square padding and resizing |
| P1 | P0 plus CLAHE |
| P2 | P1 plus mild unsharp masking |

CLAHE improved visibility in darker examples but also amplified texture and
changed image appearance. Unsharp masking provided limited additional visual
benefit and sometimes strengthened noise. The final modelling pipeline used
the conservative P0 configuration, with all selection decisions restricted to
the validation split.
