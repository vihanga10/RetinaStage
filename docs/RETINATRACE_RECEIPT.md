# RetinaTrace verifiable prediction receipt

## Purpose

RetinaTrace creates a compact JSON record of one RetinaStage result. It allows
a saved result to be checked for later modification without storing the retinal
photograph inside the receipt. This is an educational software-integrity and
traceability feature; it does not validate the diagnosis or replace qualified
human review.

## Receipt contents

Each receipt records:

- a UTC creation timestamp and receipt schema version;
- the existing SHA-256 hashes of the input bytes and selected model artifact;
- the selected grade, label, calibrated confidence and uncertainty state;
- all five ordered calibrated probabilities;
- the six technical-quality measurements, quality flags and interpretation;
- the frozen temperature and confidence threshold;
- the human-review decision and reasons;
- optional Grad-CAM target, layer, input-space label and limitation text; and
- the educational-use notice and receipt privacy declarations.

The receipt deliberately excludes the original image, original filename,
Grad-CAM heatmap bytes and Grad-CAM overlay bytes. The input hash is retained
for traceability, but a hash is not a replacement copy of the source image.

## Integrity method

Before hashing, the receipt is serialized as canonical UTF-8 JSON with sorted
keys and compact separators. The `integrity.receipt_sha256` field is excluded
from its own digest scope. Verification removes that field, repeats the same
canonical serialization and compares the newly calculated SHA-256 value with
the stored value using a timing-safe comparison.

An unchanged receipt verifies successfully. Changing a prediction field,
probability, review reason or other checksummed content causes verification to
fail.

This checksum is **not a digital signature**. Anyone able to construct a new
receipt can also calculate a new checksum. The current implementation therefore
detects accidental or unsophisticated modification when the stored checksum is
not recomputed, but it does not authenticate an issuer, establish trusted time,
or protect against deliberate checksum replacement. A future deployment could
add a server-held signing key and public-key verification as a separate,
explicitly versioned feature.

## API and browser flow

1. The normal `/api/v1/predict` and optional `/api/v1/explain` calls finish.
2. The browser removes Grad-CAM image data URLs and sends the prediction plus
   sanitized explanation metadata to `/api/v1/retinatrace/receipt`.
3. The API validates cross-field consistency and returns a checksummed receipt.
4. The browser can download the verification JSON using a name derived from
   the receipt hash, not the original image filename.
5. The browser can also render an A4 human-readable receipt and open the native
   print dialog. On macOS, the dialog can save this presentation as PDF.
6. A user can select a saved JSON receipt. The browser applies format and size
   checks, then sends it to `/api/v1/retinatrace/verify`.
7. The interface reports whether the saved content matches its stored checksum.

The verification JSON and printable receipt serve different purposes. The JSON
preserves the exact structured fields required for checksum verification. The
printable version presents those fields for human review and coursework
demonstration; a PDF exported from the print dialog is not accepted as input to
the verification endpoint.

Receipt creation and verification do not load the TensorFlow model, perform
another prediction or store an uploaded retinal image.

## Fixed evaluation

Run from the repository root:

```bash
PYTHONPATH=src:. python scripts/evaluate_retinatrace.py
```

The fixed evaluation uses one controlled prediction object and controlled
explanation metadata. It performs nine checks:

1. an unchanged receipt verifies;
2. a changed predicted grade is detected;
3. a changed class probability is detected;
4. a changed review reason is detected;
5. all five ordered class probabilities are preserved;
6. the input and model hashes are preserved;
7. image and Grad-CAM image bytes are absent;
8. the original filename is absent; and
9. repeated serialization is byte-stable.

The current tracked run passed 9 of 9 checks. The three controlled modification
cases were all detected, giving a tamper-detection rate of 1.0000 for this fixed
matrix. All defined privacy checks also passed. Row-level results are stored in
`results/retinatrace_evaluation/retinatrace_evaluation_results.csv`, with the
aggregate summary in `retinatrace_evaluation_summary.json`.

The evaluation does not open retinal images, run model inference, use the
calibration or final-test split, or claim use of a cryptographic signature. Its
results describe the receipt implementation only and must not be interpreted as
additional evidence of model accuracy, clinical validity or security against a
malicious party who can replace both content and checksum.
