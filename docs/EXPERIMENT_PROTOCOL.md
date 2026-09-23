# RetinaStage experiment protocol

## Data roles

The frozen manifest defines four non-overlapping roles:

| Split | Records | Permitted use |
|---|---:|---|
| Train | 2,267 | Optimise model parameters |
| Validation | 523 | Select architecture, loss, epoch, and checkpoint |
| Calibration | 174 | Fit temperature and the uncertainty threshold |
| Test | 523 | One final evaluation after all choices are frozen |

The test labels must not influence preprocessing, model selection,
hyperparameters, calibration, or the uncertainty threshold.

## Experiment sequence

1. Train the frozen EfficientNetB0 baseline with balanced class weights.
2. Fine-tune the final 40 backbone candidates while freezing batch
   normalisation layers.
3. Repeat fine-tuning with the hybrid ordinal loss.
4. Compare checkpoints using validation metrics, prioritising macro F1 and
   reporting quadratic weighted kappa and grade MAE.
5. Freeze the selected checkpoint.
6. Fit temperature scaling and the confidence threshold on calibration data.
7. Evaluate the test split once and write the completion marker.
8. Perform Grad-CAM and quantitative error analysis without changing the
   selected model or policy.

## Selection rationale

Accuracy alone is inappropriate for this imbalanced five-grade task. The
hybrid ordinal checkpoint selected by validation macro F1 improved macro F1
and QWK over the frozen baseline, while the cross-entropy fine-tuned model had
the highest validation accuracy and lowest grade MAE. The chosen rule was
fixed before test evaluation.

## Reproduction commands

Run from the repository root after installing `requirements.txt` and placing
the APTOS images in `data/raw/train_images/`:

```bash
python scripts/train_baseline.py
python scripts/train_finetuned.py
python scripts/train_ordinal.py
python scripts/evaluate_checkpoint.py \
  artifacts/training/ordinal_stage2/best_macro_f1_model.keras \
  --output-directory artifacts/evaluation/ordinal_validation
python scripts/calibrate_model.py
```

The following command is documented for a fresh reproduction only. It must
not be rerun to tune the completed coursework experiment:

```bash
python scripts/evaluate_final_test.py
```

After the fixed test evaluation:

```bash
python scripts/generate_gradcam.py
python scripts/analyze_test_errors.py
```

## Interpretation limits

- APTOS labels are dataset references, not independently adjudicated clinical
  ground truth.
- Patient-level separation cannot be verified because patient identifiers are
  unavailable.
- Severe and proliferative DR have low support, so their estimates are less
  stable than the No DR result.
- Grad-CAM is a qualitative attention visualisation and not validated lesion
  localisation.
- The uncertainty policy is an experimental abstention mechanism, not a
  clinical referral rule.
- The system is an educational prototype and is not clinically validated.
