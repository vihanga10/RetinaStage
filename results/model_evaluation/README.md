# Model evaluation evidence

The machine-readable summary in `experiment_summary.json` records the final
metrics produced by the coursework notebook. Large checkpoints and generated
archives remain outside Git and are retained in the experiment backup.

Model selection used the 523-record validation split. Temperature and the
uncertainty threshold used the separate 174-record calibration split. The
523-record test split was evaluated once after all choices were frozen.

| Validation experiment | Accuracy | Macro F1 | QWK | Grade MAE |
|---|---:|---:|---:|---:|
| Frozen baseline | 0.7533 | 0.5749 | 0.8402 | 0.3231 |
| Cross-entropy fine-tuning | 0.7935 | 0.5652 | 0.8406 | 0.2830 |
| Hybrid ordinal, selected | 0.7839 | **0.5891** | **0.8430** | 0.2945 |

The hybrid ordinal checkpoint was selected by validation macro F1 because the
class distribution is imbalanced. Its locked final-test accuracy was 0.7667,
macro F1 was 0.5833, QWK was 0.8348, and 92.73% of predictions were within one
grade of the reference label.

These results demonstrate an educational computer-vision experiment. They do
not establish clinical safety, diagnostic efficacy, or lesion localisation.
