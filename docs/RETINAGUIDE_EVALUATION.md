# RetinaGuide response-layer evaluation

## Objective and scope

RetinaGuide is a deterministic explanation layer over one RetinaStage API
prediction. This evaluation asks whether it routes fixed questions correctly,
uses only the intended response fields, includes required factual and safety
content, and returns the same result when a question is repeated.

It does **not** evaluate diabetic-retinopathy classification, clinical
correctness, unrestricted conversation or the quality of a generative language
model. The runner uses no retinal images, no model inference, no final-test
records and no external language-model service.

## Controlled evaluation matrix

The version-1 matrix contains 15 cases across three API-shaped contexts:

| Context | Purpose | Cases |
|---|---|---:|
| Confident prediction | Stage, confidence, probabilities, no-review and summary behaviour | 6 |
| Uncertain prediction | Low-confidence review, next-step boundary and safety behaviour | 5 |
| Technical-quality review | Quality flag, Grad-CAM limitation and safety behaviour | 4 |

The cases cover nine routed intents: `stage`, `confidence`, `probabilities`,
`review`, `quality`, `gradcam`, `next_steps`, `summary` and `medical_advice`.
Three cases directly request diagnosis, treatment or medication and must be
refused.

## Metrics

| Metric | Passing rule |
|---|---|
| Intent accuracy | Detected intent exactly matches the case expectation |
| Grounding accuracy | Returned grounded-field set exactly matches the expected set |
| Content accuracy | Every required case-specific phrase is present |
| Determinism rate | Two executions produce identical response objects |
| Safety compliance | Medical-advice cases route to refusal with the fixed safety notice |

A case passes only when every applicable check passes. A failed case makes the
command exit with a non-zero status.

## Reproduction

From the repository root:

```bash
PYTHONPATH=src:. python -m unittest tests.test_retinaguide_evaluation -v
python scripts/evaluate_retinaguide.py
```

The runner creates stable, reviewable artifacts:

- `results/retinaguide_evaluation/retinaguide_evaluation_results.csv`
- `results/retinaguide_evaluation/retinaguide_evaluation_summary.json`

## Results

| Result | Value |
|---|---:|
| Fixed cases | 15 |
| Cases passed | 15 |
| Intent accuracy | 1.0000 |
| Grounding accuracy | 1.0000 |
| Required-content accuracy | 1.0000 |
| Determinism rate | 1.0000 |
| Safety compliance rate | 1.0000 |

These perfect values mean that the implemented fixed rules satisfy this known
test matrix. They must not be presented as evidence of clinical safety or as a
claim that all possible user wording is covered.

## Manual end-to-end checks

The complete local application was also exercised with validation images on
25 September 2026. These checks demonstrate integration among inference,
review policy, Grad-CAM, the frontend and RetinaGuide; they remain separate
from the automated response-layer metrics.

| Image | Reference grade | Displayed result | Confidence | Review state |
|---|---:|---|---:|---|
| `c3b15bf9b4bc.png` | 0 | Grade 0, No DR | 100.0% | No automated flag |
| `7d261f986bef.png` | 2 | Grade 1, Mild | 97.1% | No automated flag |
| `582115961a3d.png` | 1 | Grade 2, Moderate | 28.5% | Low model confidence |

The confidently incorrect example is intentionally retained: it demonstrates
why a high probability is not proof of correctness and why the interface does
not present the model output as a diagnosis.

## Limitations

- Fixed prompts cannot establish coverage of unrestricted natural language.
- Required phrases verify expected content, not human readability or usefulness.
- Controlled JSON contexts isolate chatbot behaviour from model performance.
- Manual checks are demonstrations, not a new statistical evaluation set.
- Qualified human review remains required for any real-world interpretation.
