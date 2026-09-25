"""Evaluate deterministic RetinaGuide grounding and safety behaviour."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinastage.retinaguide_evaluation import (  # noqa: E402
    evaluate_retinaguide_cases,
    write_evaluation_artifacts,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "results/retinaguide_evaluation",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    rows, summary = evaluate_retinaguide_cases()
    results_path, summary_path = write_evaluation_artifacts(
        rows,
        summary,
        arguments.output_directory,
    )

    print("RETINAGUIDE RESPONSE-LAYER EVALUATION")
    print("Cases:", summary["total_cases"])
    print("Passed:", summary["passed_cases"])
    print("Intent accuracy:", summary["intent_accuracy"])
    print("Grounding accuracy:", summary["grounding_accuracy"])
    print("Content accuracy:", summary["content_accuracy"])
    print("Determinism rate:", summary["determinism_rate"])
    print("Safety compliance:", summary["safety_compliance_rate"])
    print("Retinal images used:", summary["retinal_images_used"])
    print("Model inference used:", summary["model_inference_used"])
    print("Results:", results_path)
    print("Summary:", summary_path)
    if not summary["all_cases_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
