"""Evaluate RetinaTrace integrity, privacy and repeatability controls."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinastage.retinatrace_evaluation import (  # noqa: E402
    evaluate_retinatrace,
    write_evaluation_artifacts,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "results/retinatrace_evaluation",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    rows, summary = evaluate_retinatrace()
    results_path, summary_path = write_evaluation_artifacts(
        rows,
        summary,
        arguments.output_directory,
    )
    print("RETINATRACE RECEIPT EVALUATION")
    print("Checks:", summary["total_checks"])
    print("Passed:", summary["passed_checks"])
    print("Pass rate:", summary["pass_rate"])
    print("Tamper detection rate:", summary["tamper_detection_rate"])
    print("Privacy checks passed:", summary["privacy_checks_passed"])
    print("Retinal images used:", summary["retinal_images_used"])
    print("Model inference used:", summary["model_inference_used"])
    print("Results:", results_path)
    print("Summary:", summary_path)
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
