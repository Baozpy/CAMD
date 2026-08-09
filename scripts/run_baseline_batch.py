from pathlib import Path

from camd.detectors.llm_detector import (
    LLMDetector,
)
from camd.evaluation.dataset import (
    load_dataset,
)
from camd.evaluation.evaluator import (
    evaluate_results,
)
from camd.evaluation.result_writer import (
    prediction_to_dict,
    save_results,
)
from camd.llm.client import OpenAIClient


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "samples.jsonl"
)

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "samples"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "baseline_predictions.jsonl"
)


def load_source_code(
    file_name: str,
) -> str:

    source_path = SOURCE_DIR / file_name

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source file does not exist: "
            f"{source_path}"
        )

    return source_path.read_text(
        encoding="utf-8"
    )


def print_metrics(metrics) -> None:

    print("\n")
    print("=" * 60)
    print("CAMD Baseline Evaluation")
    print("=" * 60)

    print(
        f"Total samples: "
        f"{metrics.total}"
    )

    print(
        f"True Positive: "
        f"{metrics.true_positive}"
    )

    print(
        f"True Negative: "
        f"{metrics.true_negative}"
    )

    print(
        f"False Positive: "
        f"{metrics.false_positive}"
    )

    print(
        f"False Negative: "
        f"{metrics.false_negative}"
    )

    print("-" * 60)

    print(
        f"Accuracy: "
        f"{metrics.accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics.precision:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics.recall:.4f}"
    )

    print(
        f"F1: "
        f"{metrics.f1:.4f}"
    )

    print(
        f"Localization Accuracy: "
        f"{metrics.localization_accuracy:.4f}"
    )

    print("=" * 60)


def main():

    print("=" * 60)
    print("CAMD - Batch LLM Defect Detection")
    print("=" * 60)

    samples = load_dataset(
        DATASET_FILE
    )

    print(
        f"Loaded {len(samples)} samples."
    )

    client = OpenAIClient()

    detector = LLMDetector(
        llm_client=client
    )

    results = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        print(
            f"\n[{index}/{len(samples)}] "
            f"Analyzing {sample.file}"
        )

        code = load_source_code(
            sample.file
        )

        try:
            prediction = detector.detect(
                code
            )

        except Exception as exc:

            print(
                f"Failed to analyze "
                f"{sample.file}: {exc}"
            )

            continue

        print(
            f"Prediction: "
            f"{prediction.is_defective}"
        )

        print(
            f"Type: "
            f"{prediction.defect_type}"
        )

        print(
            f"Line: "
            f"{prediction.location.line}"
        )

        print(
            f"Confidence: "
            f"{prediction.confidence:.2f}"
        )

        result = prediction_to_dict(
            sample=sample,
            prediction=prediction,
        )

        results.append(result)

    save_results(
        results=results,
        output_path=OUTPUT_FILE,
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )

    metrics = evaluate_results(
        results
    )

    print_metrics(metrics)


if __name__ == "__main__":
    main()