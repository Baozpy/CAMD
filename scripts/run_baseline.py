from pathlib import Path

from camd.detectors.llm_detector import LLMDetector
from camd.llm.client import OpenAIClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "NullPointerExample.java"
)


def load_code(file_path: Path) -> str:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Java source file not found: {file_path}"
        )

    return file_path.read_text(
        encoding="utf-8"
    )


def main():

    print("=" * 60)
    print("CAMD - LLM Software Defect Detection Baseline")
    print("=" * 60)

    code = load_code(SAMPLE_FILE)

    print(f"\nAnalyzing file: {SAMPLE_FILE.name}")

    client = OpenAIClient()

    detector = LLMDetector(
        llm_client=client
    )

    prediction = detector.detect(code)

    print("\nPrediction")
    print("-" * 60)

    print(
        f"Defective: {prediction.is_defective}"
    )

    print(
        f"Defect type: {prediction.defect_type}"
    )

    print(
        f"Line: {prediction.location.line}"
    )

    print(
        f"Function: {prediction.location.function}"
    )

    print(
        f"Confidence: {prediction.confidence:.2f}"
    )

    print("\nExplanation:")

    print(
        prediction.explanation
    )

    print("=" * 60)


if __name__ == "__main__":
    main()