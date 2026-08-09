from pathlib import Path

from camd.detectors.llm_detector import LLMDetector
from camd.llm.client import OpenAIClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1b"
    / "src"
    / "main"
    / "java"
    / "org"
    / "apache"
    / "commons"
    / "lang3"
    / "math"
    / "NumberUtils.java"
)


def load_code(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Source file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def main():

    print("=" * 70)
    print("CAMD - Defects4J Real-World Baseline")
    print("=" * 70)

    print("\nProject: Lang")
    print("Bug ID: 1")
    print("Version: buggy")
    print(
        "Modified class: "
        "org.apache.commons.lang3.math.NumberUtils"
    )

    code = load_code(SOURCE_FILE)

    print(
        f"\nSource file: {SOURCE_FILE.name}"
    )

    print(
        f"Source length: {len(code)} characters"
    )

    client = OpenAIClient()

    detector = LLMDetector(
        llm_client=client
    )

    prediction = detector.detect(code)

    print("\nPrediction")
    print("-" * 70)

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
    print(prediction.explanation)

    print("=" * 70)


if __name__ == "__main__":
    main()