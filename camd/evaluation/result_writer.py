import json
from pathlib import Path

from camd.context.models import (
    DefectPrediction,
    DefectSample,
)


def prediction_to_dict(
    sample: DefectSample,
    prediction: DefectPrediction,
) -> dict:

    return {
        "sample_id": sample.sample_id,
        "file": sample.file,
        "ground_truth": {
            "is_defective": sample.label,
            "defect_type": sample.defect_type,
            "buggy_line": sample.buggy_line,
        },
        "prediction": {
            "is_defective": prediction.is_defective,
            "defect_type": prediction.defect_type,
            "location": {
                "line": prediction.location.line,
                "function": prediction.location.function,
            },
            "explanation": prediction.explanation,
            "confidence": prediction.confidence,
        },
    }


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
            )

            file.write("\n")