import json

from camd.context.models import (
    DefectLocation,
    DefectPrediction,
)
from camd.detectors.base import BaseDetector
from camd.llm.prompts import (
    SYSTEM_PROMPT,
    build_detection_prompt,
)


class LLMDetector(BaseDetector):

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def detect(self, code: str) -> DefectPrediction:

        prompt = build_detection_prompt(code)

        response = self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        try:
            result = json.loads(response)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON:\n{response}"
            ) from exc

        location = result.get("location", {})

        confidence = float(
            result.get("confidence", 0.0)
        )

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        return DefectPrediction(
            is_defective=result.get(
                "is_defective",
                False,
            ),
            defect_type=result.get(
                "defect_type",
                "none",
            ),
            location=DefectLocation(
                line=int(
                    location.get("line", 0)
                ),
                function=location.get(
                    "function",
                    "none",
                ),
            ),
            explanation=result.get(
                "explanation",
                "",
            ),
            confidence=confidence,
        )