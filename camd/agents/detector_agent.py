import json

from camd.agents.models import (
    DetectorAssessment,
)


DETECTOR_SYSTEM_PROMPT = """
You are the Detector Agent in a software defect localization system.

You will receive:

1. One candidate Java method.
2. Selected related context.
3. Static-analysis evidence.
4. A failing test.

Your job is to determine whether the candidate method is likely
to explain the CURRENT failing test.

Important:

1. Focus on the current failing test, not every possible defect.
2. A method may contain a real defect but still be unrelated
   to the current failure.
3. Identify the concrete connection between the candidate code
   and the failing assertions.
4. Avoid proposing a fix.
5. target_defect_probability must be between 0.0 and 1.0.

Return ONLY valid JSON:

{
  "hypothesis": "string",
  "supporting_evidence": [
    "string"
  ],
  "target_defect_probability": 0.0
}

Do not include Markdown.
"""


class DetectorAgent:

    def __init__(
        self,
        llm_client,
    ):
        self.llm_client = llm_client

    def analyze(
        self,
        method_name: str,
        candidate_context: str,
        failing_test_context: str,
    ) -> DetectorAssessment:

        prompt = f"""
Evaluate whether the following candidate method is likely to
cause the current failing test.

CANDIDATE
=========

Method:
{method_name}

{candidate_context}


FAILING TEST
============

{failing_test_context}

Determine the strongest defect hypothesis linking the candidate
to the failing test.

Do not discuss unrelated defects.
"""

        response = self.llm_client.generate(
            system_prompt=(
                DETECTOR_SYSTEM_PROMPT
            ),
            user_prompt=prompt,
        )

        result = self._parse_response(
            response
        )

        probability = self._normalize_probability(
            result.get(
                "target_defect_probability",
                0.0,
            )
        )

        return DetectorAssessment(
            method_name=method_name,
            hypothesis=result.get(
                "hypothesis",
                "",
            ),
            supporting_evidence=result.get(
                "supporting_evidence",
                [],
            ),
            target_defect_probability=(
                probability
            ),
        )

    @staticmethod
    def _normalize_probability(
        value,
    ) -> float:

        score = float(value)

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    @staticmethod
    def _parse_response(
        response: str,
    ) -> dict:

        text = response.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip()
                == "```"
            ):
                lines = lines[:-1]

            text = "\n".join(
                lines
            )

        try:
            return json.loads(
                text
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Detector returned invalid JSON:\n"
                f"{text}"
            ) from exc