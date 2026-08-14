import json

from camd.agents.models import (
    CriticAssessment,
    DetectorAssessment,
)


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent in a software defect localization system.

Your task is to critically evaluate a Detector Agent's claim
that a candidate method explains the CURRENT failing test.

Important:

1. Distinguish between:
   - a real defect in the method
   - the defect responsible for the current failing test

2. Look for mismatches between:
   - candidate behavior
   - failing test inputs
   - expected outputs
   - Detector reasoning

3. Do not agree automatically with the Detector.
4. Do not reject automatically either.
5. target_defect_probability must be between 0.0 and 1.0.

Return ONLY valid JSON:

{
  "agrees_with_detector": true,
  "weaknesses": [
    "string"
  ],
  "alternative_explanation": "string",
  "target_defect_probability": 0.0
}

Do not include Markdown.
"""


class CriticAgent:

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
        detector_result: DetectorAssessment,
    ) -> CriticAssessment:

        detector_evidence = "\n".join(
            f"- {item}"
            for item in (
                detector_result
                .supporting_evidence
            )
        )

        prompt = f"""
Critically evaluate the Detector Agent's conclusion.

CANDIDATE
=========

Method:
{method_name}

{candidate_context}


FAILING TEST
============

{failing_test_context}


DETECTOR ASSESSMENT
===================

Hypothesis:
{detector_result.hypothesis}

Supporting evidence:
{detector_evidence}

Detector target defect probability:
{detector_result.target_defect_probability:.4f}

Determine whether this candidate actually explains the current
failing test, rather than merely containing some unrelated defect.
"""

        response = self.llm_client.generate(
            system_prompt=(
                CRITIC_SYSTEM_PROMPT
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

        return CriticAssessment(
            method_name=method_name,

            agrees_with_detector=bool(
                result.get(
                    "agrees_with_detector",
                    False,
                )
            ),

            weaknesses=result.get(
                "weaknesses",
                [],
            ),

            alternative_explanation=(
                result.get(
                    "alternative_explanation",
                    "",
                )
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
                "Critic returned invalid JSON:\n"
                f"{text}"
            ) from exc