import json

from camd.agents.models import (
    CriticAssessment,
    DetectorAssessment,
    JudgeAssessment,
)


JUDGE_SYSTEM_PROMPT = """
You are the Judge Agent in a software defect localization system.

You must make the final decision about whether a candidate method
is the most plausible source of the CURRENT failing test.

You will receive:

1. Candidate method evidence.
2. Failing test evidence.
3. Detector analysis.
4. Critic analysis.

Important:

1. Judge relevance to the CURRENT failing test.
2. A candidate may contain a real unrelated defect.
3. Prefer causal explanations that directly account for the
   failing test inputs and expected outputs.
4. Do not simply average Detector and Critic probabilities.
5. Resolve disagreements using the code and test evidence.
6. target_defect_probability must be between 0.0 and 1.0.

Return ONLY valid JSON:

{
  "is_target_defect": true,
  "target_defect_probability": 0.0,
  "defect_type": "string",
  "reason": "string"
}

Do not include Markdown.
"""


class JudgeAgent:

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
        critic_result: CriticAssessment,
    ) -> JudgeAssessment:

        detector_evidence = "\n".join(
            f"- {item}"
            for item in (
                detector_result
                .supporting_evidence
            )
        )

        critic_weaknesses = "\n".join(
            f"- {item}"
            for item in (
                critic_result
                .weaknesses
            )
        )

        prompt = f"""
Make the final defect-localization decision.

CANDIDATE
=========

Method:
{method_name}

{candidate_context}


FAILING TEST
============

{failing_test_context}


DETECTOR
========

Hypothesis:
{detector_result.hypothesis}

Evidence:
{detector_evidence}

Probability:
{detector_result.target_defect_probability:.4f}


CRITIC
======

Agrees with Detector:
{critic_result.agrees_with_detector}

Weaknesses:
{critic_weaknesses}

Alternative explanation:
{critic_result.alternative_explanation}

Probability:
{critic_result.target_defect_probability:.4f}


Determine whether this candidate most plausibly explains the
current failing test.
"""

        response = self.llm_client.generate(
            system_prompt=(
                JUDGE_SYSTEM_PROMPT
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

        return JudgeAssessment(
            method_name=method_name,

            is_target_defect=bool(
                result.get(
                    "is_target_defect",
                    False,
                )
            ),

            target_defect_probability=(
                probability
            ),

            defect_type=result.get(
                "defect_type",
                "none",
            ),

            reason=result.get(
                "reason",
                "",
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
                "Judge returned invalid JSON:\n"
                f"{text}"
            ) from exc