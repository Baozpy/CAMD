import json
from dataclasses import dataclass


@dataclass
class SuspiciousLine:
    line: int
    score: float
    reason: str


LINE_RANKER_SYSTEM_PROMPT = """
You are a line-level software defect localization model.

You will receive:

1. One Java method that has already been selected as a likely defective method.
2. Static evidence about that method.
3. Expanded failing-test evidence.

Your task is to rank the individual source lines inside the candidate method
according to how likely each line is to be responsible for the CURRENT failing test.

Important rules:

1. Focus on the current failing test.
2. Only return source lines that belong to the candidate method.
3. Prefer executable or condition-bearing lines.
4. Do not return comments, braces, or blank lines unless absolutely necessary.
5. A high score means the line is strongly implicated in the current failure.
6. Scores must be between 0.0 and 1.0.
7. Return at most the requested number of lines.
8. Do not propose a patch.

Return ONLY valid JSON in this format:

{
  "lines": [
    {
      "line": 123,
      "score": 0.95,
      "reason": "..."
    }
  ]
}

Do not include Markdown.
"""


class LineRanker:

    def __init__(
        self,
        llm_client,
    ):
        self.llm_client = llm_client

    def rank(
        self,
        method_name: str,
        method_code_with_lines: str,
        static_evidence: str,
        failing_test_context: str,
        top_k: int = 10,
    ) -> list[SuspiciousLine]:

        prompt = f"""
Rank the most suspicious source lines in the candidate method.

METHOD
======

Name:
{method_name}

Source:
{method_code_with_lines}


STATIC EVIDENCE
===============

{static_evidence}


FAILING TEST EVIDENCE
=====================

{failing_test_context}


Return at most {top_k} suspicious source lines.
"""

        response = self.llm_client.generate(
            system_prompt=(
                LINE_RANKER_SYSTEM_PROMPT
            ),
            user_prompt=prompt,
        )

        result = self._parse_response(
            response
        )

        output = []

        for item in result.get(
            "lines",
            [],
        ):

            try:
                line = int(
                    item["line"]
                )

                score = float(
                    item["score"]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            score = max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

            output.append(
                SuspiciousLine(
                    line=line,
                    score=score,
                    reason=item.get(
                        "reason",
                        "",
                    ),
                )
            )

        output.sort(
            key=lambda item: (
                item.score
            ),
            reverse=True,
        )

        return output[:top_k]

    @staticmethod
    def _parse_response(
        response: str,
    ) -> dict:

        text = response.strip()

        if text.startswith("```"):

            lines = (
                text.splitlines()
            )

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
                "LineRanker returned invalid JSON:\n"
                f"{text}"
            ) from exc