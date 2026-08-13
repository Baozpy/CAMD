import json
from dataclasses import dataclass

from camd.context.method_extractor import JavaMethod


@dataclass
class MethodSuspicion:
    method_name: str
    start_line: int
    end_line: int

    is_suspicious: bool
    suspicion_score: float

    defect_type: str
    reason: str


METHOD_RANKING_SYSTEM_PROMPT = """
You are an expert software defect detection system.

Your task is to analyze one Java method and determine whether
the method is likely to contain a real software defect.

Possible defects include:

- incorrect conditional logic
- boundary errors
- null pointer errors
- arithmetic errors
- incorrect API usage
- incorrect state updates
- exception handling errors
- data-flow errors
- parsing errors
- numeric conversion errors
- resource management errors
- general functional bugs

Important rules:

1. Do not assume every method contains a defect.
2. Do not report stylistic issues.
3. Do not report hypothetical issues without concrete evidence.
4. Focus on functional correctness.
5. suspicion_score represents how likely the method is defective.
6. suspicion_score must be between 0.0 and 1.0.

Interpretation:

0.0 = strongly appears correct
1.0 = strongly appears defective

Return ONLY valid JSON:

{
  "is_suspicious": true,
  "suspicion_score": 0.0,
  "defect_type": "string",
  "reason": "string"
}

Do not include Markdown.
Do not include text outside the JSON object.
"""


def add_line_numbers(
    method: JavaMethod,
) -> str:

    lines = method.code.splitlines()

    numbered_lines = []

    for offset, line in enumerate(lines):

        absolute_line = (
            method.start_line + offset
        )

        numbered_lines.append(
            f"{absolute_line:5d}: {line}"
        )

    return "\n".join(numbered_lines)


def build_method_ranking_prompt(
    method: JavaMethod,
) -> str:

    numbered_code = add_line_numbers(
        method
    )

    return f"""
Analyze the following Java method for software defects.

Method name:
{method.name}

Original source range:
{method.start_line}-{method.end_line}

Determine:

1. Whether this method contains suspicious logic.
2. The most likely defect category.
3. A suspicion score from 0.0 to 1.0.
4. A short explanation based only on the provided code.

Do not propose a fix.

Java method:

{numbered_code}
"""


class MethodRanker:

    def __init__(
        self,
        llm_client,
    ):
        self.llm_client = llm_client

    def analyze_method(
        self,
        method: JavaMethod,
    ) -> MethodSuspicion:

        prompt = build_method_ranking_prompt(
            method
        )

        response = self.llm_client.generate(
            system_prompt=METHOD_RANKING_SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        result = self._parse_response(
            response
        )

        suspicion_score = float(
            result.get(
                "suspicion_score",
                0.0,
            )
        )

        suspicion_score = max(
            0.0,
            min(1.0, suspicion_score),
        )

        return MethodSuspicion(
            method_name=method.name,
            start_line=method.start_line,
            end_line=method.end_line,
            is_suspicious=bool(
                result.get(
                    "is_suspicious",
                    False,
                )
            ),
            suspicion_score=suspicion_score,
            defect_type=result.get(
                "defect_type",
                "none",
            ),
            reason=result.get(
                "reason",
                "",
            ),
        )

    def rank_methods(
        self,
        methods: list[JavaMethod],
    ) -> list[MethodSuspicion]:

        results = []

        total = len(methods)

        for index, method in enumerate(
            methods,
            start=1,
        ):

            print(
                f"[{index}/{total}] "
                f"Analyzing "
                f"{method.name} "
                f"({method.start_line}-"
                f"{method.end_line})"
            )

            try:
                result = self.analyze_method(
                    method
                )

            except Exception as exc:

                print(
                    f"  Failed: {exc}"
                )

                continue

            print(
                f"  Suspicious: "
                f"{result.is_suspicious}"
            )

            print(
                f"  Score: "
                f"{result.suspicion_score:.2f}"
            )

            print(
                f"  Type: "
                f"{result.defect_type}"
            )

            results.append(result)

        results.sort(
            key=lambda item: (
                item.suspicion_score
            ),
            reverse=True,
        )

        return results

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

            text = "\n".join(lines)

        try:
            return json.loads(text)

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Method ranking model returned "
                f"invalid JSON:\n{text}"
            ) from exc