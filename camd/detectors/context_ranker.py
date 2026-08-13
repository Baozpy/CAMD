import json
from dataclasses import dataclass

from camd.context.context_builder import (
    ContextBuilder,
)
from camd.context.context_formatter import (
    format_method_context,
)
from camd.context.method_extractor import (
    JavaMethod,
)


@dataclass
class ContextMethodSuspicion:
    method_name: str
    start_line: int
    end_line: int

    is_suspicious: bool
    suspicion_score: float

    defect_type: str
    reason: str

    callee_count: int
    caller_count: int


CONTEXT_RANKING_SYSTEM_PROMPT = """
You are an expert software defect detection system.

Your task is to analyze a Java target method together with
selected structural context from the same class.

The context may contain:

- the complete target method
- direct callee method signatures
- direct caller method signatures

Your goal is to determine whether the TARGET METHOD is likely
to contain a real functional software defect.

Important:

1. Judge the target method, not the surrounding methods.
2. Use caller/callee information only as supporting context.
3. Do not assume every target method contains a defect.
4. Do not report stylistic issues.
5. Do not report purely hypothetical issues without concrete evidence.
6. Focus on functional correctness.
7. suspicion_score must be between 0.0 and 1.0.

Interpretation:

0.0 = target method strongly appears correct
1.0 = target method strongly appears defective

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


def build_context_ranking_prompt(
    formatted_context: str,
) -> str:

    return f"""
Analyze the following Java target method and its structural context.

Determine:

1. Whether the TARGET METHOD likely contains a real defect.
2. The most likely defect category.
3. A suspicion score from 0.0 to 1.0.
4. A concise explanation based on the target code and context.

Do not propose a fix.

Context:

{formatted_context}
"""


class ContextMethodRanker:

    def __init__(
        self,
        llm_client,
        methods: list[JavaMethod],
    ):

        self.llm_client = llm_client

        self.context_builder = (
            ContextBuilder(
                methods=methods
            )
        )

    def analyze_method(
        self,
        method: JavaMethod,
    ) -> ContextMethodSuspicion:

        context = (
            self.context_builder.build(
                method
            )
        )

        formatted_context = (
            format_method_context(
                context
            )
        )

        prompt = (
            build_context_ranking_prompt(
                formatted_context
            )
        )

        response = (
            self.llm_client.generate(
                system_prompt=(
                    CONTEXT_RANKING_SYSTEM_PROMPT
                ),
                user_prompt=prompt,
            )
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
            min(
                1.0,
                suspicion_score,
            ),
        )

        return ContextMethodSuspicion(
            method_name=method.name,
            start_line=method.start_line,
            end_line=method.end_line,

            is_suspicious=bool(
                result.get(
                    "is_suspicious",
                    False,
                )
            ),

            suspicion_score=(
                suspicion_score
            ),

            defect_type=result.get(
                "defect_type",
                "none",
            ),

            reason=result.get(
                "reason",
                "",
            ),

            callee_count=len(
                context.callees
            ),

            caller_count=len(
                context.callers
            ),
        )

    def rank_methods(
        self,
        methods: list[JavaMethod],
    ) -> list[ContextMethodSuspicion]:

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

                result = (
                    self.analyze_method(
                        method
                    )
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

            print(
                f"  Context: "
                f"{result.callee_count} callees, "
                f"{result.caller_count} callers"
            )

            results.append(
                result
            )

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
                "Context ranking model "
                "returned invalid JSON:\n"
                f"{text}"
            ) from exc