import json
from dataclasses import dataclass

from camd.context.method_extractor import (
    JavaMethod,
)
from camd.context.semantic_context_builder import (
    SemanticContextBuilder,
    SemanticMethodContext,
)


@dataclass
class SemanticMethodSuspicion:
    method_name: str
    start_line: int
    end_line: int

    is_suspicious: bool
    suspicion_score: float

    defect_type: str
    reason: str

    selected_context_count: int
    selected_context_methods: list[str]


SEMANTIC_CONTEXT_SYSTEM_PROMPT = """
You are an expert software defect detection system.

You will receive:

1. A TARGET Java method.
2. A small set of selected related methods from the same class.

The related methods were selected because they may help explain
the behavior, data flow, value conversion, or control decisions
of the target method.

Your task is to determine whether the TARGET METHOD contains
a real functional software defect.

Important rules:

1. Judge the TARGET METHOD, not the context methods.
2. Use related methods only to understand the target behavior.
3. Do not report stylistic issues.
4. Do not assume every target method is defective.
5. Avoid speculative defects without concrete evidence.
6. suspicion_score must be between 0.0 and 1.0.

Interpretation:

0.0 = target strongly appears correct
1.0 = target strongly appears defective

Return ONLY valid JSON:

{
  "is_suspicious": true,
  "suspicion_score": 0.0,
  "defect_type": "string",
  "reason": "string"
}

Do not include Markdown.
Do not include any text outside the JSON object.
"""


def add_line_numbers(
    method: JavaMethod,
) -> str:

    output = []

    for offset, line in enumerate(
        method.code.splitlines()
    ):

        line_number = (
            method.start_line
            + offset
        )

        output.append(
            f"{line_number:5d}: {line}"
        )

    return "\n".join(output)


def format_semantic_context(
    context: SemanticMethodContext,
) -> str:

    output = []

    output.append(
        "TARGET METHOD"
    )

    output.append(
        "============="
    )

    output.append(
        f"Name: {context.target.name}"
    )

    output.append(
        f"Lines: "
        f"{context.target.start_line}-"
        f"{context.target.end_line}"
    )

    output.append("")

    output.append(
        add_line_numbers(
            context.target
        )
    )

    output.append("")
    output.append(
        "SELECTED RELATED METHODS"
    )

    output.append(
        "========================"
    )

    if not context.selected_methods:
        output.append(
            "None"
        )

    for item in (
        context.selected_methods
    ):

        method = item.method

        output.append("")

        output.append(
            f"Relation: "
            f"{item.relation}"
        )

        output.append(
            f"Method: "
            f"{method.name}"
        )

        output.append(
            f"Lines: "
            f"{method.start_line}-"
            f"{method.end_line}"
        )

        output.append(
            f"Relevance score: "
            f"{item.relevance_score:.2f}"
        )

        output.append(
            f"Selection reason: "
            f"{item.reason}"
        )

        output.append("")

        output.append(
            add_line_numbers(
                method
            )
        )

    return "\n".join(
        output
    )


class SemanticContextRanker:

    def __init__(
        self,
        llm_client,
        methods: list[JavaMethod],
        top_k_callees: int = 3,
        top_k_callers: int = 2,
    ):

        self.llm_client = (
            llm_client
        )

        self.context_builder = (
            SemanticContextBuilder(
                methods=methods,
                top_k_callees=(
                    top_k_callees
                ),
                top_k_callers=(
                    top_k_callers
                ),
            )
        )

    def analyze_method(
        self,
        method: JavaMethod,
    ) -> SemanticMethodSuspicion:

        context = (
            self.context_builder.build(
                method
            )
        )

        formatted_context = (
            format_semantic_context(
                context
            )
        )

        prompt = f"""
Analyze the TARGET METHOD using the selected semantic context.

Determine:

1. Whether the target contains a real functional defect.
2. The most likely defect type.
3. A suspicion score between 0.0 and 1.0.
4. A concise explanation.

Do not propose a fix.

{formatted_context}
"""

        response = (
            self.llm_client.generate(
                system_prompt=(
                    SEMANTIC_CONTEXT_SYSTEM_PROMPT
                ),
                user_prompt=prompt,
            )
        )

        result = (
            self._parse_response(
                response
            )
        )

        score = float(
            result.get(
                "suspicion_score",
                0.0,
            )
        )

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        selected_names = [
            item.method.name
            for item in (
                context.selected_methods
            )
        ]

        return SemanticMethodSuspicion(
            method_name=method.name,
            start_line=method.start_line,
            end_line=method.end_line,
            is_suspicious=bool(
                result.get(
                    "is_suspicious",
                    False,
                )
            ),
            suspicion_score=score,
            defect_type=result.get(
                "defect_type",
                "none",
            ),
            reason=result.get(
                "reason",
                "",
            ),
            selected_context_count=len(
                context.selected_methods
            ),
            selected_context_methods=(
                selected_names
            ),
        )

    def rank_methods(
        self,
        methods: list[JavaMethod],
    ) -> list[
        SemanticMethodSuspicion
    ]:

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
                "  Selected context: "
                + (
                    ", ".join(
                        result
                        .selected_context_methods
                    )
                    if result
                    .selected_context_methods
                    else "None"
                )
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
                "Semantic context model "
                "returned invalid JSON:\n"
                f"{text}"
            ) from exc