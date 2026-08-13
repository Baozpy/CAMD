import json
from dataclasses import dataclass

from camd.context.method_extractor import (
    JavaMethod,
)
from camd.context.semantic_context_builder import (
    SemanticContextBuilder,
    SemanticMethodContext,
)
from camd.static.ast_analyzer import (
    JavaASTAnalyzer,
)
from camd.static.evidence_builder import (
    StaticEvidenceBuilder,
)


@dataclass
class StaticContextMethodSuspicion:
    method_name: str
    start_line: int
    end_line: int

    is_suspicious: bool
    suspicion_score: float

    defect_type: str
    reason: str

    selected_context_count: int
    selected_context_methods: list[str]

    condition_count: int
    loop_count: int
    return_count: int
    throw_count: int


STATIC_CONTEXT_SYSTEM_PROMPT = """
You are an expert software defect detection system.

You will receive:

1. A TARGET Java method.
2. A small set of selected semantically related methods.
3. Static-analysis evidence extracted from the TARGET METHOD's AST.

The static-analysis evidence is factual structural information.
It may include:

- branch counts
- loop counts
- return statements
- throw statements
- method calls
- comparisons
- numeric literals
- null checks
- thrown exceptions

Your task is to determine whether the TARGET METHOD contains
a real functional software defect.

Important rules:

1. Judge the TARGET METHOD, not the context methods.
2. Use related methods only as supporting semantic context.
3. Treat static-analysis evidence as factual structural evidence,
   not as proof that a defect exists.
4. Pay special attention to relationships between conditions,
   boundary values, method calls, and return behavior.
5. Do not report stylistic issues.
6. Do not assume every method is defective.
7. Avoid speculative defects without concrete code evidence.
8. suspicion_score must be between 0.0 and 1.0.

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

    return "\n".join(
        output
    )


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


class StaticContextRanker:

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

        self.ast_analyzer = (
            JavaASTAnalyzer()
        )

        self.evidence_builder = (
            StaticEvidenceBuilder()
        )

    def analyze_method(
        self,
        method: JavaMethod,
    ) -> StaticContextMethodSuspicion:

        semantic_context = (
            self.context_builder.build(
                method
            )
        )

        formatted_context = (
            format_semantic_context(
                semantic_context
            )
        )

        evidence = (
            self.ast_analyzer.analyze(
                method
            )
        )

        formatted_evidence = (
            self.evidence_builder.build_text(
                evidence
            )
        )

        prompt = f"""
Analyze the TARGET METHOD using both semantic context
and static-analysis evidence.

Determine:

1. Whether the target contains a real functional defect.
2. The most likely defect category.
3. A suspicion score between 0.0 and 1.0.
4. A concise explanation grounded in the code and evidence.

Do not propose a fix.

SEMANTIC CONTEXT
================

{formatted_context}


STATIC EVIDENCE
===============

{formatted_evidence}
"""

        response = (
            self.llm_client.generate(
                system_prompt=(
                    STATIC_CONTEXT_SYSTEM_PROMPT
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
                semantic_context.selected_methods
            )
        ]

        return StaticContextMethodSuspicion(
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
                semantic_context.selected_methods
            ),

            selected_context_methods=(
                selected_names
            ),

            condition_count=(
                evidence.condition_count
            ),

            loop_count=(
                evidence.loop_count
            ),

            return_count=(
                evidence.return_count
            ),

            throw_count=(
                evidence.throw_count
            ),
        )

    def rank_methods(
        self,
        methods: list[JavaMethod],
    ) -> list[
        StaticContextMethodSuspicion
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

            print(
                "  Static evidence: "
                f"{result.condition_count} conditions, "
                f"{result.loop_count} loops, "
                f"{result.return_count} returns, "
                f"{result.throw_count} throws"
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
                "Static-context model "
                "returned invalid JSON:\n"
                f"{text}"
            ) from exc