from __future__ import annotations

from dataclasses import asdict, dataclass

from camd.agents.detector_agent import DetectorAgent
from camd.verification.frozen_candidate_loader import (
    FrozenBugCase,
    FrozenCandidate,
    FrozenFailingTest,
)


@dataclass(frozen=True)
class ProgramWideDetectorResult:
    benchmark_id: str

    pool_position: int

    class_name: str
    method_name: str
    source_file: str

    start_line: int
    end_line: int

    hypothesis: str
    supporting_evidence: tuple[str, ...]

    target_defect_probability: float

    base_rank: int | None
    base_score: float | None

    from_base: bool
    from_stack: bool
    from_call: bool

    stack_depth: int | None
    call_depth: int | None

    def to_dict(
        self,
    ) -> dict:
        return asdict(
            self
        )


class ProgramWideDetector:

    def __init__(
        self,
        llm_client,
        *,
        include_retrieval_evidence: bool = True,
        max_failing_test_chars: int = 12000,
        max_candidate_chars: int = 12000,
    ) -> None:

        self.agent = DetectorAgent(
            llm_client=llm_client
        )

        self.include_retrieval_evidence = (
            include_retrieval_evidence
        )

        self.max_failing_test_chars = (
            max_failing_test_chars
        )

        self.max_candidate_chars = (
            max_candidate_chars
        )

    # =========================================================
    # Public API
    # =========================================================

    def analyze_candidate(
        self,
        case: FrozenBugCase,
        candidate: FrozenCandidate,
    ) -> ProgramWideDetectorResult:

        candidate_context = (
            self.build_candidate_context(
                candidate
            )
        )

        failing_test_context = (
            self.build_failing_test_context(
                case
            )
        )

        assessment = (
            self.agent.analyze(
                method_name=(
                    self.candidate_identifier(
                        candidate
                    )
                ),
                candidate_context=(
                    candidate_context
                ),
                failing_test_context=(
                    failing_test_context
                ),
            )
        )

        return ProgramWideDetectorResult(
            benchmark_id=(
                case.benchmark_id
            ),
            pool_position=(
                candidate.pool_position
            ),
            class_name=(
                candidate.class_name
            ),
            method_name=(
                candidate.method_name
            ),
            source_file=(
                candidate.source_file
            ),
            start_line=(
                candidate.start_line
            ),
            end_line=(
                candidate.end_line
            ),
            hypothesis=(
                assessment.hypothesis
            ),
            supporting_evidence=tuple(
                assessment.supporting_evidence
            ),
            target_defect_probability=(
                assessment
                .target_defect_probability
            ),
            base_rank=(
                candidate.base_rank
            ),
            base_score=(
                candidate.base_score
            ),
            from_base=(
                candidate.from_base
            ),
            from_stack=(
                candidate.from_stack
            ),
            from_call=(
                candidate.from_call
            ),
            stack_depth=(
                candidate.stack_depth
            ),
            call_depth=(
                candidate.call_depth
            ),
        )

    def analyze_case(
        self,
        case: FrozenBugCase,
    ) -> list[
        ProgramWideDetectorResult
    ]:

        results = []

        for candidate in (
            case.candidates
        ):

            result = (
                self.analyze_candidate(
                    case,
                    candidate,
                )
            )

            results.append(
                result
            )

        return results

    # =========================================================
    # Candidate context
    # =========================================================

    def build_candidate_context(
        self,
        candidate: FrozenCandidate,
    ) -> str:

        parts = [
            "Candidate Java method:",
            "",
            (
                f"Class: "
                f"{candidate.class_name}"
            ),
            (
                f"Method: "
                f"{candidate.method_name}"
            ),
            (
                f"Source file: "
                f"{candidate.source_file}"
            ),
            (
                f"Lines: "
                f"{candidate.start_line}-"
                f"{candidate.end_line}"
            ),
        ]

        if (
            self.include_retrieval_evidence
        ):
            parts.extend(
                [
                    "",
                    "Retrieval evidence:",
                    (
                        f"- Base rank: "
                        f"{candidate.base_rank}"
                    ),
                    (
                        f"- Base score: "
                        f"{candidate.base_score}"
                    ),
                    (
                        f"- Selected by base retrieval: "
                        f"{candidate.from_base}"
                    ),
                    (
                        f"- Selected by exact stack evidence: "
                        f"{candidate.from_stack}"
                    ),
                    (
                        f"- Stack depth: "
                        f"{candidate.stack_depth}"
                    ),
                    (
                        f"- Selected by call-chain evidence: "
                        f"{candidate.from_call}"
                    ),
                    (
                        f"- Call-chain depth: "
                        f"{candidate.call_depth}"
                    ),
                ]
            )

        code = (
            candidate.code
            or ""
        )

        if (
            len(code)
            > self.max_candidate_chars
        ):
            code = (
                code[
                    :self.max_candidate_chars
                ]
                + "\n"
                + "/* truncated */"
            )

        parts.extend(
            [
                "",
                "Method code:",
                "```java",
                code,
                "```",
            ]
        )

        return "\n".join(
            parts
        )

    # =========================================================
    # Failure context
    # =========================================================

    def build_failing_test_context(
        self,
        case: FrozenBugCase,
    ) -> str:

        parts = [
            (
                f"Benchmark: "
                f"{case.benchmark_id}"
            ),
            (
                f"Project: "
                f"{case.project}"
            ),
            "",
            (
                "The following failing-test "
                "evidence was frozen before "
                "verification."
            ),
        ]

        if not case.failing_tests:

            parts.append(
                "\nNo frozen failing-test "
                "source is available."
            )

            return "\n".join(
                parts
            )

        for index, test in enumerate(
            case.failing_tests,
            start=1,
        ):

            parts.extend(
                [
                    "",
                    "=" * 72,
                    (
                        f"Failing test "
                        f"{index}/"
                        f"{len(case.failing_tests)}"
                    ),
                    "=" * 72,
                    (
                        f"Name: "
                        f"{test.full_name}"
                    ),
                    (
                        f"Class: "
                        f"{test.class_name}"
                    ),
                    (
                        f"Method: "
                        f"{test.method_name}"
                    ),
                ]
            )

            if (
                test.source_file
                is not None
            ):
                parts.append(
                    (
                        f"Source file: "
                        f"{test.source_file}"
                    )
                )

            if (
                test.start_line
                is not None
            ):
                parts.append(
                    (
                        f"Lines: "
                        f"{test.start_line}-"
                        f"{test.end_line}"
                    )
                )

            code = (
                test.code
                or ""
            )

            if code:
                parts.extend(
                    [
                        "",
                        "Test code:",
                        "```java",
                        code,
                        "```",
                    ]
                )

        text = "\n".join(
            parts
        )

        if (
            len(text)
            > self.max_failing_test_chars
        ):
            text = (
                text[
                    :self.max_failing_test_chars
                ]
                + "\n"
                + "[failure context truncated]"
            )

        return text

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def candidate_identifier(
        candidate: FrozenCandidate,
    ) -> str:

        return (
            f"{candidate.class_name}."
            f"{candidate.method_name}"
            f"[{candidate.start_line}-"
            f"{candidate.end_line}]"
        )