from __future__ import annotations

from dataclasses import asdict, dataclass

from camd.agents.critic_agent import CriticAgent
from camd.agents.models import DetectorAssessment
from camd.verification.detector import (
    ProgramWideDetector,
    ProgramWideDetectorResult,
)
from camd.verification.frozen_candidate_loader import (
    FrozenBugCase,
    FrozenCandidate,
)


@dataclass(frozen=True)
class ProgramWideCriticResult:
    benchmark_id: str

    pool_position: int

    class_name: str
    method_name: str
    source_file: str

    start_line: int
    end_line: int

    detector_probability: float
    detector_hypothesis: str

    agrees_with_detector: bool
    weaknesses: tuple[str, ...]
    alternative_explanation: str

    critic_probability: float

    def to_dict(
        self,
    ) -> dict:
        return asdict(
            self
        )


class ProgramWideCritic:

    def __init__(
        self,
        llm_client,
        *,
        include_retrieval_evidence: bool = True,
        max_failing_test_chars: int = 12000,
        max_candidate_chars: int = 12000,
    ) -> None:

        self.agent = CriticAgent(
            llm_client=llm_client
        )

        # Reuse the exact same context construction
        # used by the Detector wrapper.
        #
        # We only use its context-building helpers here;
        # DetectorAgent.analyze() is never called.
        self.context_builder = ProgramWideDetector(
            llm_client=llm_client,
            include_retrieval_evidence=(
                include_retrieval_evidence
            ),
            max_failing_test_chars=(
                max_failing_test_chars
            ),
            max_candidate_chars=(
                max_candidate_chars
            ),
        )

    # =========================================================
    # Public API
    # =========================================================

    def analyze_candidate(
        self,
        case: FrozenBugCase,
        candidate: FrozenCandidate,
        detector_result: ProgramWideDetectorResult,
    ) -> ProgramWideCriticResult:

        self._validate_alignment(
            case=case,
            candidate=candidate,
            detector_result=detector_result,
        )

        candidate_context = (
            self.context_builder
            .build_candidate_context(
                candidate
            )
        )

        failing_test_context = (
            self.context_builder
            .build_failing_test_context(
                case
            )
        )

        # Convert the program-wide Detector result
        # back into the original DetectorAssessment
        # expected by CriticAgent.
        detector_assessment = (
            DetectorAssessment(
                method_name=(
                    self.context_builder
                    .candidate_identifier(
                        candidate
                    )
                ),
                hypothesis=(
                    detector_result.hypothesis
                ),
                supporting_evidence=list(
                    detector_result
                    .supporting_evidence
                ),
                target_defect_probability=(
                    detector_result
                    .target_defect_probability
                ),
            )
        )

        assessment = (
            self.agent.analyze(
                method_name=(
                    self.context_builder
                    .candidate_identifier(
                        candidate
                    )
                ),
                candidate_context=(
                    candidate_context
                ),
                failing_test_context=(
                    failing_test_context
                ),
                detector_result=(
                    detector_assessment
                ),
            )
        )

        return ProgramWideCriticResult(
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
            detector_probability=(
                detector_result
                .target_defect_probability
            ),
            detector_hypothesis=(
                detector_result.hypothesis
            ),
            agrees_with_detector=(
                assessment
                .agrees_with_detector
            ),
            weaknesses=tuple(
                assessment.weaknesses
            ),
            alternative_explanation=(
                assessment
                .alternative_explanation
            ),
            critic_probability=(
                assessment
                .target_defect_probability
            ),
        )

    # =========================================================
    # Validation
    # =========================================================

    @staticmethod
    def _validate_alignment(
        case: FrozenBugCase,
        candidate: FrozenCandidate,
        detector_result: ProgramWideDetectorResult,
    ) -> None:

        if (
            detector_result.benchmark_id
            != case.benchmark_id
        ):
            raise ValueError(
                "Detector result belongs to "
                f"{detector_result.benchmark_id}, "
                "but Critic received case "
                f"{case.benchmark_id}."
            )

        if (
            detector_result.pool_position
            != candidate.pool_position
        ):
            raise ValueError(
                "Detector result pool position "
                f"{detector_result.pool_position} "
                "does not match candidate "
                f"{candidate.pool_position}."
            )

        detector_key = (
            detector_result.class_name,
            detector_result.source_file,
            detector_result.start_line,
            detector_result.end_line,
        )

        candidate_key = (
            candidate.class_name,
            candidate.source_file,
            candidate.start_line,
            candidate.end_line,
        )

        if detector_key != candidate_key:
            raise ValueError(
                "Detector result does not match "
                "the supplied candidate.\n"
                f"Detector: {detector_key}\n"
                f"Candidate: {candidate_key}"
            )