from __future__ import annotations

from dataclasses import asdict, dataclass

from camd.agents.judge_agent import JudgeAgent
from camd.agents.models import (
    CriticAssessment,
    DetectorAssessment,
)
from camd.verification.critic import (
    ProgramWideCriticResult,
)
from camd.verification.detector import (
    ProgramWideDetector,
    ProgramWideDetectorResult,
)
from camd.verification.frozen_candidate_loader import (
    FrozenBugCase,
    FrozenCandidate,
)


@dataclass(frozen=True)
class ProgramWideJudgeResult:
    benchmark_id: str

    pool_position: int

    class_name: str
    method_name: str
    source_file: str

    start_line: int
    end_line: int

    detector_probability: float
    critic_probability: float

    is_target_defect: bool
    judge_probability: float

    defect_type: str
    reason: str

    def to_dict(
        self,
    ) -> dict:
        return asdict(
            self
        )


class ProgramWideJudge:

    def __init__(
        self,
        llm_client,
        *,
        include_retrieval_evidence: bool = True,
        max_failing_test_chars: int = 12000,
        max_candidate_chars: int = 12000,
    ) -> None:

        self.agent = JudgeAgent(
            llm_client=llm_client
        )

        self.context_builder = (
            ProgramWideDetector(
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
        )

    # =========================================================
    # Public API
    # =========================================================

    def analyze_candidate(
        self,
        case: FrozenBugCase,
        candidate: FrozenCandidate,
        detector_result: ProgramWideDetectorResult,
        critic_result: ProgramWideCriticResult,
    ) -> ProgramWideJudgeResult:

        self._validate_alignment(
            case=case,
            candidate=candidate,
            detector_result=detector_result,
            critic_result=critic_result,
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

        critic_assessment = (
            CriticAssessment(
                method_name=(
                    self.context_builder
                    .candidate_identifier(
                        candidate
                    )
                ),
                agrees_with_detector=(
                    critic_result
                    .agrees_with_detector
                ),
                weaknesses=list(
                    critic_result
                    .weaknesses
                ),
                alternative_explanation=(
                    critic_result
                    .alternative_explanation
                ),
                target_defect_probability=(
                    critic_result
                    .critic_probability
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
                critic_result=(
                    critic_assessment
                ),
            )
        )

        return ProgramWideJudgeResult(
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
            critic_probability=(
                critic_result
                .critic_probability
            ),
            is_target_defect=(
                assessment
                .is_target_defect
            ),
            judge_probability=(
                assessment
                .target_defect_probability
            ),
            defect_type=(
                assessment.defect_type
            ),
            reason=(
                assessment.reason
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
        critic_result: ProgramWideCriticResult,
    ) -> None:

        if (
            detector_result.benchmark_id
            != case.benchmark_id
        ):
            raise ValueError(
                "Detector result belongs to "
                f"{detector_result.benchmark_id}, "
                "but Judge received case "
                f"{case.benchmark_id}."
            )

        if (
            critic_result.benchmark_id
            != case.benchmark_id
        ):
            raise ValueError(
                "Critic result belongs to "
                f"{critic_result.benchmark_id}, "
                "but Judge received case "
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

        if (
            critic_result.pool_position
            != candidate.pool_position
        ):
            raise ValueError(
                "Critic result pool position "
                f"{critic_result.pool_position} "
                "does not match candidate "
                f"{candidate.pool_position}."
            )

        candidate_key = (
            candidate.class_name,
            candidate.source_file,
            candidate.start_line,
            candidate.end_line,
        )

        detector_key = (
            detector_result.class_name,
            detector_result.source_file,
            detector_result.start_line,
            detector_result.end_line,
        )

        critic_key = (
            critic_result.class_name,
            critic_result.source_file,
            critic_result.start_line,
            critic_result.end_line,
        )

        if detector_key != candidate_key:
            raise ValueError(
                "Detector result does not match "
                "the supplied candidate."
            )

        if critic_key != candidate_key:
            raise ValueError(
                "Critic result does not match "
                "the supplied candidate."
            )