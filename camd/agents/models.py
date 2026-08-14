from dataclasses import dataclass


@dataclass
class DetectorAssessment:
    method_name: str
    hypothesis: str
    supporting_evidence: list[str]
    target_defect_probability: float


@dataclass
class CriticAssessment:
    method_name: str
    agrees_with_detector: bool
    weaknesses: list[str]
    alternative_explanation: str
    target_defect_probability: float


@dataclass
class JudgeAssessment:
    method_name: str
    is_target_defect: bool
    target_defect_probability: float
    defect_type: str
    reason: str