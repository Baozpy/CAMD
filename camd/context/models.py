from dataclasses import dataclass


@dataclass
class DefectLocation:
    line: int
    function: str


@dataclass
class DefectPrediction:
    is_defective: bool
    defect_type: str
    location: DefectLocation
    explanation: str
    confidence: float


@dataclass
class DefectSample:
    sample_id: str
    file: str
    label: bool
    defect_type: str
    buggy_line: int