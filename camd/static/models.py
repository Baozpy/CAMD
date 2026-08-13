from dataclasses import dataclass, field


@dataclass
class StaticEvidence:
    method_name: str
    start_line: int
    end_line: int

    condition_count: int = 0
    loop_count: int = 0
    return_count: int = 0
    throw_count: int = 0

    method_calls: list[str] = field(
        default_factory=list
    )

    comparisons: list[str] = field(
        default_factory=list
    )

    numeric_literals: list[str] = field(
        default_factory=list
    )

    null_checks: list[str] = field(
        default_factory=list
    )

    thrown_exceptions: list[str] = field(
        default_factory=list
    )