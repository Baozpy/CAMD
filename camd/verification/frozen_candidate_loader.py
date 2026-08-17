from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_BUDGETS = {
    10,
    20,
    50,
    100,
}


@dataclass(frozen=True)
class FrozenCandidate:
    pool_position: int

    class_name: str
    method_name: str
    source_file: str

    start_line: int
    end_line: int

    code: str

    base_rank: int | None
    base_score: float | None

    from_base: bool
    from_stack: bool
    from_call: bool

    stack_depth: int | None
    stack_evidence_type: str | None

    call_depth: int | None
    runtime_class: str | None
    call_evidence_type: str | None
    originating_test: str | None

    is_ground_truth: bool


@dataclass(frozen=True)
class FrozenGroundTruth:
    class_name: str
    method_name: str
    source_file: str

    start_line: int
    end_line: int

    code: str


@dataclass(frozen=True)
class FrozenFailingTest:
    full_name: str
    class_name: str
    method_name: str

    source_file: str | None

    start_line: int | None
    end_line: int | None

    code: str | None


@dataclass(frozen=True)
class FrozenBugCase:
    benchmark_id: str

    project: str
    bug_id: int

    applicability: str
    method_applicable: bool

    production_method_count: int

    ground_truth: tuple[
        FrozenGroundTruth,
        ...
    ]

    failing_tests: tuple[
        FrozenFailingTest,
        ...
    ]

    failure_trace_count: int

    budget: int

    candidate_recall: bool | None

    candidates: tuple[
        FrozenCandidate,
        ...
    ]


class FrozenCandidateLoader:

    def __init__(
        self,
        manifest_path: str | Path,
    ) -> None:

        self.manifest_path = Path(
            manifest_path
        )

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                "Frozen candidate manifest "
                f"does not exist: "
                f"{self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.data = json.load(
                file
            )

        self._records = (
            self.data.get(
                "records",
                []
            )
        )

        self._record_index = {}

        for record in self._records:

            benchmark_id = (
                record.get(
                    "benchmark_id"
                )
            )

            if not benchmark_id:
                continue

            if (
                benchmark_id
                in self._record_index
            ):

                raise ValueError(
                    "Duplicate benchmark_id "
                    "in frozen manifest: "
                    f"{benchmark_id}"
                )

            self._record_index[
                benchmark_id
            ] = record

    # =========================================================
    # Public metadata
    # =========================================================

    @property
    def artifact_name(
        self,
    ) -> str | None:

        return self.data.get(
            "artifact"
        )

    @property
    def benchmark_path(
        self,
    ) -> str | None:

        return self.data.get(
            "benchmark"
        )

    @property
    def frozen_configuration(
        self,
    ) -> dict[str, Any]:

        return dict(
            self.data.get(
                "frozen_configuration",
                {}
            )
        )

    def benchmark_ids(
        self,
        *,
        only_successful: bool = True,
        only_method_applicable: bool = False,
    ) -> list[str]:

        result = []

        for record in self._records:

            if (
                only_successful
                and not record.get(
                    "success"
                )
            ):
                continue

            if (
                only_method_applicable
                and not record.get(
                    "method_applicable"
                )
            ):
                continue

            benchmark_id = (
                record.get(
                    "benchmark_id"
                )
            )

            if benchmark_id:

                result.append(
                    benchmark_id
                )

        return result

    def get_raw_record(
        self,
        benchmark_id: str,
    ) -> dict[str, Any]:

        record = (
            self._record_index.get(
                benchmark_id
            )
        )

        if record is None:

            raise KeyError(
                "Unknown benchmark_id: "
                f"{benchmark_id}"
            )

        return record

    # =========================================================
    # Deserialization
    # =========================================================

    @staticmethod
    def _parse_ground_truth(
        raw_items,
    ) -> tuple[
        FrozenGroundTruth,
        ...
    ]:

        result = []

        for item in raw_items:

            result.append(
                FrozenGroundTruth(
                    class_name=(
                        item[
                            "class_name"
                        ]
                    ),
                    method_name=(
                        item[
                            "method_name"
                        ]
                    ),
                    source_file=(
                        item[
                            "source_file"
                        ]
                    ),
                    start_line=int(
                        item[
                            "start_line"
                        ]
                    ),
                    end_line=int(
                        item[
                            "end_line"
                        ]
                    ),
                    code=(
                        item.get(
                            "code"
                        )
                        or ""
                    ),
                )
            )

        return tuple(
            result
        )

    @staticmethod
    def _parse_failing_tests(
        raw_items,
    ) -> tuple[
        FrozenFailingTest,
        ...
    ]:

        result = []

        for item in raw_items:

            result.append(
                FrozenFailingTest(
                    full_name=(
                        item[
                            "full_name"
                        ]
                    ),
                    class_name=(
                        item[
                            "class_name"
                        ]
                    ),
                    method_name=(
                        item[
                            "method_name"
                        ]
                    ),
                    source_file=(
                        item.get(
                            "source_file"
                        )
                    ),
                    start_line=(
                        int(
                            item[
                                "start_line"
                            ]
                        )
                        if item.get(
                            "start_line"
                        )
                        is not None
                        else None
                    ),
                    end_line=(
                        int(
                            item[
                                "end_line"
                            ]
                        )
                        if item.get(
                            "end_line"
                        )
                        is not None
                        else None
                    ),
                    code=(
                        item.get(
                            "code"
                        )
                    ),
                )
            )

        return tuple(
            result
        )

    @staticmethod
    def _parse_candidates(
        raw_items,
    ) -> tuple[
        FrozenCandidate,
        ...
    ]:

        result = []

        for item in raw_items:

            result.append(
                FrozenCandidate(
                    pool_position=int(
                        item[
                            "pool_position"
                        ]
                    ),
                    class_name=(
                        item[
                            "class_name"
                        ]
                    ),
                    method_name=(
                        item[
                            "method_name"
                        ]
                    ),
                    source_file=(
                        item[
                            "source_file"
                        ]
                    ),
                    start_line=int(
                        item[
                            "start_line"
                        ]
                    ),
                    end_line=int(
                        item[
                            "end_line"
                        ]
                    ),
                    code=(
                        item.get(
                            "code"
                        )
                        or ""
                    ),
                    base_rank=(
                        int(
                            item[
                                "base_rank"
                            ]
                        )
                        if item.get(
                            "base_rank"
                        )
                        is not None
                        else None
                    ),
                    base_score=(
                        float(
                            item[
                                "base_score"
                            ]
                        )
                        if item.get(
                            "base_score"
                        )
                        is not None
                        else None
                    ),
                    from_base=bool(
                        item.get(
                            "from_base",
                            False,
                        )
                    ),
                    from_stack=bool(
                        item.get(
                            "from_stack",
                            False,
                        )
                    ),
                    from_call=bool(
                        item.get(
                            "from_call",
                            False,
                        )
                    ),
                    stack_depth=(
                        int(
                            item[
                                "stack_depth"
                            ]
                        )
                        if item.get(
                            "stack_depth"
                        )
                        is not None
                        else None
                    ),
                    stack_evidence_type=(
                        item.get(
                            "stack_evidence_type"
                        )
                    ),
                    call_depth=(
                        int(
                            item[
                                "call_depth"
                            ]
                        )
                        if item.get(
                            "call_depth"
                        )
                        is not None
                        else None
                    ),
                    runtime_class=(
                        item.get(
                            "runtime_class"
                        )
                    ),
                    call_evidence_type=(
                        item.get(
                            "call_evidence_type"
                        )
                    ),
                    originating_test=(
                        item.get(
                            "originating_test"
                        )
                    ),
                    is_ground_truth=bool(
                        item.get(
                            "is_ground_truth",
                            False,
                        )
                    ),
                )
            )

        result.sort(
            key=lambda candidate: (
                candidate.pool_position
            )
        )

        return tuple(
            result
        )

    # =========================================================
    # Main case loader
    # =========================================================

    def load_case(
        self,
        benchmark_id: str,
        *,
        budget: int = 10,
        require_method_applicable: bool = True,
    ) -> FrozenBugCase:

        if budget not in VALID_BUDGETS:

            raise ValueError(
                "Unsupported budget "
                f"{budget}. "
                "Expected one of "
                f"{sorted(VALID_BUDGETS)}."
            )

        record = (
            self.get_raw_record(
                benchmark_id
            )
        )

        if not record.get(
            "success"
        ):

            raise RuntimeError(
                "Frozen record is marked "
                "unsuccessful: "
                f"{benchmark_id}"
            )

        method_applicable = bool(
            record.get(
                "method_applicable"
            )
        )

        if (
            require_method_applicable
            and not method_applicable
        ):

            raise ValueError(
                f"{benchmark_id} is not "
                "an existing-method "
                "localization case. "
                "Applicability: "
                f"{record.get('applicability')}"
            )

        pools = (
            record.get(
                "pools",
                {}
            )
        )

        raw_pool = (
            pools.get(
                str(
                    budget
                )
            )
        )

        if raw_pool is None:

            raise KeyError(
                f"Budget {budget} "
                "not found for "
                f"{benchmark_id}"
            )

        candidate_recall = (
            record.get(
                "candidate_recall",
                {},
            ).get(
                str(
                    budget
                )
            )
        )

        return FrozenBugCase(
            benchmark_id=(
                benchmark_id
            ),
            project=(
                record[
                    "project"
                ]
            ),
            bug_id=int(
                record[
                    "bug_id"
                ]
            ),
            applicability=(
                record[
                    "applicability"
                ]
            ),
            method_applicable=(
                method_applicable
            ),
            production_method_count=int(
                record[
                    "production_method_count"
                ]
            ),
            ground_truth=(
                self._parse_ground_truth(
                    record.get(
                        "ground_truth",
                        [],
                    )
                )
            ),
            failing_tests=(
                self._parse_failing_tests(
                    record.get(
                        "failing_tests",
                        [],
                    )
                )
            ),
            failure_trace_count=int(
                record.get(
                    "failure_trace_count",
                    0,
                )
            ),
            budget=(
                budget
            ),
            candidate_recall=(
                candidate_recall
            ),
            candidates=(
                self._parse_candidates(
                    raw_pool
                )
            ),
        )

    # =========================================================
    # Convenience helpers
    # =========================================================

    def load_all_cases(
        self,
        *,
        budget: int = 10,
        only_method_applicable: bool = True,
    ) -> list[
        FrozenBugCase
    ]:

        cases = []

        for benchmark_id in (
            self.benchmark_ids(
                only_successful=True,
                only_method_applicable=(
                    only_method_applicable
                ),
            )
        ):

            cases.append(
                self.load_case(
                    benchmark_id,
                    budget=budget,
                    require_method_applicable=(
                        only_method_applicable
                    ),
                )
            )

        return cases

    @staticmethod
    def ground_truth_keys(
        case: FrozenBugCase,
    ) -> set[
        tuple[
            str,
            str,
            int,
            int,
        ]
    ]:

        return {
            (
                item.class_name,
                item.source_file,
                item.start_line,
                item.end_line,
            )
            for item
            in case.ground_truth
        }

    @staticmethod
    def candidate_key(
        candidate: FrozenCandidate,
    ) -> tuple[
        str,
        str,
        int,
        int,
    ]:

        return (
            candidate.class_name,
            candidate.source_file,
            candidate.start_line,
            candidate.end_line,
        )

    def verify_case_recall(
        self,
        case: FrozenBugCase,
    ) -> bool | None:

        if not case.method_applicable:
            return None

        gt_keys = (
            self.ground_truth_keys(
                case
            )
        )

        candidate_keys = {
            self.candidate_key(
                candidate
            )
            for candidate
            in case.candidates
        }

        return bool(
            gt_keys
            & candidate_keys
        )