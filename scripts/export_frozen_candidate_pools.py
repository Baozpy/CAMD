from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from camd.context.method_extractor import extract_java_methods
from camd.evaluation.experiment_runner import (
    Defects4JExperimentRunner,
)
from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
)
from camd.retrieval.call_chain_retriever import (
    CallChainRetriever,
)
from camd.retrieval.failure_trace_parser import (
    FailureTraceParser,
)
from camd.retrieval.program_method_retriever import (
    ProgramMethod,
    ProgramMethodRetriever,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DEFAULT_BENCHMARK = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "fse_ase_benchmark_v1.json"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "defects4j"
    / "fse_ase_frozen_candidate_pools.json"
)


TOP_N_VALUES = [
    10,
    20,
    50,
    100,
]


KNOWN_METHOD_ADDITION_CASES = {
    ("Lang", 23),
    ("Chart", 23),
}


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def normalize_class_name(
    class_name: str,
) -> str:
    if not class_name:
        return ""

    return class_name.split(
        "$",
        1,
    )[0]


def get_entries(
    benchmark,
):
    if isinstance(
        benchmark,
        list,
    ):
        return benchmark

    for key in [
        "entries",
        "bugs",
        "benchmark",
        "samples",
    ]:
        value = benchmark.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            return value

    raise RuntimeError(
        "Could not find benchmark entries."
    )


def benchmark_id_from_entry(
    entry,
):
    if entry.get(
        "benchmark_id"
    ):
        return entry[
            "benchmark_id"
        ]

    return (
        f"{entry['project']}-"
        f"{int(entry['bug_id'])}"
    )


def source_key_from_path(
    source_file: Path,
    source_root: Path,
) -> str:
    try:
        return str(
            source_file.relative_to(
                source_root
            )
        )
    except ValueError:
        return str(
            source_file
        )


def candidate_key(
    candidate: ProgramMethod,
    source_root: Path,
):
    return (
        candidate.class_name,
        source_key_from_path(
            Path(
                candidate.source_file
            ),
            source_root,
        ),
        candidate.method.start_line,
        candidate.method.end_line,
    )


def serialize_candidate_identity(
    candidate: ProgramMethod,
    source_root: Path,
):
    return {
        "class_name": (
            candidate.class_name
        ),
        "method_name": (
            candidate.method.name
        ),
        "source_file": (
            source_key_from_path(
                Path(
                    candidate.source_file
                ),
                source_root,
            )
        ),
        "start_line": (
            candidate.method.start_line
        ),
        "end_line": (
            candidate.method.end_line
        ),
        "code": (
            candidate.method.code
        ),
    }


def extract_rank_score(
    ranked_item,
):
    """
    Be tolerant to the exact RankedProgramMethod
    implementation used in the current repository.
    """

    for attr in [
        "score",
        "rank_score",
        "retrieval_score",
        "final_score",
    ]:
        value = getattr(
            ranked_item,
            attr,
            None,
        )

        if isinstance(
            value,
            (int, float),
        ):
            return float(
                value
            )

    return None


def build_ground_truth(
    runner: Defects4JExperimentRunner,
):
    source_root = (
        runner.get_source_root(
            runner.buggy_dir
        )
    )

    gt_records = []
    gt_keys = set()

    for class_name in (
        runner.get_modified_classes()
    ):
        buggy_file = (
            runner.class_to_source_file(
                checkout_dir=(
                    runner.buggy_dir
                ),
                class_name=class_name,
            )
        )

        fixed_file = (
            runner.class_to_source_file(
                checkout_dir=(
                    runner.fixed_dir
                ),
                class_name=class_name,
            )
        )

        if (
            not buggy_file.exists()
            or not fixed_file.exists()
        ):
            continue

        buggy_methods = (
            extract_java_methods(
                buggy_file
            )
        )

        gt_methods = (
            runner.get_ground_truth_methods(
                buggy_file=buggy_file,
                fixed_file=fixed_file,
                buggy_methods=buggy_methods,
            )
        )

        source_key = (
            source_key_from_path(
                buggy_file,
                source_root,
            )
        )

        for method in gt_methods:
            key = (
                class_name,
                source_key,
                method.start_line,
                method.end_line,
            )

            gt_keys.add(
                key
            )

            gt_records.append(
                {
                    "class_name": (
                        class_name
                    ),
                    "method_name": (
                        method.name
                    ),
                    "source_file": (
                        source_key
                    ),
                    "start_line": (
                        method.start_line
                    ),
                    "end_line": (
                        method.end_line
                    ),
                    "code": (
                        method.code
                    ),
                }
            )

    gt_records.sort(
        key=lambda item: (
            item[
                "class_name"
            ],
            item[
                "source_file"
            ],
            item[
                "start_line"
            ],
            item[
                "end_line"
            ],
        )
    )

    return (
        gt_records,
        gt_keys,
    )


def find_exact_stack_candidates(
    methods: list[ProgramMethod],
    traces,
):
    methods_by_class = {}

    for candidate in methods:
        class_name = (
            normalize_class_name(
                candidate.class_name
            )
        )

        methods_by_class.setdefault(
            class_name,
            [],
        ).append(
            candidate
        )

    matches = {}

    for trace_index, trace in enumerate(
        traces
    ):
        test_class = (
            normalize_class_name(
                trace.test_class
            )
        )

        production_depth = 0

        for frame_index, frame in enumerate(
            trace.frames
        ):
            frame_class = (
                normalize_class_name(
                    frame.class_name
                )
            )

            if (
                frame_class
                == test_class
            ):
                continue

            class_methods = (
                methods_by_class.get(
                    frame_class,
                    [],
                )
            )

            if not class_methods:
                continue

            line_matches = []

            if (
                frame.line_number
                is not None
            ):
                for candidate in (
                    class_methods
                ):
                    method = (
                        candidate.method
                    )

                    if (
                        method.start_line
                        <= frame.line_number
                        <= method.end_line
                    ):
                        line_matches.append(
                            candidate
                        )

            if line_matches:
                candidates = (
                    line_matches
                )
                evidence_type = (
                    "stack_line"
                )
            else:
                candidates = [
                    candidate
                    for candidate
                    in class_methods
                    if (
                        candidate.method.name
                        == frame.method_name
                    )
                ]
                evidence_type = (
                    "stack_method"
                )

            for candidate in candidates:
                key = id(
                    candidate
                )

                evidence = {
                    "candidate": (
                        candidate
                    ),
                    "depth": (
                        production_depth
                    ),
                    "trace_index": (
                        trace_index
                    ),
                    "frame_index": (
                        frame_index
                    ),
                    "evidence_type": (
                        evidence_type
                    ),
                }

                previous = (
                    matches.get(
                        key
                    )
                )

                if previous is None:
                    matches[
                        key
                    ] = evidence
                    continue

                previous_order = (
                    previous["depth"],
                    previous[
                        "trace_index"
                    ],
                    previous[
                        "frame_index"
                    ],
                )

                current_order = (
                    evidence["depth"],
                    evidence[
                        "trace_index"
                    ],
                    evidence[
                        "frame_index"
                    ],
                )

                if (
                    current_order
                    < previous_order
                ):
                    matches[
                        key
                    ] = evidence

            production_depth += 1

    return matches


def make_base_records(
    ranked,
    source_root: Path,
):
    """
    Convert base ranking into a stable ordered map.
    """

    result = {}

    for rank, ranked_item in enumerate(
        ranked,
        start=1,
    ):
        candidate = (
            ranked_item.candidate
        )

        key = candidate_key(
            candidate,
            source_root,
        )

        if key in result:
            continue

        result[
            key
        ] = {
            "candidate": (
                candidate
            ),
            "base_rank": (
                rank
            ),
            "base_score": (
                extract_rank_score(
                    ranked_item
                )
            ),
        }

    return result


def make_stack_records(
    stack_matches,
    source_root: Path,
):
    records = []

    for evidence in (
        stack_matches.values()
    ):
        candidate = (
            evidence[
                "candidate"
            ]
        )

        records.append(
            {
                "key": (
                    candidate_key(
                        candidate,
                        source_root,
                    )
                ),
                "candidate": (
                    candidate
                ),
                "stack_depth": (
                    evidence[
                        "depth"
                    ]
                ),
                "stack_evidence_type": (
                    evidence[
                        "evidence_type"
                    ]
                ),
                "trace_index": (
                    evidence[
                        "trace_index"
                    ]
                ),
                "frame_index": (
                    evidence[
                        "frame_index"
                    ]
                ),
            }
        )

    records.sort(
        key=lambda item: (
            item[
                "stack_depth"
            ],
            item[
                "trace_index"
            ],
            item[
                "frame_index"
            ],
            item[
                "candidate"
            ].class_name,
            item[
                "candidate"
            ].method.start_line,
            item[
                "candidate"
            ].method.end_line,
        )
    )

    return records


def make_call_records(
    call_candidates,
    source_root: Path,
):
    records = []

    for item in call_candidates:
        candidate = (
            item.candidate
        )

        records.append(
            {
                "key": (
                    candidate_key(
                        candidate,
                        source_root,
                    )
                ),
                "candidate": (
                    candidate
                ),
                "call_depth": (
                    item.depth
                ),
                "runtime_class": (
                    item.runtime_class
                ),
                "call_evidence_type": (
                    item.evidence_type
                ),
                "originating_test": (
                    item.originating_test
                ),
            }
        )

    records.sort(
        key=lambda item: (
            item[
                "call_depth"
            ],
            item[
                "candidate"
            ].class_name,
            item[
                "candidate"
            ].method.start_line,
            item[
                "candidate"
            ].method.end_line,
            item[
                "candidate"
            ].method.name,
        )
    )

    return records


def build_pool(
    n: int,
    ranked,
    base_records,
    stack_records,
    call_records,
    source_root: Path,
    gt_keys,
):
    """
    Frozen deterministic order:

    1. base Top-N
    2. stack-only candidates
    3. call-only candidates

    Existing candidates are enriched with evidence
    instead of duplicated.
    """

    ordered_keys = []

    metadata = {}

    # -----------------------------------------------------
    # 1. Base Top-N
    # -----------------------------------------------------

    for ranked_item in (
        ranked[:n]
    ):
        candidate = (
            ranked_item.candidate
        )

        key = candidate_key(
            candidate,
            source_root,
        )

        if key not in metadata:
            metadata[
                key
            ] = {
                "candidate": (
                    candidate
                ),
                "base_rank": (
                    base_records[
                        key
                    ][
                        "base_rank"
                    ]
                ),
                "base_score": (
                    base_records[
                        key
                    ][
                        "base_score"
                    ]
                ),
                "from_base": True,
                "from_stack": False,
                "from_call": False,
                "stack_depth": None,
                "stack_evidence_type": None,
                "call_depth": None,
                "runtime_class": None,
                "call_evidence_type": None,
                "originating_test": None,
            }

            ordered_keys.append(
                key
            )

    # -----------------------------------------------------
    # 2. Add / enrich stack candidates
    # -----------------------------------------------------

    for item in stack_records:
        key = item[
            "key"
        ]

        if key not in metadata:
            candidate = (
                item[
                    "candidate"
                ]
            )

            base_info = (
                base_records.get(
                    key
                )
            )

            metadata[
                key
            ] = {
                "candidate": (
                    candidate
                ),
                "base_rank": (
                    base_info[
                        "base_rank"
                    ]
                    if base_info
                    else None
                ),
                "base_score": (
                    base_info[
                        "base_score"
                    ]
                    if base_info
                    else None
                ),
                "from_base": False,
                "from_stack": True,
                "from_call": False,
                "stack_depth": (
                    item[
                        "stack_depth"
                    ]
                ),
                "stack_evidence_type": (
                    item[
                        "stack_evidence_type"
                    ]
                ),
                "call_depth": None,
                "runtime_class": None,
                "call_evidence_type": None,
                "originating_test": None,
            }

            ordered_keys.append(
                key
            )

        else:
            metadata[
                key
            ][
                "from_stack"
            ] = True

            metadata[
                key
            ][
                "stack_depth"
            ] = item[
                "stack_depth"
            ]

            metadata[
                key
            ][
                "stack_evidence_type"
            ] = item[
                "stack_evidence_type"
            ]

    # -----------------------------------------------------
    # 3. Add / enrich call-chain candidates
    # -----------------------------------------------------

    for item in call_records:
        key = item[
            "key"
        ]

        if key not in metadata:
            candidate = (
                item[
                    "candidate"
                ]
            )

            base_info = (
                base_records.get(
                    key
                )
            )

            metadata[
                key
            ] = {
                "candidate": (
                    candidate
                ),
                "base_rank": (
                    base_info[
                        "base_rank"
                    ]
                    if base_info
                    else None
                ),
                "base_score": (
                    base_info[
                        "base_score"
                    ]
                    if base_info
                    else None
                ),
                "from_base": False,
                "from_stack": False,
                "from_call": True,
                "stack_depth": None,
                "stack_evidence_type": None,
                "call_depth": (
                    item[
                        "call_depth"
                    ]
                ),
                "runtime_class": (
                    item[
                        "runtime_class"
                    ]
                ),
                "call_evidence_type": (
                    item[
                        "call_evidence_type"
                    ]
                ),
                "originating_test": (
                    item[
                        "originating_test"
                    ]
                ),
            }

            ordered_keys.append(
                key
            )

        else:
            metadata[
                key
            ][
                "from_call"
            ] = True

            existing_depth = (
                metadata[
                    key
                ][
                    "call_depth"
                ]
            )

            if (
                existing_depth is None
                or item[
                    "call_depth"
                ]
                < existing_depth
            ):
                metadata[
                    key
                ][
                    "call_depth"
                ] = item[
                    "call_depth"
                ]

                metadata[
                    key
                ][
                    "runtime_class"
                ] = item[
                    "runtime_class"
                ]

                metadata[
                    key
                ][
                    "call_evidence_type"
                ] = item[
                    "call_evidence_type"
                ]

                metadata[
                    key
                ][
                    "originating_test"
                ] = item[
                    "originating_test"
                ]

    # -----------------------------------------------------
    # Serialize
    # -----------------------------------------------------

    pool = []

    for pool_position, key in enumerate(
        ordered_keys,
        start=1,
    ):
        item = metadata[
            key
        ]

        serialized = (
            serialize_candidate_identity(
                item[
                    "candidate"
                ],
                source_root,
            )
        )

        serialized.update(
            {
                "pool_position": (
                    pool_position
                ),
                "base_rank": (
                    item[
                        "base_rank"
                    ]
                ),
                "base_score": (
                    item[
                        "base_score"
                    ]
                ),
                "from_base": (
                    item[
                        "from_base"
                    ]
                ),
                "from_stack": (
                    item[
                        "from_stack"
                    ]
                ),
                "from_call": (
                    item[
                        "from_call"
                    ]
                ),
                "stack_depth": (
                    item[
                        "stack_depth"
                    ]
                ),
                "stack_evidence_type": (
                    item[
                        "stack_evidence_type"
                    ]
                ),
                "call_depth": (
                    item[
                        "call_depth"
                    ]
                ),
                "runtime_class": (
                    item[
                        "runtime_class"
                    ]
                ),
                "call_evidence_type": (
                    item[
                        "call_evidence_type"
                    ]
                ),
                "originating_test": (
                    item[
                        "originating_test"
                    ]
                ),
                "is_ground_truth": (
                    key
                    in gt_keys
                ),
            }
        )

        pool.append(
            serialized
        )

    return pool


def parse_args():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=(
            DEFAULT_BENCHMARK
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--test-helper-depth",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--production-depth",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--keep-checkouts",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = (
        parse_args()
    )

    benchmark = (
        load_json(
            args.benchmark
        )
    )

    entries = (
        get_entries(
            benchmark
        )
    )

    records = []

    print()
    print("=" * 100)
    print(
        "CAMD-R Frozen Candidate Pool Export"
    )
    print("=" * 100)

    print(
        f"Benchmark: {args.benchmark}"
    )

    print(
        f"Selected bugs: {len(entries)}"
    )

    print(
        "Frozen configuration:"
    )

    print(
        "  Base retriever: v1"
    )

    print(
        "  Exact stack: enabled"
    )

    print(
        "  Call chain: enabled"
    )

    print(
        "  Test helper depth: "
        f"{args.test_helper_depth}"
    )

    print(
        "  Production depth: "
        f"{args.production_depth}"
    )

    for index, entry in enumerate(
        entries,
        start=1,
    ):
        project = (
            entry[
                "project"
            ]
        )

        bug_id = int(
            entry[
                "bug_id"
            ]
        )

        benchmark_id = (
            benchmark_id_from_entry(
                entry
            )
        )

        print()
        print("=" * 100)

        print(
            f"[{index}/{len(entries)}] "
            f"{benchmark_id}"
        )

        print("=" * 100)

        runner = (
            Defects4JExperimentRunner(
                project_root=(
                    PROJECT_ROOT
                ),
                project=project,
                bug_id=bug_id,
                top_k=5,
            )
        )

        try:
            runner.prepare_checkouts()

            source_root = (
                runner.get_source_root(
                    runner.buggy_dir
                )
            )

            is_method_addition = (
                (
                    project,
                    bug_id,
                )
                in
                KNOWN_METHOD_ADDITION_CASES
            )

            (
                gt_records,
                gt_keys,
            ) = build_ground_truth(
                runner
            )

            if (
                not gt_keys
                and not is_method_addition
            ):
                raise RuntimeError(
                    "No ground-truth methods "
                    "found for expected "
                    "existing-method bug."
                )

            failing_tests = (
                FailingTestExtractor(
                    checkout_dir=(
                        runner.buggy_dir
                    )
                ).extract()
            )

            failing_names = [
                test.full_name
                for test
                in failing_tests
            ]

            failing_text = "\n".join(
                (
                    test.code
                    or test.full_name
                )
                for test
                in failing_tests
            )

            retriever = (
                ProgramMethodRetriever(
                    project=project,
                    bug_id=bug_id,
                    use_stack_trace=False,
                )
            )

            program_methods = (
                retriever
                .extract_program_methods(
                    source_root
                )
            )

            ranked = (
                retriever.rank(
                    methods=(
                        program_methods
                    ),
                    failing_test_names=(
                        failing_names
                    ),
                    failing_test_text=(
                        failing_text
                    ),
                    failure_traces=[],
                )
            )

            base_records = (
                make_base_records(
                    ranked,
                    source_root,
                )
            )

            traces = (
                FailureTraceParser()
                .parse_file(
                    runner.buggy_dir
                    / "failing_tests"
                )
            )

            stack_matches = (
                find_exact_stack_candidates(
                    methods=(
                        program_methods
                    ),
                    traces=(
                        traces
                    ),
                )
            )

            stack_records = (
                make_stack_records(
                    stack_matches,
                    source_root,
                )
            )

            call_retriever = (
                CallChainRetriever(
                    max_test_helper_depth=(
                        args
                        .test_helper_depth
                    ),
                    max_production_depth=(
                        args
                        .production_depth
                    ),
                )
            )

            call_candidates = (
                call_retriever.retrieve(
                    program_methods=(
                        program_methods
                    ),
                    failing_tests=(
                        failing_tests
                    ),
                )
            )

            call_records = (
                make_call_records(
                    call_candidates,
                    source_root,
                )
            )

            pools = {}

            recall = {}

            for n in TOP_N_VALUES:
                pool = (
                    build_pool(
                        n=n,
                        ranked=ranked,
                        base_records=(
                            base_records
                        ),
                        stack_records=(
                            stack_records
                        ),
                        call_records=(
                            call_records
                        ),
                        source_root=(
                            source_root
                        ),
                        gt_keys=(
                            gt_keys
                        ),
                    )
                )

                pools[
                    str(n)
                ] = pool

                if is_method_addition:
                    recall[
                        str(n)
                    ] = None
                else:
                    recall[
                        str(n)
                    ] = any(
                        item[
                            "is_ground_truth"
                        ]
                        for item in pool
                    )

            record = {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": (
                    project
                ),
                "bug_id": (
                    bug_id
                ),
                "applicability": (
                    "method_addition"
                    if is_method_addition
                    else "existing_method"
                ),
                "method_applicable": (
                    not is_method_addition
                ),
                "production_method_count": (
                    len(
                        program_methods
                    )
                ),
                "ground_truth": (
                    gt_records
                ),
                "failing_tests": [
                    {
                        "full_name": (
                            test.full_name
                        ),
                        "class_name": (
                            test.class_name
                        ),
                        "method_name": (
                            test.method_name
                        ),
                        "source_file": (
                            str(
                                test.source_file
                            )
                            if test.source_file
                            else None
                        ),
                        "start_line": (
                            test.start_line
                        ),
                        "end_line": (
                            test.end_line
                        ),
                        "code": (
                            test.code
                        ),
                    }
                    for test
                    in failing_tests
                ],
                "failure_trace_count": (
                    len(
                        traces
                    )
                ),
                "candidate_recall": (
                    recall
                ),
                "pool_sizes": {
                    str(n): len(
                        pools[
                            str(n)
                        ]
                    )
                    for n
                    in TOP_N_VALUES
                },
                "pools": (
                    pools
                ),
                "success": True,
            }

            records.append(
                record
            )

            print(
                "Production methods: "
                f"{len(program_methods)}"
            )

            print(
                "Applicability: "
                f"{record['applicability']}"
            )

            print(
                "GT methods: "
                f"{len(gt_records)}"
            )

            print(
                "Stack candidates: "
                f"{len(stack_records)}"
            )

            print(
                "Call candidates: "
                f"{len(call_records)}"
            )

            for n in TOP_N_VALUES:
                print(
                    f"@{n:<3} "
                    f"pool="
                    f"{record['pool_sizes'][str(n)]} "
                    f"GT="
                    f"{record['candidate_recall'][str(n)]}"
                )

        except Exception as exc:
            records.append(
                {
                    "benchmark_id": (
                        benchmark_id
                    ),
                    "project": (
                        project
                    ),
                    "bug_id": (
                        bug_id
                    ),
                    "success": False,
                    "error": (
                        str(
                            exc
                        )
                    ),
                }
            )

            print(
                f"FAILED: {exc}"
            )

        payload = {
            "artifact": (
                "CAMD-R frozen candidate pools"
            ),
            "benchmark": (
                str(
                    args.benchmark
                )
            ),
            "frozen_configuration": {
                "base_retriever": (
                    "program_method_retriever_v1"
                ),
                "exact_stack_augmentation": (
                    True
                ),
                "call_chain_augmentation": (
                    True
                ),
                "test_helper_depth": (
                    args
                    .test_helper_depth
                ),
                "production_depth": (
                    args
                    .production_depth
                ),
                "top_n_values": (
                    TOP_N_VALUES
                ),
                "pool_order": [
                    "base_top_n",
                    "stack_only",
                    "call_only",
                ],
            },
            "records": (
                records
            ),
        }

        save_json(
            args.output,
            payload,
        )

        if not args.keep_checkouts:
            if (
                runner.buggy_dir.exists()
            ):
                shutil.rmtree(
                    runner.buggy_dir
                )

            if (
                runner.fixed_dir.exists()
            ):
                shutil.rmtree(
                    runner.fixed_dir
                )

    successful = [
        record
        for record in records
        if record.get(
            "success"
        )
    ]

    applicable = [
        record
        for record in successful
        if record.get(
            "method_applicable"
        )
    ]

    method_additions = [
        record
        for record in successful
        if (
            record.get(
                "applicability"
            )
            == "method_addition"
        )
    ]

    summary = {
        "selected_bugs": (
            len(records)
        ),
        "successful_bugs": (
            len(successful)
        ),
        "existing_method_bugs": (
            len(applicable)
        ),
        "method_addition_bugs": (
            len(method_additions)
        ),
        "failed_bugs": (
            len(records)
            - len(successful)
        ),
        "recall": {},
    }

    for n in TOP_N_VALUES:
        hits = sum(
            bool(
                record[
                    "candidate_recall"
                ][str(n)]
            )
            for record
            in applicable
        )

        summary[
            "recall"
        ][str(n)] = {
            "hits": (
                hits
            ),
            "total": (
                len(applicable)
            ),
            "recall": (
                hits
                / len(applicable)
                if applicable
                else None
            ),
        }

    final_payload = {
        "artifact": (
            "CAMD-R frozen candidate pools"
        ),
        "benchmark": (
            str(
                args.benchmark
            )
        ),
        "frozen_configuration": {
            "base_retriever": (
                "program_method_retriever_v1"
            ),
            "exact_stack_augmentation": (
                True
            ),
            "call_chain_augmentation": (
                True
            ),
            "test_helper_depth": (
                args
                .test_helper_depth
            ),
            "production_depth": (
                args
                .production_depth
            ),
            "top_n_values": (
                TOP_N_VALUES
            ),
            "pool_order": [
                "base_top_n",
                "stack_only",
                "call_only",
            ],
        },
        "summary": (
            summary
        ),
        "records": (
            records
        ),
    }

    save_json(
        args.output,
        final_payload,
    )

    print()
    print("=" * 100)
    print(
        "Frozen Candidate Pool Export Summary"
    )
    print("=" * 100)

    print(
        "Selected: "
        f"{summary['selected_bugs']}"
    )

    print(
        "Successful: "
        f"{summary['successful_bugs']}"
    )

    print(
        "Existing-method: "
        f"{summary['existing_method_bugs']}"
    )

    print(
        "Method-addition: "
        f"{summary['method_addition_bugs']}"
    )

    print(
        "Failed: "
        f"{summary['failed_bugs']}"
    )

    for n in TOP_N_VALUES:
        info = (
            summary[
                "recall"
            ][str(n)]
        )

        print(
            f"R@{n:<3} "
            f"{info['recall']:.4f} "
            f"({info['hits']}/"
            f"{info['total']})"
        )

    print()
    print(
        f"Saved:\n{args.output}"
    )


if __name__ == "__main__":
    main()