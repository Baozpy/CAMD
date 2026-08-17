from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from camd.context.method_extractor import (
    extract_java_methods,
)
from camd.evaluation.experiment_runner import (
    Defects4JExperimentRunner,
)
from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
)
from camd.retrieval.failure_trace_parser import (
    FailureTraceParser,
)
from camd.retrieval.program_method_retriever import (
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
    / "fse_ase_retrieval_v1.json"
)


TOP_K_VALUES = [
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
]


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def save_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate non-oracle "
            "program-wide candidate retrieval."
        )
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--max-bugs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--keep-checkouts",
        action="store_true",
    )

    parser.add_argument(
        "--use-stack-trace",
        action="store_true",
        help=(
            "Enable stack-trace retrieval "
            "signal (Retriever v2a)."
        ),
    )

    parser.add_argument(
        "--stack-weight",
        type=float,
        default=0.50,
        help=(
            "Weight applied to the "
            "stack-trace signal."
        ),
    )

    return parser.parse_args()


def build_failing_test_text(
    checkout_dir: Path,
) -> tuple[
    list[str],
    str,
]:

    extractor = (
        FailingTestExtractor(
            checkout_dir=checkout_dir
        )
    )

    tests = (
        extractor.extract()
    )

    names = []
    text_parts = []

    for test in tests:

        names.append(
            test.full_name
        )

        text_parts.append(
            test.full_name
        )

        if test.code:
            text_parts.append(
                test.code
            )

        text_parts.append("")

    return (
        names,
        "\n".join(
            text_parts
        ),
    )


def load_failure_traces(
    checkout_dir: Path,
):

    failing_file = (
        checkout_dir
        / "failing_tests"
    )

    parser = (
        FailureTraceParser()
    )

    return parser.parse_file(
        failing_file
    )


def build_ground_truth_keys(
    runner: Defects4JExperimentRunner,
    modified_classes: list[str],
) -> set[
    tuple[
        str,
        str,
        int,
        int,
    ]
]:

    gt_keys = set()

    source_root = (
        runner.get_source_root(
            runner.buggy_dir
        )
    )

    for class_name in (
        modified_classes
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

        try:

            relative_path = (
                buggy_file.relative_to(
                    source_root
                )
            )

            source_file_key = str(
                relative_path
            )

        except ValueError:

            source_file_key = str(
                buggy_file
            )

        for method in gt_methods:

            gt_keys.add(
                (
                    class_name,
                    source_file_key,
                    method.start_line,
                    method.end_line,
                )
            )

    return gt_keys


def candidate_key(
    item,
    source_root: Path,
):

    candidate = (
        item.candidate
    )

    source_file = Path(
        candidate.source_file
    )

    try:

        relative = (
            source_file.relative_to(
                source_root
            )
        )

        source_key = str(
            relative
        )

    except ValueError:

        source_key = str(
            source_file
        )

    return (
        candidate.class_name,
        source_key,
        candidate.method.start_line,
        candidate.method.end_line,
    )


def find_best_gt_rank(
    ranked,
    gt_keys,
    source_root: Path,
):

    for rank, item in enumerate(
        ranked,
        start=1,
    ):

        key = candidate_key(
            item,
            source_root=(
                source_root
            ),
        )

        if key in gt_keys:

            return rank

    return None


def summarize(
    records: list[dict],
) -> dict:

    valid_records = [
        record
        for record in records
        if record.get(
            "success"
        )
    ]

    total = len(
        valid_records
    )

    overall = {
        "evaluated_bugs": total,
        "recall": {},
    }

    if total == 0:
        return overall

    for k in TOP_K_VALUES:

        hits = sum(
            1
            for record
            in valid_records
            if (
                record.get(
                    "best_gt_rank"
                )
                is not None
                and record[
                    "best_gt_rank"
                ] <= k
            )
        )

        overall[
            "recall"
        ][
            f"top_{k}"
        ] = (
            hits / total
        )

    project_names = sorted(
        {
            record["project"]
            for record
            in valid_records
        }
    )

    by_project = {}

    for project in project_names:

        project_records = [
            record
            for record
            in valid_records
            if (
                record["project"]
                == project
            )
        ]

        project_total = len(
            project_records
        )

        metrics = {
            "evaluated_bugs": (
                project_total
            ),
            "recall": {},
        }

        for k in TOP_K_VALUES:

            hits = sum(
                1
                for record
                in project_records
                if (
                    record.get(
                        "best_gt_rank"
                    )
                    is not None
                    and record[
                        "best_gt_rank"
                    ] <= k
                )
            )

            metrics[
                "recall"
            ][
                f"top_{k}"
            ] = (
                hits
                / project_total
                if project_total
                else 0.0
            )

        by_project[
            project
        ] = metrics

    overall[
        "by_project"
    ] = by_project

    return overall


def main() -> None:

    args = parse_args()

    benchmark = load_json(
        args.benchmark
    )

    all_entries = benchmark[
        "entries"
    ]

    if args.start_index < 1:

        raise ValueError(
            "--start-index must be >= 1."
        )

    selected_entries = (
        all_entries[
            args.start_index - 1:
        ]
    )

    if args.max_bugs is not None:

        if args.max_bugs <= 0:

            raise ValueError(
                "--max-bugs must be > 0."
            )

        selected_entries = (
            selected_entries[
                :args.max_bugs
            ]
        )

    existing_records = []

    if args.output.exists():

        existing = load_json(
            args.output
        )

        existing_records = (
            existing.get(
                "records",
                []
            )
        )

    records_by_id = {
        record[
            "benchmark_id"
        ]: record
        for record
        in existing_records
        if (
            isinstance(
                record,
                dict,
            )
            and "benchmark_id"
            in record
        )
    }

    remaining = [
        entry
        for entry
        in selected_entries
        if entry[
            "benchmark_id"
        ]
        not in records_by_id
    ]

    mode = (
        "v2a_stack"
        if args.use_stack_trace
        else "v1"
    )

    print()
    print("=" * 100)
    print(
        "CAMD Program-Wide "
        "Candidate Retrieval"
    )
    print("=" * 100)

    print(
        f"Mode: {mode}"
    )

    print(
        "Use stack trace: "
        f"{args.use_stack_trace}"
    )

    if args.use_stack_trace:

        print(
            "Stack weight: "
            f"{args.stack_weight}"
        )

    print(
        "Selected entries: "
        f"{len(selected_entries)}"
    )

    print(
        "Already evaluated: "
        f"{len(selected_entries) - len(remaining)}"
    )

    print(
        f"Remaining: "
        f"{len(remaining)}"
    )

    for index, entry in enumerate(
        remaining,
        start=1,
    ):

        project = entry[
            "project"
        ]

        bug_id = int(
            entry[
                "bug_id"
            ]
        )

        benchmark_id = (
            entry[
                "benchmark_id"
            ]
        )

        print()
        print("=" * 100)

        print(
            f"[{index}/"
            f"{len(remaining)}] "
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

            modified_classes = (
                runner.get_modified_classes()
            )

            gt_keys = (
                build_ground_truth_keys(
                    runner=runner,
                    modified_classes=(
                        modified_classes
                    ),
                )
            )

            if not gt_keys:

                raise RuntimeError(
                    "No ground-truth "
                    "methods found."
                )

            (
                failing_names,
                failing_text,
            ) = (
                build_failing_test_text(
                    runner.buggy_dir
                )
            )

            failure_traces = []

            if args.use_stack_trace:

                failure_traces = (
                    load_failure_traces(
                        runner.buggy_dir
                    )
                )

            retriever = (
                ProgramMethodRetriever(
                    project=project,
                    bug_id=bug_id,
                    use_stack_trace=(
                        args.use_stack_trace
                    ),
                    stack_weight=(
                        args.stack_weight
                    ),
                )
            )

            program_methods = (
                retriever
                .extract_program_methods(
                    source_root
                )
            )

            if not program_methods:

                raise RuntimeError(
                    "No production methods "
                    "were extracted."
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
                    failure_traces=(
                        failure_traces
                    ),
                )
            )

            best_gt_rank = (
                find_best_gt_rank(
                    ranked=ranked,
                    gt_keys=gt_keys,
                    source_root=(
                        source_root
                    ),
                )
            )

            stack_candidate_count = sum(
                1
                for item in ranked
                if item.stack_score > 0
            )

            max_stack_score = max(
                (
                    item.stack_score
                    for item in ranked
                ),
                default=0.0,
            )

            top_candidates = []

            for rank, item in enumerate(
                ranked[:300],
                start=1,
            ):

                candidate = (
                    item.candidate
                )

                top_candidates.append(
                    {
                        "rank": rank,
                        "class_name": (
                            candidate.class_name
                        ),
                        "source_file": (
                            candidate.source_file
                        ),
                        "method": (
                            candidate.method.name
                        ),
                        "start_line": (
                            candidate
                            .method
                            .start_line
                        ),
                        "end_line": (
                            candidate
                            .method
                            .end_line
                        ),
                        "score": (
                            item.score
                        ),
                        "base_score": (
                            item.base_score
                        ),
                        "signals": {
                            "direct_method_reference": (
                                item
                                .direct_method_reference
                            ),
                            "class_reference": (
                                item
                                .class_reference
                            ),
                            "name_overlap": (
                                item
                                .name_overlap
                            ),
                            "test_name_overlap": (
                                item
                                .test_name_overlap
                            ),
                            "lexical_overlap": (
                                item
                                .lexical_overlap
                            ),
                            "stack_score": (
                                item
                                .stack_score
                            ),
                            "stack_exact_match": (
                                item
                                .stack_exact_match
                            ),
                            "stack_class_match": (
                                item
                                .stack_class_match
                            ),
                            "stack_depth": (
                                item
                                .stack_depth
                            ),
                        },
                    }
                )

            record = {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": project,
                "bug_id": bug_id,
                "success": True,
                "retriever_mode": mode,
                "use_stack_trace": (
                    args.use_stack_trace
                ),
                "stack_weight": (
                    args.stack_weight
                    if args.use_stack_trace
                    else 0.0
                ),
                "production_method_count": (
                    len(
                        program_methods
                    )
                ),
                "ground_truth_method_count": (
                    len(
                        gt_keys
                    )
                ),
                "failing_tests": (
                    failing_names
                ),
                "failure_trace_count": (
                    len(
                        failure_traces
                    )
                ),
                "stack_candidate_count": (
                    stack_candidate_count
                ),
                "max_stack_score": (
                    max_stack_score
                ),
                "best_gt_rank": (
                    best_gt_rank
                ),
                "top_candidates": (
                    top_candidates
                ),
            }

            for k in TOP_K_VALUES:

                record[
                    f"recall_at_{k}"
                ] = (
                    best_gt_rank
                    is not None
                    and best_gt_rank <= k
                )

            print(
                "Production methods: "
                f"{len(program_methods)}"
            )

            print(
                "GT methods: "
                f"{len(gt_keys)}"
            )

            if args.use_stack_trace:

                print(
                    "Failure traces: "
                    f"{len(failure_traces)}"
                )

                print(
                    "Candidates with stack signal: "
                    f"{stack_candidate_count}"
                )

                print(
                    "Max stack score: "
                    f"{max_stack_score:.4f}"
                )

            print(
                "Best GT rank: "
                f"{best_gt_rank}"
            )

            recall_text = " ".join(
                f"@{k}="
                f"{record[f'recall_at_{k}']}"
                for k in [
                    10,
                    20,
                    50,
                    100,
                ]
            )

            print(
                f"Recall: {recall_text}"
            )

        except Exception as exc:

            record = {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": project,
                "bug_id": bug_id,
                "success": False,
                "retriever_mode": mode,
                "error": str(
                    exc
                ),
            }

            print(
                f"FAILED: {exc}"
            )

        records_by_id[
            benchmark_id
        ] = record

        ordered_records = [
            records_by_id[
                item[
                    "benchmark_id"
                ]
            ]
            for item
            in all_entries
            if item[
                "benchmark_id"
            ]
            in records_by_id
        ]

        payload = {
            "benchmark": str(
                args.benchmark
            ),
            "retriever_mode": mode,
            "use_stack_trace": (
                args.use_stack_trace
            ),
            "stack_weight": (
                args.stack_weight
                if args.use_stack_trace
                else 0.0
            ),
            "top_k_values": (
                TOP_K_VALUES
            ),
            "records": (
                ordered_records
            ),
            "summary": (
                summarize(
                    ordered_records
                )
            ),
        }

        save_json(
            args.output,
            payload,
        )

        if not args.keep_checkouts:

            if runner.buggy_dir.exists():

                shutil.rmtree(
                    runner.buggy_dir
                )

            if runner.fixed_dir.exists():

                shutil.rmtree(
                    runner.fixed_dir
                )

    final_records = [
        records_by_id[
            item[
                "benchmark_id"
            ]
        ]
        for item
        in all_entries
        if item[
            "benchmark_id"
        ]
        in records_by_id
    ]

    final_summary = summarize(
        final_records
    )

    save_json(
        args.output,
        {
            "benchmark": str(
                args.benchmark
            ),
            "retriever_mode": mode,
            "use_stack_trace": (
                args.use_stack_trace
            ),
            "stack_weight": (
                args.stack_weight
                if args.use_stack_trace
                else 0.0
            ),
            "top_k_values": (
                TOP_K_VALUES
            ),
            "records": (
                final_records
            ),
            "summary": (
                final_summary
            ),
        },
    )

    print()
    print("=" * 100)
    print(
        "Retrieval Summary"
    )
    print("=" * 100)

    print(
        f"Mode: {mode}"
    )

    print(
        "Evaluated bugs: "
        f"{final_summary.get('evaluated_bugs', 0)}"
    )

    for k in TOP_K_VALUES:

        value = (
            final_summary
            .get(
                "recall",
                {}
            )
            .get(
                f"top_{k}",
                0.0,
            )
        )

        print(
            f"Recall@{k}: "
            f"{value:.4f}"
        )

    print()

    by_project = (
        final_summary.get(
            "by_project",
            {}
        )
    )

    for project in [
        "Lang",
        "Math",
        "Chart",
        "Time",
        "Mockito",
    ]:

        if project not in by_project:
            continue

        info = (
            by_project[
                project
            ]
        )

        print(
            f"{project}: "
            f"n={info['evaluated_bugs']}"
        )

        for k in [
            10,
            20,
            50,
            100,
        ]:

            value = (
                info[
                    "recall"
                ].get(
                    f"top_{k}",
                    0.0,
                )
            )

            print(
                f"  Recall@{k}: "
                f"{value:.4f}"
            )

    print()
    print(
        f"Saved:\n{args.output}"
    )


if __name__ == "__main__":
    main()