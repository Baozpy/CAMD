from __future__ import annotations

import argparse
import json
import shutil
import statistics
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
    / "fse_ase_retrieval_dev_v1.json"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "defects4j"
    / "fse_ase_retrieval_dev_stack_augment_results.json"
)


TOP_N_VALUES = [
    10,
    20,
    50,
    100,
]


# =========================================================
# I/O
# =========================================================

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
            "Evaluate exact stack-trace "
            "candidate augmentation on top "
            "of Retriever v1."
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
        "--keep-checkouts",
        action="store_true",
    )

    parser.add_argument(
        "--max-bugs",
        type=int,
        default=None,
    )

    return parser.parse_args()


# =========================================================
# Failing test
# =========================================================

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

    parser = (
        FailureTraceParser()
    )

    return parser.parse_file(
        checkout_dir
        / "failing_tests"
    )


# =========================================================
# Ground truth
# =========================================================

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

            relative = (
                buggy_file.relative_to(
                    source_root
                )
            )

            source_key = str(
                relative
            )

        except ValueError:

            source_key = str(
                buggy_file
            )

        for method in gt_methods:

            gt_keys.add(
                (
                    class_name,
                    source_key,
                    method.start_line,
                    method.end_line,
                )
            )

    return gt_keys


# =========================================================
# Candidate identity
# =========================================================

def candidate_key(
    candidate: ProgramMethod,
    source_root: Path,
) -> tuple[
    str,
    str,
    int,
    int,
]:

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


def normalize_class_name(
    class_name: str,
) -> str:

    if not class_name:
        return ""

    return class_name.split(
        "$",
        1,
    )[0]


# =========================================================
# Exact stack matching
# =========================================================

def find_exact_stack_candidates(
    methods: list[ProgramMethod],
    traces,
) -> dict[
    tuple[
        str,
        str,
        int,
        int,
    ],
    dict,
]:

    """
    Return program methods that are supported
    by exact stack evidence.

    Exact evidence means either:

    1. Same class and stack source line falls
       inside the candidate method; or

    2. Same class + same method name when
       usable line alignment is unavailable.

    Same-class-only evidence is NOT used.
    """

    matches = {}

    methods_by_class: dict[
        str,
        list[ProgramMethod],
    ] = {}

    for candidate in methods:

        normalized_class = (
            normalize_class_name(
                candidate.class_name
            )
        )

        methods_by_class.setdefault(
            normalized_class,
            [],
        ).append(
            candidate
        )

    for trace in traces:

        test_class = (
            normalize_class_name(
                trace.test_class
            )
        )

        production_depth = 0

        for frame in trace.frames:

            frame_class = (
                normalize_class_name(
                    frame.class_name
                )
            )

            # Ignore the failing test class.
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
                        is not None
                        and method.end_line
                        is not None
                        and (
                            method.start_line
                            <= frame.line_number
                            <= method.end_line
                        )
                    ):

                        line_matches.append(
                            candidate
                        )

            # Strongest evidence:
            # class + physical source line.
            if line_matches:

                for candidate in (
                    line_matches
                ):

                    key = (
                        id(candidate)
                    )

                    previous = (
                        matches.get(
                            key
                        )
                    )

                    evidence = {
                        "candidate": (
                            candidate
                        ),
                        "match_type": (
                            "line"
                        ),
                        "stack_depth": (
                            production_depth
                        ),
                        "frame_class": (
                            frame.class_name
                        ),
                        "frame_method": (
                            frame.method_name
                        ),
                        "frame_file": (
                            frame.file_name
                        ),
                        "frame_line": (
                            frame.line_number
                        ),
                    }

                    if (
                        previous is None
                        or production_depth
                        <
                        previous[
                            "stack_depth"
                        ]
                    ):

                        matches[
                            key
                        ] = evidence

            else:

                # Fallback only when no physical
                # line mapping is available.
                for candidate in (
                    class_methods
                ):

                    if (
                        candidate.method.name
                        != frame.method_name
                    ):
                        continue

                    key = id(
                        candidate
                    )

                    previous = (
                        matches.get(
                            key
                        )
                    )

                    evidence = {
                        "candidate": (
                            candidate
                        ),
                        "match_type": (
                            "method_name"
                        ),
                        "stack_depth": (
                            production_depth
                        ),
                        "frame_class": (
                            frame.class_name
                        ),
                        "frame_method": (
                            frame.method_name
                        ),
                        "frame_file": (
                            frame.file_name
                        ),
                        "frame_line": (
                            frame.line_number
                        ),
                    }

                    if (
                        previous is None
                        or production_depth
                        <
                        previous[
                            "stack_depth"
                        ]
                    ):

                        matches[
                            key
                        ] = evidence

            production_depth += 1

    return matches


# =========================================================
# Evaluation
# =========================================================

def evaluate_one_bug(
    project: str,
    bug_id: int,
):

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
            "No ground-truth methods found."
        )

    (
        failing_names,
        failing_text,
    ) = (
        build_failing_test_text(
            runner.buggy_dir
        )
    )

    failure_traces = (
        load_failure_traces(
            runner.buggy_dir
        )
    )

    # Important:
    # pure Retriever v1 ranking.
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

    exact_stack_matches = (
        find_exact_stack_candidates(
            methods=(
                program_methods
            ),
            traces=(
                failure_traces
            ),
        )
    )

    gt_candidate_keys = (
        gt_keys
    )

    record = {
        "project": project,
        "bug_id": bug_id,
        "success": True,
        "production_method_count": (
            len(program_methods)
        ),
        "ground_truth_method_count": (
            len(gt_keys)
        ),
        "failure_trace_count": (
            len(failure_traces)
        ),
        "exact_stack_candidate_count": (
            len(
                exact_stack_matches
            )
        ),
        "budgets": {},
    }

    # Save exact stack evidence.
    stack_candidates_json = []

    for evidence in (
        exact_stack_matches.values()
    ):

        candidate = (
            evidence[
                "candidate"
            ]
        )

        key = candidate_key(
            candidate,
            source_root,
        )

        stack_candidates_json.append(
            {
                "class_name": (
                    candidate.class_name
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
                "match_type": (
                    evidence[
                        "match_type"
                    ]
                ),
                "stack_depth": (
                    evidence[
                        "stack_depth"
                    ]
                ),
                "frame_class": (
                    evidence[
                        "frame_class"
                    ]
                ),
                "frame_method": (
                    evidence[
                        "frame_method"
                    ]
                ),
                "frame_file": (
                    evidence[
                        "frame_file"
                    ]
                ),
                "frame_line": (
                    evidence[
                        "frame_line"
                    ]
                ),
                "is_ground_truth": (
                    key
                    in gt_candidate_keys
                ),
            }
        )

    stack_candidates_json.sort(
        key=lambda item: (
            item[
                "stack_depth"
            ],
            item[
                "class_name"
            ],
            item[
                "start_line"
            ],
        )
    )

    record[
        "exact_stack_candidates"
    ] = stack_candidates_json

    for n in TOP_N_VALUES:

        base_items = (
            ranked[:n]
        )

        base_keys = {
            candidate_key(
                item.candidate,
                source_root,
            )
            for item
            in base_items
        }

        exact_stack_keys = {
            candidate_key(
                evidence[
                    "candidate"
                ],
                source_root,
            )
            for evidence
            in exact_stack_matches.values()
        }

        augmented_keys = (
            base_keys
            | exact_stack_keys
        )

        base_hit = bool(
            gt_candidate_keys
            & base_keys
        )

        augmented_hit = bool(
            gt_candidate_keys
            & augmented_keys
        )

        extra_stack_keys = (
            exact_stack_keys
            - base_keys
        )

        record[
            "budgets"
        ][
            str(n)
        ] = {
            "base_pool_size": (
                len(
                    base_keys
                )
            ),
            "augmented_pool_size": (
                len(
                    augmented_keys
                )
            ),
            "extra_stack_candidates": (
                len(
                    extra_stack_keys
                )
            ),
            "base_hit": (
                base_hit
            ),
            "augmented_hit": (
                augmented_hit
            ),
            "improved": (
                not base_hit
                and augmented_hit
            ),
            "regressed": (
                base_hit
                and not augmented_hit
            ),
        }

    return (
        record,
        runner,
    )


def summarize(
    records: list[dict],
) -> dict:

    valid = [
        record
        for record in records
        if record.get(
            "success"
        )
    ]

    summary = {
        "evaluated_bugs": (
            len(valid)
        ),
        "budgets": {},
    }

    if not valid:

        return summary

    for n in TOP_N_VALUES:

        key = str(
            n
        )

        base_hits = sum(
            record[
                "budgets"
            ][
                key
            ][
                "base_hit"
            ]
            for record
            in valid
        )

        augmented_hits = sum(
            record[
                "budgets"
            ][
                key
            ][
                "augmented_hit"
            ]
            for record
            in valid
        )

        improved = sum(
            record[
                "budgets"
            ][
                key
            ][
                "improved"
            ]
            for record
            in valid
        )

        regressed = sum(
            record[
                "budgets"
            ][
                key
            ][
                "regressed"
            ]
            for record
            in valid
        )

        extras = [
            record[
                "budgets"
            ][
                key
            ][
                "extra_stack_candidates"
            ]
            for record
            in valid
        ]

        augmented_sizes = [
            record[
                "budgets"
            ][
                key
            ][
                "augmented_pool_size"
            ]
            for record
            in valid
        ]

        summary[
            "budgets"
        ][
            key
        ] = {
            "base_hits": (
                base_hits
            ),
            "augmented_hits": (
                augmented_hits
            ),
            "base_recall": (
                base_hits
                / len(valid)
            ),
            "augmented_recall": (
                augmented_hits
                / len(valid)
            ),
            "delta_recall": (
                (
                    augmented_hits
                    - base_hits
                )
                / len(valid)
            ),
            "improved_bugs": (
                improved
            ),
            "regressed_bugs": (
                regressed
            ),
            "mean_extra_stack_candidates": (
                sum(extras)
                / len(extras)
            ),
            "median_extra_stack_candidates": (
                statistics.median(
                    extras
                )
            ),
            "max_extra_stack_candidates": (
                max(
                    extras
                )
            ),
            "mean_augmented_pool_size": (
                sum(
                    augmented_sizes
                )
                / len(
                    augmented_sizes
                )
            ),
        }

    return summary


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    benchmark = load_json(
        args.benchmark
    )

    entries = benchmark[
        "entries"
    ]

    if (
        args.max_bugs
        is not None
    ):

        entries = (
            entries[
                :args.max_bugs
            ]
        )

    records = []

    print()
    print("=" * 100)
    print(
        "CAMD Exact Stack "
        "Candidate Augmentation"
    )
    print("=" * 100)

    print(
        f"Benchmark: "
        f"{args.benchmark}"
    )

    print(
        f"Selected bugs: "
        f"{len(entries)}"
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
            entry[
                "benchmark_id"
            ]
        )

        print()
        print("=" * 100)

        print(
            f"[{index}/"
            f"{len(entries)}] "
            f"{benchmark_id}"
        )

        print("=" * 100)

        runner = None

        try:

            (
                record,
                runner,
            ) = (
                evaluate_one_bug(
                    project=project,
                    bug_id=bug_id,
                )
            )

            record[
                "benchmark_id"
            ] = benchmark_id

            print(
                "Production methods: "
                f"{record['production_method_count']}"
            )

            print(
                "GT methods: "
                f"{record['ground_truth_method_count']}"
            )

            print(
                "Failure traces: "
                f"{record['failure_trace_count']}"
            )

            print(
                "Exact stack candidates: "
                f"{record['exact_stack_candidate_count']}"
            )

            for n in TOP_N_VALUES:

                budget = (
                    record[
                        "budgets"
                    ][
                        str(n)
                    ]
                )

                print(
                    f"@{n:<3} "
                    f"base={budget['base_hit']} "
                    f"aug={budget['augmented_hit']} "
                    f"extra={budget['extra_stack_candidates']}"
                )

        except Exception as exc:

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
                "success": False,
                "error": str(
                    exc
                ),
            }

            print(
                f"FAILED: {exc}"
            )

        records.append(
            record
        )

        if (
            runner is not None
            and not args.keep_checkouts
        ):

            if runner.buggy_dir.exists():

                shutil.rmtree(
                    runner.buggy_dir
                )

            if runner.fixed_dir.exists():

                shutil.rmtree(
                    runner.fixed_dir
                )

        payload = {
            "benchmark": str(
                args.benchmark
            ),
            "method": (
                "retriever_v1_plus_"
                "exact_stack_augmentation"
            ),
            "budgets": (
                TOP_N_VALUES
            ),
            "records": (
                records
            ),
            "summary": (
                summarize(
                    records
                )
            ),
        }

        save_json(
            args.output,
            payload,
        )

    summary = summarize(
        records
    )

    save_json(
        args.output,
        {
            "benchmark": str(
                args.benchmark
            ),
            "method": (
                "retriever_v1_plus_"
                "exact_stack_augmentation"
            ),
            "budgets": (
                TOP_N_VALUES
            ),
            "records": (
                records
            ),
            "summary": (
                summary
            ),
        },
    )

    print()
    print("=" * 100)
    print(
        "Stack Augmentation Summary"
    )
    print("=" * 100)

    print(
        f"Evaluated bugs: "
        f"{summary['evaluated_bugs']}"
    )

    for n in TOP_N_VALUES:

        info = (
            summary[
                "budgets"
            ][
                str(n)
            ]
        )

        print()
        print(
            f"Top-{n}"
        )

        print(
            "  Base Recall:      "
            f"{info['base_recall']:.4f}"
        )

        print(
            "  Augmented Recall: "
            f"{info['augmented_recall']:.4f}"
        )

        print(
            "  Delta:            "
            f"{info['delta_recall']:+.4f}"
        )

        print(
            "  Improved bugs:    "
            f"{info['improved_bugs']}"
        )

        print(
            "  Regressed bugs:   "
            f"{info['regressed_bugs']}"
        )

        print(
            "  Mean extra:       "
            f"{info['mean_extra_stack_candidates']:.2f}"
        )

        print(
            "  Median extra:     "
            f"{info['median_extra_stack_candidates']:.2f}"
        )

        print(
            "  Max extra:        "
            f"{info['max_extra_stack_candidates']}"
        )

        print(
            "  Mean pool size:   "
            f"{info['mean_augmented_pool_size']:.2f}"
        )

    print()
    print(
        f"Saved:\n{args.output}"
    )


if __name__ == "__main__":
    main()