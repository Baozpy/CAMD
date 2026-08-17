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
    / "fse_ase_retrieval_dev_v1.json"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "defects4j"
    / "fse_ase_retrieval_dev_call_augment_results.json"
)


TOP_N_VALUES = [
    10,
    20,
    50,
    100,
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


def candidate_key(
    candidate: ProgramMethod,
    source_root: Path,
):

    source_file = Path(
        candidate.source_file
    )

    try:

        source_key = str(
            source_file.relative_to(
                source_root
            )
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


def build_ground_truth_keys(
    runner: Defects4JExperimentRunner,
):

    source_root = (
        runner.get_source_root(
            runner.buggy_dir
        )
    )

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

        try:

            source_key = str(
                buggy_file.relative_to(
                    source_root
                )
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

            for candidate in candidates:

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
                    "depth": (
                        production_depth
                    ),
                }

                if (
                    previous is None
                    or production_depth
                    < previous["depth"]
                ):

                    matches[
                        key
                    ] = evidence

            production_depth += 1

    return matches


def summarize(
    records,
):

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
            record["budgets"][key][
                "base_hit"
            ]
            for record in valid
        )

        stack_hits = sum(
            record["budgets"][key][
                "stack_hit"
            ]
            for record in valid
        )

        call_hits = sum(
            record["budgets"][key][
                "call_hit"
            ]
            for record in valid
        )

        stack_extra = [
            record["budgets"][key][
                "extra_stack_candidates"
            ]
            for record in valid
        ]

        call_extra = [
            record["budgets"][key][
                "extra_call_candidates"
            ]
            for record in valid
        ]

        final_sizes = [
            record["budgets"][key][
                "final_pool_size"
            ]
            for record in valid
        ]

        summary[
            "budgets"
        ][key] = {
            "base_recall": (
                base_hits
                / len(valid)
            ),
            "stack_recall": (
                stack_hits
                / len(valid)
            ),
            "call_recall": (
                call_hits
                / len(valid)
            ),
            "delta_stack_vs_base": (
                (
                    stack_hits
                    - base_hits
                )
                / len(valid)
            ),
            "delta_call_vs_stack": (
                (
                    call_hits
                    - stack_hits
                )
                / len(valid)
            ),
            "delta_call_vs_base": (
                (
                    call_hits
                    - base_hits
                )
                / len(valid)
            ),
            "improved_by_stack": sum(
                (
                    not record[
                        "budgets"
                    ][key]["base_hit"]
                    and record[
                        "budgets"
                    ][key]["stack_hit"]
                )
                for record in valid
            ),
            "improved_by_call": sum(
                (
                    not record[
                        "budgets"
                    ][key]["stack_hit"]
                    and record[
                        "budgets"
                    ][key]["call_hit"]
                )
                for record in valid
            ),
            "regressed_stack": sum(
                (
                    record[
                        "budgets"
                    ][key]["base_hit"]
                    and not record[
                        "budgets"
                    ][key]["stack_hit"]
                )
                for record in valid
            ),
            "regressed_call": sum(
                (
                    record[
                        "budgets"
                    ][key]["stack_hit"]
                    and not record[
                        "budgets"
                    ][key]["call_hit"]
                )
                for record in valid
            ),
            "mean_extra_stack": (
                sum(stack_extra)
                / len(stack_extra)
            ),
            "mean_extra_call": (
                sum(call_extra)
                / len(call_extra)
            ),
            "median_extra_call": (
                statistics.median(
                    call_extra
                )
            ),
            "max_extra_call": (
                max(
                    call_extra
                )
            ),
            "mean_final_pool_size": (
                sum(final_sizes)
                / len(final_sizes)
            ),
        }

    return summary


def parse_args():

    parser = argparse.ArgumentParser()

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

    args = parse_args()

    benchmark = load_json(
        args.benchmark
    )

    records = []

    entries = benchmark[
        "entries"
    ]

    print()
    print("=" * 100)
    print(
        "CAMD Call-Chain "
        "Candidate Augmentation"
    )
    print("=" * 100)

    for index, entry in enumerate(
        entries,
        start=1,
    ):

        benchmark_id = (
            entry[
                "benchmark_id"
            ]
        )

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

        print()
        print("=" * 100)

        print(
            f"[{index}/"
            f"{len(entries)}] "
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

            gt_keys = (
                build_ground_truth_keys(
                    runner
                )
            )

            if not gt_keys:

                raise RuntimeError(
                    "No ground-truth "
                    "methods found."
                )

            failing_extractor = (
                FailingTestExtractor(
                    checkout_dir=(
                        runner.buggy_dir
                    )
                )
            )

            failing_tests = (
                failing_extractor
                .extract()
            )

            failing_names = [
                test.full_name
                for test in failing_tests
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

            stack_keys = {
                candidate_key(
                    evidence[
                        "candidate"
                    ],
                    source_root,
                )
                for evidence
                in stack_matches.values()
            }

            call_keys = {
                candidate_key(
                    item.candidate,
                    source_root,
                )
                for item
                in call_candidates
            }

            record = {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": project,
                "bug_id": bug_id,
                "success": True,
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
                "stack_candidate_count": (
                    len(
                        stack_keys
                    )
                ),
                "call_candidate_count": (
                    len(
                        call_keys
                    )
                ),
                "call_gt_hit": bool(
                    gt_keys
                    & call_keys
                ),
                "call_gt_depth": None,
                "budgets": {},
            }

            gt_depths = [
                item.depth
                for item
                in call_candidates
                if (
                    candidate_key(
                        item.candidate,
                        source_root,
                    )
                    in gt_keys
                )
            ]

            if gt_depths:

                record[
                    "call_gt_depth"
                ] = min(
                    gt_depths
                )

            for n in TOP_N_VALUES:

                base_keys = {
                    candidate_key(
                        item.candidate,
                        source_root,
                    )
                    for item
                    in ranked[:n]
                }

                stack_pool = (
                    base_keys
                    | stack_keys
                )

                final_pool = (
                    stack_pool
                    | call_keys
                )

                base_hit = bool(
                    gt_keys
                    & base_keys
                )

                stack_hit = bool(
                    gt_keys
                    & stack_pool
                )

                call_hit = bool(
                    gt_keys
                    & final_pool
                )

                record[
                    "budgets"
                ][str(n)] = {
                    "base_hit": (
                        base_hit
                    ),
                    "stack_hit": (
                        stack_hit
                    ),
                    "call_hit": (
                        call_hit
                    ),
                    "base_pool_size": (
                        len(base_keys)
                    ),
                    "stack_pool_size": (
                        len(stack_pool)
                    ),
                    "final_pool_size": (
                        len(final_pool)
                    ),
                    "extra_stack_candidates": (
                        len(
                            stack_keys
                            - base_keys
                        )
                    ),
                    "extra_call_candidates": (
                        len(
                            call_keys
                            - stack_pool
                        )
                    ),
                }

            print(
                "Production methods: "
                f"{len(program_methods)}"
            )

            print(
                "GT methods: "
                f"{len(gt_keys)}"
            )

            print(
                "Exact stack candidates: "
                f"{len(stack_keys)}"
            )

            print(
                "Call-chain candidates: "
                f"{len(call_keys)}"
            )

            print(
                "Call-chain contains GT: "
                f"{record['call_gt_hit']}"
            )

            print(
                "GT call depth: "
                f"{record['call_gt_depth']}"
            )

            for n in TOP_N_VALUES:

                info = (
                    record[
                        "budgets"
                    ][str(n)]
                )

                print(
                    f"@{n:<3} "
                    f"base={info['base_hit']} "
                    f"stack={info['stack_hit']} "
                    f"call={info['call_hit']} "
                    f"extra_stack="
                    f"{info['extra_stack_candidates']} "
                    f"extra_call="
                    f"{info['extra_call_candidates']} "
                    f"final="
                    f"{info['final_pool_size']}"
                )

        except Exception as exc:

            record = {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": project,
                "bug_id": bug_id,
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

        payload = {
            "benchmark": str(
                args.benchmark
            ),
            "method": (
                "retriever_v1_plus_"
                "stack_plus_call_chain"
            ),
            "test_helper_depth": (
                args.test_helper_depth
            ),
            "production_depth": (
                args.production_depth
            ),
            "records": records,
            "summary": summarize(
                records
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
                "stack_plus_call_chain"
            ),
            "test_helper_depth": (
                args.test_helper_depth
            ),
            "production_depth": (
                args.production_depth
            ),
            "records": records,
            "summary": summary,
        },
    )

    print()
    print("=" * 100)
    print(
        "Call Augmentation Summary"
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
            ][str(n)]
        )

        print()
        print(
            f"Top-{n}"
        )

        print(
            "  Base Recall:   "
            f"{info['base_recall']:.4f}"
        )

        print(
            "  Stack Recall:  "
            f"{info['stack_recall']:.4f}"
        )

        print(
            "  Call Recall:   "
            f"{info['call_recall']:.4f}"
        )

        print(
            "  Stack Δ Base:  "
            f"{info['delta_stack_vs_base']:+.4f}"
        )

        print(
            "  Call Δ Stack:  "
            f"{info['delta_call_vs_stack']:+.4f}"
        )

        print(
            "  Call Δ Base:   "
            f"{info['delta_call_vs_base']:+.4f}"
        )

        print(
            "  Improved stack:"
            f" {info['improved_by_stack']}"
        )

        print(
            "  Improved call: "
            f"{info['improved_by_call']}"
        )

        print(
            "  Regress stack: "
            f"{info['regressed_stack']}"
        )

        print(
            "  Regress call:  "
            f"{info['regressed_call']}"
        )

        print(
            "  Mean +stack:   "
            f"{info['mean_extra_stack']:.2f}"
        )

        print(
            "  Mean +call:    "
            f"{info['mean_extra_call']:.2f}"
        )

        print(
            "  Median +call:  "
            f"{info['median_extra_call']:.2f}"
        )

        print(
            "  Max +call:     "
            f"{info['max_extra_call']}"
        )

        print(
            "  Mean final N:  "
            f"{info['mean_final_pool_size']:.2f}"
        )

    print()
    print(
        f"Saved:\n{args.output}"
    )


if __name__ == "__main__":
    main()