from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import defaultdict
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
    / "fse_ase_benchmark_v1.json"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "defects4j"
    / "fse_ase_final_retrieval_results.json"
)


TOP_N_VALUES = [
    10,
    20,
    50,
    100,
]


# These two frozen final-set bugs require adding a method
# that does not exist in the buggy revision.
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


def build_ground_truth_keys(
    runner: Defects4JExperimentRunner,
):

    source_root = (
        runner.get_source_root(
            runner.buggy_dir
        )
    )

    gt_keys = set()

    modified_classes = (
        runner.get_modified_classes()
    )

    for class_name in modified_classes:

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

                previous = (
                    matches.get(
                        key
                    )
                )

                evidence = {
                    "candidate": candidate,
                    "depth": (
                        production_depth
                    ),
                    "evidence_type": (
                        evidence_type
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

    project = entry[
        "project"
    ]

    bug_id = int(
        entry[
            "bug_id"
        ]
    )

    return (
        f"{project}-{bug_id}"
    )


def summarize_subset(
    records,
):

    valid = [
        r
        for r in records
        if (
            r.get("success")
            and r.get("method_applicable")
        )
    ]

    result = {
        "evaluated_bugs": (
            len(valid)
        ),
        "budgets": {},
    }

    if not valid:
        return result

    for n in TOP_N_VALUES:

        key = str(n)

        base_hits = sum(
            r["budgets"][key][
                "base_hit"
            ]
            for r in valid
        )

        stack_hits = sum(
            r["budgets"][key][
                "stack_hit"
            ]
            for r in valid
        )

        final_hits = sum(
            r["budgets"][key][
                "final_hit"
            ]
            for r in valid
        )

        extra_stack = [
            r["budgets"][key][
                "extra_stack_candidates"
            ]
            for r in valid
        ]

        extra_call = [
            r["budgets"][key][
                "extra_call_candidates"
            ]
            for r in valid
        ]

        final_sizes = [
            r["budgets"][key][
                "final_pool_size"
            ]
            for r in valid
        ]

        result[
            "budgets"
        ][key] = {
            "base_hits": (
                base_hits
            ),
            "stack_hits": (
                stack_hits
            ),
            "final_hits": (
                final_hits
            ),
            "base_recall": (
                base_hits
                / len(valid)
            ),
            "stack_recall": (
                stack_hits
                / len(valid)
            ),
            "final_recall": (
                final_hits
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
                    final_hits
                    - stack_hits
                )
                / len(valid)
            ),
            "delta_final_vs_base": (
                (
                    final_hits
                    - base_hits
                )
                / len(valid)
            ),
            "improved_by_stack": sum(
                (
                    not r[
                        "budgets"
                    ][key]["base_hit"]
                    and r[
                        "budgets"
                    ][key]["stack_hit"]
                )
                for r in valid
            ),
            "improved_by_call": sum(
                (
                    not r[
                        "budgets"
                    ][key]["stack_hit"]
                    and r[
                        "budgets"
                    ][key]["final_hit"]
                )
                for r in valid
            ),
            "regressed_by_stack": sum(
                (
                    r[
                        "budgets"
                    ][key]["base_hit"]
                    and not r[
                        "budgets"
                    ][key]["stack_hit"]
                )
                for r in valid
            ),
            "regressed_by_call": sum(
                (
                    r[
                        "budgets"
                    ][key]["stack_hit"]
                    and not r[
                        "budgets"
                    ][key]["final_hit"]
                )
                for r in valid
            ),
            "mean_extra_stack": (
                sum(extra_stack)
                / len(extra_stack)
            ),
            "median_extra_stack": (
                statistics.median(
                    extra_stack
                )
            ),
            "max_extra_stack": (
                max(extra_stack)
            ),
            "mean_extra_call": (
                sum(extra_call)
                / len(extra_call)
            ),
            "median_extra_call": (
                statistics.median(
                    extra_call
                )
            ),
            "max_extra_call": (
                max(extra_call)
            ),
            "mean_final_pool_size": (
                sum(final_sizes)
                / len(final_sizes)
            ),
            "median_final_pool_size": (
                statistics.median(
                    final_sizes
                )
            ),
            "max_final_pool_size": (
                max(final_sizes)
            ),
        }

    return result


def summarize(
    records,
):

    successful = [
        r
        for r in records
        if r.get(
            "success"
        )
    ]

    applicable = [
        r
        for r in successful
        if r.get(
            "method_applicable"
        )
    ]

    method_addition = [
        r
        for r in successful
        if (
            r.get(
                "applicability"
            )
            == "method_addition"
        )
    ]

    failed = [
        r
        for r in records
        if not r.get(
            "success"
        )
    ]

    project_groups = (
        defaultdict(list)
    )

    for record in records:

        project_groups[
            record["project"]
        ].append(
            record
        )

    per_project = {}

    for project in sorted(
        project_groups
    ):

        per_project[
            project
        ] = summarize_subset(
            project_groups[
                project
            ]
        )

    return {
        "selected_bugs": (
            len(records)
        ),
        "processable_bugs": (
            len(successful)
        ),
        "method_applicable_bugs": (
            len(applicable)
        ),
        "method_addition_bugs": (
            len(method_addition)
        ),
        "failed_bugs": (
            len(failed)
        ),
        "overall": (
            summarize_subset(
                records
            )
        ),
        "per_project": (
            per_project
        ),
    }


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

    args = parse_args()

    benchmark = load_json(
        args.benchmark
    )

    entries = get_entries(
        benchmark
    )

    records = []

    print()
    print("=" * 100)
    print(
        "CAMD-R Frozen Final Retrieval Evaluation"
    )
    print("=" * 100)

    print(
        f"Benchmark: {args.benchmark}"
    )

    print(
        f"Selected bugs: {len(entries)}"
    )

    print(
        "Frozen settings:"
    )

    print(
        "  Base retriever: v1"
    )

    print(
        "  Exact-stack augmentation: enabled"
    )

    print(
        "  Call-chain augmentation: enabled"
    )

    print(
        "  Test helper depth: "
        f"{args.test_helper_depth}"
    )

    print(
        "  Production depth: "
        f"{args.production_depth}"
    )

    print()

    for index, entry in enumerate(
        entries,
        start=1,
    ):

        project = (
            entry["project"]
        )

        bug_id = int(
            entry["bug_id"]
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

            method_addition = (
                (
                    project,
                    bug_id,
                )
                in
                KNOWN_METHOD_ADDITION_CASES
            )

            gt_keys = (
                build_ground_truth_keys(
                    runner
                )
            )

            if (
                not gt_keys
                and not method_addition
            ):

                raise RuntimeError(
                    "No ground-truth methods "
                    "found for an expected "
                    "method-applicable bug."
                )

            failing_extractor = (
                FailingTestExtractor(
                    checkout_dir=(
                        runner.buggy_dir
                    )
                )
            )

            failing_tests = (
                failing_extractor.extract()
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
                    traces=traces,
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

            call_keys = {
                candidate_key(
                    item.candidate,
                    source_root,
                )
                for item
                in call_candidates
            }

            call_gt_depth = None

            if gt_keys:

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

                    call_gt_depth = min(
                        gt_depths
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
                "success": True,
                "applicability": (
                    "method_addition"
                    if method_addition
                    else (
                        "existing_method"
                    )
                ),
                "method_applicable": (
                    not method_addition
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
                "failing_test_count": (
                    len(
                        failing_tests
                    )
                ),
                "failure_trace_count": (
                    len(
                        traces
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
                "stack_contains_gt": (
                    bool(
                        gt_keys
                        & stack_keys
                    )
                    if gt_keys
                    else None
                ),
                "call_contains_gt": (
                    bool(
                        gt_keys
                        & call_keys
                    )
                    if gt_keys
                    else None
                ),
                "call_gt_depth": (
                    call_gt_depth
                ),
                "budgets": {},
            }

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

                if gt_keys:

                    base_hit = bool(
                        gt_keys
                        & base_keys
                    )

                    stack_hit = bool(
                        gt_keys
                        & stack_pool
                    )

                    final_hit = bool(
                        gt_keys
                        & final_pool
                    )

                else:

                    base_hit = None
                    stack_hit = None
                    final_hit = None

                record[
                    "budgets"
                ][str(n)] = {
                    "base_hit": (
                        base_hit
                    ),
                    "stack_hit": (
                        stack_hit
                    ),
                    "final_hit": (
                        final_hit
                    ),
                    "base_pool_size": (
                        len(
                            base_keys
                        )
                    ),
                    "stack_pool_size": (
                        len(
                            stack_pool
                        )
                    ),
                    "final_pool_size": (
                        len(
                            final_pool
                        )
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
                "Applicability: "
                f"{record['applicability']}"
            )

            print(
                "GT methods: "
                f"{len(gt_keys)}"
            )

            print(
                "Failing tests: "
                f"{len(failing_tests)}"
            )

            print(
                "Failure traces: "
                f"{len(traces)}"
            )

            print(
                "Exact stack candidates: "
                f"{len(stack_keys)}"
            )

            print(
                "Call candidates: "
                f"{len(call_keys)}"
            )

            if not method_addition:

                print(
                    "Stack contains GT: "
                    f"{record['stack_contains_gt']}"
                )

                print(
                    "Call contains GT: "
                    f"{record['call_contains_gt']}"
                )

                print(
                    "Call GT depth: "
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
                        f"base="
                        f"{info['base_hit']} "
                        f"stack="
                        f"{info['stack_hit']} "
                        f"final="
                        f"{info['final_hit']} "
                        f"+stack="
                        f"{info['extra_stack_candidates']} "
                        f"+call="
                        f"{info['extra_call_candidates']} "
                        f"pool="
                        f"{info['final_pool_size']}"
                    )

            else:

                print(
                    "Localization Recall: N/A "
                    "(method addition)"
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
                "method_applicable": False,
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

        current_payload = {
            "benchmark": str(
                args.benchmark
            ),
            "evaluation": (
                "frozen_final_retrieval"
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
            },
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
            current_payload,
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

    final_summary = (
        summarize(
            records
        )
    )

    payload = {
        "benchmark": str(
            args.benchmark
        ),
        "evaluation": (
            "frozen_final_retrieval"
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
                args.test_helper_depth
            ),
            "production_depth": (
                args.production_depth
            ),
            "top_n_values": (
                TOP_N_VALUES
            ),
        },
        "records": records,
        "summary": final_summary,
    }

    save_json(
        args.output,
        payload,
    )

    print()
    print("=" * 100)
    print(
        "Frozen Final Retrieval Summary"
    )
    print("=" * 100)

    print(
        "Selected bugs: "
        f"{final_summary['selected_bugs']}"
    )

    print(
        "Processable bugs: "
        f"{final_summary['processable_bugs']}"
    )

    print(
        "Existing-method applicable: "
        f"{final_summary['method_applicable_bugs']}"
    )

    print(
        "Method-addition N/A: "
        f"{final_summary['method_addition_bugs']}"
    )

    print(
        "Failed: "
        f"{final_summary['failed_bugs']}"
    )

    overall = (
        final_summary[
            "overall"
        ]
    )

    print()
    print(
        "Overall existing-method "
        "localization:"
    )

    for n in TOP_N_VALUES:

        info = (
            overall[
                "budgets"
            ][str(n)]
        )

        print()
        print(
            f"Top-{n}"
        )

        print(
            "  Base Recall:   "
            f"{info['base_recall']:.4f} "
            f"({info['base_hits']}/"
            f"{overall['evaluated_bugs']})"
        )

        print(
            "  +Stack Recall: "
            f"{info['stack_recall']:.4f} "
            f"({info['stack_hits']}/"
            f"{overall['evaluated_bugs']})"
        )

        print(
            "  Final Recall:  "
            f"{info['final_recall']:.4f} "
            f"({info['final_hits']}/"
            f"{overall['evaluated_bugs']})"
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
            "  Final Δ Base:  "
            f"{info['delta_final_vs_base']:+.4f}"
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
            f"{info['regressed_by_stack']}"
        )

        print(
            "  Regress call:  "
            f"{info['regressed_by_call']}"
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
            "  Mean pool:     "
            f"{info['mean_final_pool_size']:.2f}"
        )

    print()
    print("=" * 100)
    print(
        "Per-project Final Recall"
    )
    print("=" * 100)

    for (
        project,
        project_summary,
    ) in final_summary[
        "per_project"
    ].items():

        print()
        print(
            f"{project} "
            f"(n="
            f"{project_summary['evaluated_bugs']})"
        )

        if (
            not project_summary[
                "budgets"
            ]
        ):
            continue

        for n in TOP_N_VALUES:

            info = (
                project_summary[
                    "budgets"
                ][str(n)]
            )

            print(
                f"  R@{n:<3} "
                f"{info['final_recall']:.4f}"
            )

    print()
    print("=" * 100)
    print(
        "Remaining Final Misses @10"
    )
    print("=" * 100)

    misses = []

    for record in records:

        if (
            not record.get(
                "success"
            )
            or not record.get(
                "method_applicable"
            )
        ):
            continue

        if not record[
            "budgets"
        ]["10"]["final_hit"]:

            misses.append(
                record
            )

    if not misses:

        print(
            "None"
        )

    else:

        for record in misses:

            print(
                f"{record['benchmark_id']} "
                f"| methods="
                f"{record['production_method_count']} "
                f"| stack="
                f"{record['stack_candidate_count']} "
                f"| call="
                f"{record['call_candidate_count']} "
                f"| stack_gt="
                f"{record['stack_contains_gt']} "
                f"| call_gt="
                f"{record['call_contains_gt']} "
                f"| call_depth="
                f"{record['call_gt_depth']}"
            )

    print()
    print(
        f"Saved:\n{args.output}"
    )


if __name__ == "__main__":
    main()  