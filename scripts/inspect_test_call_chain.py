from __future__ import annotations

import argparse
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from camd.context.method_extractor import extract_java_methods
from camd.evaluation.experiment_runner import Defects4JExperimentRunner
from camd.evaluation.failing_test_extractor import FailingTestExtractor
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


CALL_PATTERN = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
)


IGNORED_CALL_NAMES = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "throw",
    "new",
    "super",
    "this",
    "assert",
    "synchronized",
}


@dataclass(frozen=True)
class MethodKey:
    class_name: str
    method_name: str
    start_line: int
    end_line: int


def extract_called_names(
    code: str | None,
) -> set[str]:

    if not code:
        return set()

    names = set()

    for match in CALL_PATTERN.finditer(
        code
    ):

        name = match.group(1)

        if (
            name
            and name
            not in IGNORED_CALL_NAMES
        ):
            names.add(name)

    return names


def method_key(
    candidate: ProgramMethod,
) -> MethodKey:

    return MethodKey(
        class_name=(
            candidate.class_name
        ),
        method_name=(
            candidate.method.name
        ),
        start_line=(
            candidate.method.start_line
        ),
        end_line=(
            candidate.method.end_line
        ),
    )


def build_ground_truth_keys(
    runner: Defects4JExperimentRunner,
) -> set[MethodKey]:

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

        for method in gt_methods:

            gt_keys.add(
                MethodKey(
                    class_name=(
                        class_name
                    ),
                    method_name=(
                        method.name
                    ),
                    start_line=(
                        method.start_line
                    ),
                    end_line=(
                        method.end_line
                    ),
                )
            )

    return gt_keys


def build_name_index(
    methods: list[ProgramMethod],
) -> dict[
    str,
    list[ProgramMethod],
]:

    index = defaultdict(list)

    for candidate in methods:

        index[
            candidate.method.name
        ].append(
            candidate
        )

    return index


def resolve_calls(
    caller: ProgramMethod | None,
    called_names: set[str],
    name_index: dict[
        str,
        list[ProgramMethod],
    ],
    max_targets_per_name: int,
) -> list[ProgramMethod]:

    resolved = []
    seen = set()

    for called_name in sorted(
        called_names
    ):

        targets = list(
            name_index.get(
                called_name,
                [],
            )
        )

        if not targets:
            continue

        # Prefer same-class methods where possible.
        if caller is not None:

            same_class = [
                target
                for target in targets
                if (
                    target.class_name
                    == caller.class_name
                )
            ]

            other_classes = [
                target
                for target in targets
                if (
                    target.class_name
                    != caller.class_name
                )
            ]

            targets = (
                same_class
                + other_classes
            )

        targets = (
            targets[
                :max_targets_per_name
            ]
        )

        for target in targets:

            key = method_key(
                target
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            resolved.append(
                target
            )

    return resolved


def get_failing_tests(
    checkout_dir: Path,
):

    extractor = (
        FailingTestExtractor(
            checkout_dir=checkout_dir
        )
    )

    return extractor.extract()


def print_method(
    prefix: str,
    candidate: ProgramMethod,
    is_gt: bool,
) -> None:

    marker = (
        "  <GT>"
        if is_gt
        else ""
    )

    print(
        f"{prefix}"
        f"{candidate.class_name}."
        f"{candidate.method.name} "
        f"[{candidate.method.start_line}-"
        f"{candidate.method.end_line}]"
        f"{marker}"
    )


def inspect_bug(
    project: str,
    bug_id: int,
    max_depth: int,
    max_targets_per_name: int,
    max_print_per_depth: int,
) -> None:

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

    gt_keys = (
        build_ground_truth_keys(
            runner
        )
    )

    if not gt_keys:

        raise RuntimeError(
            "No ground-truth methods found."
        )

    retriever = (
        ProgramMethodRetriever(
            project=project,
            bug_id=bug_id,
            use_stack_trace=False,
        )
    )

    program_methods = (
        retriever.extract_program_methods(
            source_root
        )
    )

    name_index = (
        build_name_index(
            program_methods
        )
    )

    failing_tests = (
        get_failing_tests(
            runner.buggy_dir
        )
    )

    print()
    print("=" * 100)
    print(
        f"Call-Chain Feasibility: "
        f"{project}-{bug_id}"
    )
    print("=" * 100)

    print(
        f"Production methods: "
        f"{len(program_methods)}"
    )

    print(
        f"Ground-truth methods: "
        f"{len(gt_keys)}"
    )

    print(
        f"Failing tests: "
        f"{len(failing_tests)}"
    )

    print()

    print("Ground truth:")

    for gt in sorted(
        gt_keys,
        key=lambda x: (
            x.class_name,
            x.start_line,
        ),
    ):

        print(
            f"  {gt.class_name}."
            f"{gt.method_name} "
            f"[{gt.start_line}-{gt.end_line}]"
        )

    # -----------------------------------------------------
    # Depth 1:
    # calls appearing directly in failing-test source.
    # -----------------------------------------------------

    initial_called_names = set()

    for test in failing_tests:

        called = extract_called_names(
            test.code
        )

        initial_called_names.update(
            called
        )

        print()
        print(
            f"Failing test: "
            f"{test.full_name}"
        )

        print(
            "  syntactic calls: "
            + (
                ", ".join(
                    sorted(called)
                )
                if called
                else "(none)"
            )
        )

    print()
    print(
        f"Unique test call names: "
        f"{len(initial_called_names)}"
    )

    depth_one = (
        resolve_calls(
            caller=None,
            called_names=(
                initial_called_names
            ),
            name_index=name_index,
            max_targets_per_name=(
                max_targets_per_name
            ),
        )
    )

    queue = deque()

    visited = set()

    reachable_by_depth = (
        defaultdict(list)
    )

    for candidate in depth_one:

        key = method_key(
            candidate
        )

        if key in visited:
            continue

        visited.add(
            key
        )

        reachable_by_depth[
            1
        ].append(
            candidate
        )

        queue.append(
            (
                candidate,
                1,
            )
        )

    # -----------------------------------------------------
    # BFS expansion
    # -----------------------------------------------------

    while queue:

        caller, depth = (
            queue.popleft()
        )

        if depth >= max_depth:
            continue

        called_names = (
            extract_called_names(
                caller.method.code
            )
        )

        targets = (
            resolve_calls(
                caller=caller,
                called_names=(
                    called_names
                ),
                name_index=(
                    name_index
                ),
                max_targets_per_name=(
                    max_targets_per_name
                ),
            )
        )

        next_depth = (
            depth + 1
        )

        for target in targets:

            key = method_key(
                target
            )

            if key in visited:
                continue

            visited.add(
                key
            )

            reachable_by_depth[
                next_depth
            ].append(
                target
            )

            queue.append(
                (
                    target,
                    next_depth,
                )
            )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    first_gt_depth = None
    cumulative = set()

    for depth in range(
        1,
        max_depth + 1,
    ):

        methods = (
            reachable_by_depth[
                depth
            ]
        )

        depth_keys = {
            method_key(
                candidate
            )
            for candidate in methods
        }

        cumulative.update(
            depth_keys
        )

        gt_at_depth = (
            gt_keys
            & depth_keys
        )

        gt_cumulative = (
            gt_keys
            & cumulative
        )

        if (
            first_gt_depth is None
            and gt_at_depth
        ):
            first_gt_depth = (
                depth
            )

        print()
        print("-" * 100)

        print(
            f"Depth {depth}"
        )

        print(
            f"  New reachable methods: "
            f"{len(methods)}"
        )

        print(
            f"  Cumulative reachable: "
            f"{len(cumulative)}"
        )

        print(
            f"  GT at this depth: "
            f"{len(gt_at_depth)}"
        )

        print(
            f"  GT cumulative: "
            f"{len(gt_cumulative)}"
            f"/{len(gt_keys)}"
        )

        # Print GT first if present.
        for candidate in methods:

            key = method_key(
                candidate
            )

            if key in gt_keys:

                print_method(
                    prefix="  ",
                    candidate=candidate,
                    is_gt=True,
                )

        print_count = 0

        for candidate in methods:

            key = method_key(
                candidate
            )

            if key in gt_keys:
                continue

            if (
                print_count
                >= max_print_per_depth
            ):
                break

            print_method(
                prefix="  ",
                candidate=candidate,
                is_gt=False,
            )

            print_count += 1

        if (
            len(methods)
            > max_print_per_depth
        ):

            print(
                "  ... "
                f"{len(methods) - max_print_per_depth} "
                "more"
            )

    print()
    print("=" * 100)
    print("Feasibility Result")
    print("=" * 100)

    if first_gt_depth is None:

        print(
            "GT reachable within depth "
            f"{max_depth}: NO"
        )

    else:

        print(
            "GT reachable: YES"
        )

        print(
            f"First GT depth: "
            f"{first_gt_depth}"
        )

    print(
        f"Total reachable methods: "
        f"{len(visited)}"
    )

    print(
        "Reachable fraction: "
        f"{len(visited) / len(program_methods):.4f}"
    )


def parse_args():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--project",
        required=True,
    )

    parser.add_argument(
        "--bug-id",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-targets-per-name",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-print-per-depth",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    inspect_bug(
        project=args.project,
        bug_id=args.bug_id,
        max_depth=args.max_depth,
        max_targets_per_name=(
            args.max_targets_per_name
        ),
        max_print_per_depth=(
            args.max_print_per_depth
        ),
    )


if __name__ == "__main__":
    main()