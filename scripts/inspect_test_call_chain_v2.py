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

NEW_CLASS_PATTERN = re.compile(
    r"\bnew\s+([A-Z][A-Za-z0-9_$]*)\s*\("
)

DECLARATION_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9_$]*)\s+"
    r"([a-zA-Z_$][A-Za-z0-9_$]*)\s*="
)

RECEIVER_CALL_PATTERN = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_$]*)"
    r"\."
    r"([A-Za-z_$][A-Za-z0-9_$]*)"
    r"\s*\("
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


@dataclass
class TestMethod:
    class_name: str
    method_name: str
    code: str


def method_key(
    candidate: ProgramMethod,
) -> MethodKey:

    return MethodKey(
        class_name=candidate.class_name,
        method_name=candidate.method.name,
        start_line=candidate.method.start_line,
        end_line=candidate.method.end_line,
    )


def simple_class_name(
    class_name: str,
) -> str:

    return class_name.split(".")[-1]


def extract_called_names(
    code: str | None,
) -> set[str]:

    if not code:
        return set()

    result = set()

    for match in CALL_PATTERN.finditer(
        code
    ):

        name = match.group(1)

        if (
            name
            and name
            not in IGNORED_CALL_NAMES
        ):
            result.add(name)

    return result


def extract_new_classes(
    code: str | None,
) -> set[str]:

    if not code:
        return set()

    return {
        match.group(1)
        for match in NEW_CLASS_PATTERN.finditer(
            code
        )
    }


def extract_local_types(
    code: str | None,
) -> dict[str, str]:

    if not code:
        return {}

    mapping = {}

    for match in DECLARATION_PATTERN.finditer(
        code
    ):

        class_name = match.group(1)
        variable_name = match.group(2)

        mapping[
            variable_name
        ] = class_name

    return mapping


def extract_receiver_calls(
    code: str | None,
) -> list[
    tuple[
        str,
        str,
    ]
]:

    if not code:
        return []

    result = []

    for match in RECEIVER_CALL_PATTERN.finditer(
        code
    ):

        receiver = (
            match.group(1)
        )

        method = (
            match.group(2)
        )

        result.append(
            (
                receiver,
                method,
            )
        )

    return result


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
                    class_name=class_name,
                    method_name=method.name,
                    start_line=method.start_line,
                    end_line=method.end_line,
                )
            )

    return gt_keys


def build_class_index(
    methods: list[ProgramMethod],
) -> dict[
    str,
    list[ProgramMethod],
]:

    index = defaultdict(list)

    for candidate in methods:

        index[
            candidate.class_name
        ].append(
            candidate
        )

    return index


def build_simple_class_index(
    methods: list[ProgramMethod],
) -> dict[
    str,
    list[str],
]:

    index = defaultdict(list)

    for candidate in methods:

        simple = simple_class_name(
            candidate.class_name
        )

        if (
            candidate.class_name
            not in index[simple]
        ):
            index[simple].append(
                candidate.class_name
            )

    return index


def build_class_method_index(
    methods: list[ProgramMethod],
) -> dict[
    tuple[
        str,
        str,
    ],
    list[ProgramMethod],
]:

    index = defaultdict(list)

    for candidate in methods:

        index[
            (
                candidate.class_name,
                candidate.method.name,
            )
        ].append(
            candidate
        )

    return index


def find_test_source(
    checkout_dir: Path,
    class_name: str,
) -> Path | None:

    filename = (
        class_name.split(".")[-1]
        + ".java"
    )

    candidates = list(
        checkout_dir.rglob(
            filename
        )
    )

    for candidate in candidates:

        path_text = str(
            candidate
        ).replace(
            "\\",
            "/",
        )

        if (
            "/test/"
            in path_text
            or "/tests/"
            in path_text
        ):
            return candidate

    if candidates:
        return candidates[0]

    return None


def extract_test_methods(
    source_file: Path,
) -> dict[
    str,
    TestMethod,
]:

    methods = (
        extract_java_methods(
            source_file
        )
    )

    result = {}

    class_name = (
        source_file.stem
    )

    for method in methods:

        result[
            method.name
        ] = TestMethod(
            class_name=class_name,
            method_name=method.name,
            code=method.code,
        )

    return result


def build_test_helper_closure(
    failing_method_name: str,
    test_methods: dict[
        str,
        TestMethod,
    ],
    max_depth: int,
) -> tuple[
    list[TestMethod],
    set[str],
]:

    visited = set()

    queue = deque(
        [
            (
                failing_method_name,
                0,
            )
        ]
    )

    closure = []

    all_calls = set()

    while queue:

        method_name, depth = (
            queue.popleft()
        )

        if method_name in visited:
            continue

        visited.add(
            method_name
        )

        method = (
            test_methods.get(
                method_name
            )
        )

        if method is None:
            continue

        closure.append(
            method
        )

        called_names = (
            extract_called_names(
                method.code
            )
        )

        all_calls.update(
            called_names
        )

        if depth >= max_depth:
            continue

        for called_name in (
            called_names
        ):

            if (
                called_name
                in test_methods
                and called_name
                not in visited
            ):

                queue.append(
                    (
                        called_name,
                        depth + 1,
                    )
                )

    return (
        closure,
        all_calls,
    )


def infer_entry_classes(
    helper_methods: list[TestMethod],
    simple_class_index: dict[
        str,
        list[str],
    ],
) -> set[str]:

    inferred = set()

    for helper in helper_methods:

        code = helper.code

        new_classes = (
            extract_new_classes(
                code
            )
        )

        local_types = (
            extract_local_types(
                code
            )
        )

        candidate_simple_names = (
            set(new_classes)
            | set(
                local_types.values()
            )
        )

        for simple_name in (
            candidate_simple_names
        ):

            matches = (
                simple_class_index.get(
                    simple_name,
                    []
                )
            )

            for full_name in matches:

                inferred.add(
                    full_name
                )

    return inferred


def infer_entry_methods(
    helper_methods: list[TestMethod],
    entry_classes: set[str],
    simple_class_index: dict[
        str,
        list[str],
    ],
    class_method_index,
) -> list[ProgramMethod]:

    results = []
    seen = set()

    for helper in helper_methods:

        code = helper.code

        local_types = (
            extract_local_types(
                code
            )
        )

        receiver_calls = (
            extract_receiver_calls(
                code
            )
        )

        # Constructor entry points.
        for simple_name in (
            extract_new_classes(
                code
            )
        ):

            for full_class in (
                simple_class_index.get(
                    simple_name,
                    [],
                )
            ):

                constructor_targets = (
                    class_method_index.get(
                        (
                            full_class,
                            simple_name,
                        ),
                        [],
                    )
                )

                for candidate in (
                    constructor_targets
                ):

                    key = method_key(
                        candidate
                    )

                    if key not in seen:

                        seen.add(key)
                        results.append(
                            candidate
                        )

        # Receiver-resolved calls.
        for (
            receiver,
            called_method,
        ) in receiver_calls:

            receiver_type = (
                local_types.get(
                    receiver
                )
            )

            if receiver_type is None:
                continue

            for full_class in (
                simple_class_index.get(
                    receiver_type,
                    [],
                )
            ):

                targets = (
                    class_method_index.get(
                        (
                            full_class,
                            called_method,
                        ),
                        [],
                    )
                )

                for candidate in targets:

                    key = method_key(
                        candidate
                    )

                    if key not in seen:

                        seen.add(key)
                        results.append(
                            candidate
                        )

    # If we inferred a class but not a concrete
    # receiver call, constructors still provide
    # a conservative class entry point.
    for full_class in entry_classes:

        simple_name = (
            simple_class_name(
                full_class
            )
        )

        for candidate in (
            class_method_index.get(
                (
                    full_class,
                    simple_name,
                ),
                [],
            )
        ):

            key = method_key(
                candidate
            )

            if key not in seen:

                seen.add(key)
                results.append(
                    candidate
                )

    return results


def class_local_targets(
    caller: ProgramMethod,
    class_method_index,
) -> list[ProgramMethod]:

    called_names = (
        extract_called_names(
            caller.method.code
        )
    )

    results = []
    seen = set()

    for called_name in sorted(
        called_names
    ):

        targets = (
            class_method_index.get(
                (
                    caller.class_name,
                    called_name,
                ),
                [],
            )
        )

        for target in targets:

            key = method_key(
                target
            )

            if key in seen:
                continue

            seen.add(key)
            results.append(
                target
            )

    return results


def inspect_bug(
    project: str,
    bug_id: int,
    test_helper_depth: int,
    production_depth: int,
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

    gt_keys = (
        build_ground_truth_keys(
            runner
        )
    )

    if not gt_keys:

        raise RuntimeError(
            "No ground-truth methods found."
        )

    source_root = (
        runner.get_source_root(
            runner.buggy_dir
        )
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

    class_index = (
        build_class_index(
            program_methods
        )
    )

    simple_class_index = (
        build_simple_class_index(
            program_methods
        )
    )

    class_method_index = (
        build_class_method_index(
            program_methods
        )
    )

    failing_tests = (
        FailingTestExtractor(
            checkout_dir=(
                runner.buggy_dir
            )
        ).extract()
    )

    print()
    print("=" * 100)
    print(
        f"Call-Chain v2 Feasibility: "
        f"{project}-{bug_id}"
    )
    print("=" * 100)

    print(
        f"Production methods: "
        f"{len(program_methods)}"
    )

    print(
        f"GT methods: "
        f"{len(gt_keys)}"
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
            f"[{gt.start_line}-"
            f"{gt.end_line}]"
        )

    all_entries = []
    all_helper_names = set()
    all_entry_classes = set()

    for failing_test in failing_tests:

        source_file = (
            failing_test.source_file
        )

        if source_file is None:

            source_file = (
                find_test_source(
                    runner.buggy_dir,
                    failing_test.class_name,
                )
            )

        if source_file is None:

            print()
            print(
                "Could not locate test source:"
            )

            print(
                f"  {failing_test.full_name}"
            )

            continue

        test_methods = (
            extract_test_methods(
                source_file
            )
        )

        (
            helper_closure,
            _,
        ) = (
            build_test_helper_closure(
                failing_method_name=(
                    failing_test.method_name
                ),
                test_methods=(
                    test_methods
                ),
                max_depth=(
                    test_helper_depth
                ),
            )
        )

        helper_names = {
            method.method_name
            for method
            in helper_closure
        }

        all_helper_names.update(
            helper_names
        )

        entry_classes = (
            infer_entry_classes(
                helper_methods=(
                    helper_closure
                ),
                simple_class_index=(
                    simple_class_index
                ),
            )
        )

        all_entry_classes.update(
            entry_classes
        )

        entry_methods = (
            infer_entry_methods(
                helper_methods=(
                    helper_closure
                ),
                entry_classes=(
                    entry_classes
                ),
                simple_class_index=(
                    simple_class_index
                ),
                class_method_index=(
                    class_method_index
                ),
            )
        )

        all_entries.extend(
            entry_methods
        )

        print()
        print(
            f"Failing test: "
            f"{failing_test.full_name}"
        )

        print(
            "  Test helper closure:"
        )

        for helper in (
            helper_closure
        ):

            print(
                f"    {helper.method_name}"
            )

        print(
            "  Inferred production classes:"
        )

        if entry_classes:

            for class_name in sorted(
                entry_classes
            ):

                print(
                    f"    {class_name}"
                )

        else:

            print(
                "    (none)"
            )

        print(
            "  Production entry methods:"
        )

        if entry_methods:

            for candidate in (
                entry_methods
            ):

                print(
                    "    "
                    f"{candidate.class_name}."
                    f"{candidate.method.name} "
                    f"[{candidate.method.start_line}-"
                    f"{candidate.method.end_line}]"
                )

        else:

            print(
                "    (none)"
            )

    # Deduplicate entries.
    entry_map = {}

    for candidate in all_entries:

        entry_map[
            method_key(
                candidate
            )
        ] = candidate

    entries = list(
        entry_map.values()
    )

    print()
    print("=" * 100)
    print("Production Expansion")
    print("=" * 100)

    queue = deque()

    visited = set()

    reachable_by_depth = (
        defaultdict(list)
    )

    for candidate in entries:

        key = method_key(
            candidate
        )

        if key in visited:
            continue

        visited.add(
            key
        )

        reachable_by_depth[
            0
        ].append(
            candidate
        )

        queue.append(
            (
                candidate,
                0,
            )
        )

    first_gt_depth = None

    while queue:

        caller, depth = (
            queue.popleft()
        )

        if depth >= production_depth:
            continue

        targets = (
            class_local_targets(
                caller=caller,
                class_method_index=(
                    class_method_index
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

            visited.add(key)

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

    cumulative = set()

    for depth in range(
        0,
        production_depth + 1,
    ):

        methods = (
            reachable_by_depth[
                depth
            ]
        )

        depth_keys = {
            method_key(candidate)
            for candidate in methods
        }

        cumulative.update(
            depth_keys
        )

        gt_here = (
            gt_keys
            & depth_keys
        )

        gt_cumulative = (
            gt_keys
            & cumulative
        )

        if (
            first_gt_depth is None
            and gt_here
        ):

            first_gt_depth = (
                depth
            )

        print()
        print("-" * 100)

        print(
            f"Production depth {depth}"
        )

        print(
            f"  New methods: "
            f"{len(methods)}"
        )

        print(
            f"  Cumulative: "
            f"{len(cumulative)}"
        )

        print(
            f"  GT at depth: "
            f"{len(gt_here)}"
        )

        print(
            f"  GT cumulative: "
            f"{len(gt_cumulative)}"
            f"/{len(gt_keys)}"
        )

        for candidate in methods:

            key = method_key(
                candidate
            )

            marker = (
                " <GT>"
                if key in gt_keys
                else ""
            )

            print(
                "  "
                f"{candidate.class_name}."
                f"{candidate.method.name} "
                f"[{candidate.method.start_line}-"
                f"{candidate.method.end_line}]"
                f"{marker}"
            )

    print()
    print("=" * 100)
    print("Feasibility Result")
    print("=" * 100)

    if first_gt_depth is None:

        print(
            "GT reachable: NO"
        )

    else:

        print(
            "GT reachable: YES"
        )

        print(
            f"First GT production depth: "
            f"{first_gt_depth}"
        )

    print(
        f"Test helpers visited: "
        f"{len(all_helper_names)}"
    )

    print(
        f"Production entry classes: "
        f"{len(all_entry_classes)}"
    )

    print(
        f"Production entry methods: "
        f"{len(entries)}"
    )

    print(
        f"Total production reachable: "
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
        "--test-helper-depth",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--production-depth",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    inspect_bug(
        project=args.project,
        bug_id=args.bug_id,
        test_helper_depth=(
            args.test_helper_depth
        ),
        production_depth=(
            args.production_depth
        ),
    )


if __name__ == "__main__":
    main()