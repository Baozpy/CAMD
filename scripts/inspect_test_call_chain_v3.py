from __future__ import annotations

import argparse
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from camd.context.method_extractor import extract_java_methods
from camd.evaluation.experiment_runner import (
    Defects4JExperimentRunner,
)
from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
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


CALL_PATTERN = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
)

NEW_CLASS_PATTERN = re.compile(
    r"\bnew\s+([A-Z][A-Za-z0-9_$.]*)\s*\("
)

DECLARATION_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9_$.<>?,\s]*)\s+"
    r"([a-zA-Z_$][A-Za-z0-9_$]*)\s*="
)

RECEIVER_CALL_PATTERN = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_$]*)"
    r"\."
    r"([A-Za-z_$][A-Za-z0-9_$]*)"
    r"\s*\("
)

CLASS_DECLARATION_PATTERN = re.compile(
    r"\bclass\s+"
    r"(?P<class>[A-Za-z_$][A-Za-z0-9_$]*)"
    r"(?:\s+extends\s+"
    r"(?P<extends>[A-Za-z_$][A-Za-z0-9_$.]*))?"
    r"(?:\s+implements\s+"
    r"(?P<implements>[A-Za-z0-9_$.,\s]+))?"
)

INTERFACE_DECLARATION_PATTERN = re.compile(
    r"\binterface\s+"
    r"(?P<class>[A-Za-z_$][A-Za-z0-9_$]*)"
    r"(?:\s+extends\s+"
    r"(?P<extends>[A-Za-z0-9_$.,\s]+))?"
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
    method_name: str
    code: str


@dataclass
class ClassInfo:
    class_name: str
    simple_name: str
    source_file: str
    super_classes: list[str]
    interfaces: list[str]


@dataclass
class CallState:
    candidate: ProgramMethod
    runtime_class: str
    depth: int


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

    return (
        class_name
        .split(".")[-1]
        .split("$")[0]
    )


def clean_type_name(
    value: str,
) -> str:

    value = value.strip()

    value = re.sub(
        r"<.*?>",
        "",
        value,
    )

    value = value.replace(
        "?",
        "",
    )

    value = value.strip()

    if " " in value:
        value = value.split()[-1]

    return value


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
            result.add(
                name
            )

    return result


def extract_new_classes(
    code: str | None,
) -> set[str]:

    if not code:
        return set()

    return {
        clean_type_name(
            match.group(1)
        )
        for match in NEW_CLASS_PATTERN.finditer(
            code
        )
    }


def extract_local_types(
    code: str | None,
) -> dict[str, str]:

    if not code:
        return {}

    result = {}

    for match in DECLARATION_PATTERN.finditer(
        code
    ):

        raw_type = (
            match.group(1)
        )

        variable_name = (
            match.group(2)
        )

        type_name = (
            clean_type_name(
                raw_type
            )
        )

        if type_name:

            result[
                variable_name
            ] = type_name

    return result


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

    return [
        (
            match.group(1),
            match.group(2),
        )
        for match
        in RECEIVER_CALL_PATTERN.finditer(
            code
        )
    ]


# =========================================================
# Ground truth
# =========================================================

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


# =========================================================
# Program indexes
# =========================================================

def build_class_method_index(
    methods: list[ProgramMethod],
):

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


def build_simple_class_index(
    methods: list[ProgramMethod],
):

    index = defaultdict(list)

    for candidate in methods:

        simple = (
            simple_class_name(
                candidate.class_name
            )
        )

        if (
            candidate.class_name
            not in index[simple]
        ):

            index[
                simple
            ].append(
                candidate.class_name
            )

    return index


def build_class_file_index(
    methods: list[ProgramMethod],
) -> dict[
    str,
    Path,
]:

    result = {}

    for candidate in methods:

        result.setdefault(
            candidate.class_name,
            Path(
                candidate.source_file
            ),
        )

    return result


# =========================================================
# Inheritance
# =========================================================

def parse_class_info(
    class_name: str,
    source_file: Path,
) -> ClassInfo:

    text = source_file.read_text(
        encoding="utf-8",
        errors="replace",
    )

    simple_name = (
        simple_class_name(
            class_name
        )
    )

    super_classes = []
    interfaces = []

    for match in CLASS_DECLARATION_PATTERN.finditer(
        text
    ):

        declared_name = (
            match.group(
                "class"
            )
        )

        if declared_name != simple_name:
            continue

        extends_name = (
            match.group(
                "extends"
            )
        )

        implements_text = (
            match.group(
                "implements"
            )
        )

        if extends_name:

            super_classes.append(
                clean_type_name(
                    extends_name
                )
            )

        if implements_text:

            for item in (
                implements_text.split(",")
            ):

                item = (
                    clean_type_name(
                        item
                    )
                )

                if item:

                    interfaces.append(
                        item
                    )

        break

    if not super_classes:

        for match in (
            INTERFACE_DECLARATION_PATTERN
            .finditer(
                text
            )
        ):

            declared_name = (
                match.group(
                    "class"
                )
            )

            if (
                declared_name
                != simple_name
            ):
                continue

            extends_text = (
                match.group(
                    "extends"
                )
            )

            if extends_text:

                for item in (
                    extends_text.split(",")
                ):

                    item = (
                        clean_type_name(
                            item
                        )
                    )

                    if item:

                        interfaces.append(
                            item
                        )

            break

    return ClassInfo(
        class_name=class_name,
        simple_name=simple_name,
        source_file=str(
            source_file
        ),
        super_classes=(
            super_classes
        ),
        interfaces=(
            interfaces
        ),
    )


def build_class_info_index(
    methods: list[ProgramMethod],
) -> dict[
    str,
    ClassInfo,
]:

    class_file_index = (
        build_class_file_index(
            methods
        )
    )

    result = {}

    for (
        class_name,
        source_file,
    ) in class_file_index.items():

        try:

            result[
                class_name
            ] = parse_class_info(
                class_name,
                source_file,
            )

        except Exception:

            result[
                class_name
            ] = ClassInfo(
                class_name=(
                    class_name
                ),
                simple_name=(
                    simple_class_name(
                        class_name
                    )
                ),
                source_file=str(
                    source_file
                ),
                super_classes=[],
                interfaces=[],
            )

    return result


def resolve_simple_class(
    type_name: str,
    simple_class_index,
    preferred_package: str | None = None,
) -> list[str]:

    clean = clean_type_name(
        type_name
    )

    if not clean:
        return []

    if "." in clean:

        direct = [
            class_name
            for candidates
            in simple_class_index.values()
            for class_name
            in candidates
            if class_name == clean
        ]

        if direct:
            return direct

        clean = clean.split(".")[-1]

    candidates = list(
        simple_class_index.get(
            clean,
            [],
        )
    )

    if (
        preferred_package
        and len(candidates) > 1
    ):

        preferred = [
            candidate
            for candidate
            in candidates
            if candidate.startswith(
                preferred_package
            )
        ]

        if preferred:
            return preferred

    return candidates


def build_ancestor_graph(
    class_info_index,
    simple_class_index,
) -> dict[
    str,
    list[str],
]:

    graph = {}

    for (
        class_name,
        info,
    ) in class_info_index.items():

        package = (
            class_name.rsplit(
                ".",
                1,
            )[0]
            if "."
            in class_name
            else ""
        )

        parents = []

        for raw_parent in (
            info.super_classes
            + info.interfaces
        ):

            resolved = (
                resolve_simple_class(
                    raw_parent,
                    simple_class_index,
                    preferred_package=(
                        package
                    ),
                )
            )

            for parent in resolved:

                if (
                    parent
                    not in parents
                ):
                    parents.append(
                        parent
                    )

        graph[
            class_name
        ] = parents

    return graph


def ancestor_closure(
    class_name: str,
    ancestor_graph,
    max_depth: int = 10,
) -> list[str]:

    result = []

    visited = {
        class_name
    }

    queue = deque(
        [
            (
                class_name,
                0,
            )
        ]
    )

    while queue:

        current, depth = (
            queue.popleft()
        )

        if depth >= max_depth:
            continue

        for parent in (
            ancestor_graph.get(
                current,
                []
            )
        ):

            if parent in visited:
                continue

            visited.add(
                parent
            )

            result.append(
                parent
            )

            queue.append(
                (
                    parent,
                    depth + 1,
                )
            )

    return result


# =========================================================
# Test helper closure
# =========================================================

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

    for method in methods:

        result[
            method.name
        ] = TestMethod(
            method_name=(
                method.name
            ),
            code=(
                method.code
            ),
        )

    return result


def build_test_helper_closure(
    failing_method_name: str,
    test_methods,
    max_depth: int,
) -> list[TestMethod]:

    queue = deque(
        [
            (
                failing_method_name,
                0,
            )
        ]
    )

    visited = set()

    result = []

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

        result.append(
            method
        )

        if depth >= max_depth:
            continue

        for called_name in (
            extract_called_names(
                method.code
            )
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

    return result


# =========================================================
# Method resolution
# =========================================================

def resolve_method_in_hierarchy(
    runtime_class: str,
    method_name: str,
    class_method_index,
    ancestor_graph,
) -> list[
    ProgramMethod
]:

    """
    Resolve a method against runtime class first,
    then ancestors/interfaces.
    """

    search_classes = [
        runtime_class
    ]

    search_classes.extend(
        ancestor_closure(
            runtime_class,
            ancestor_graph,
        )
    )

    for class_name in search_classes:

        targets = (
            class_method_index.get(
                (
                    class_name,
                    method_name,
                ),
                [],
            )
        )

        if targets:

            return list(
                targets
            )

    return []


def resolve_virtual_override(
    runtime_class: str,
    method_name: str,
    class_method_index,
) -> list[
    ProgramMethod
]:

    """
    If a parent method calls doOptimize(),
    dispatch back to runtime subclass override
    when available.
    """

    return list(
        class_method_index.get(
            (
                runtime_class,
                method_name,
            ),
            [],
        )
    )


# =========================================================
# Entry points from tests
# =========================================================

def infer_test_entry_states(
    helper_methods: list[TestMethod],
    simple_class_index,
    class_method_index,
    ancestor_graph,
) -> list[
    CallState
]:

    result = []

    seen = set()

    for helper in helper_methods:

        code = (
            helper.code
        )

        local_types = (
            extract_local_types(
                code
            )
        )

        # Variables constructed directly with new.
        constructor_types = (
            extract_new_classes(
                code
            )
        )

        # Add constructors.
        for raw_type in (
            constructor_types
        ):

            for runtime_class in (
                resolve_simple_class(
                    raw_type,
                    simple_class_index,
                )
            ):

                constructor_name = (
                    simple_class_name(
                        runtime_class
                    )
                )

                targets = (
                    class_method_index.get(
                        (
                            runtime_class,
                            constructor_name,
                        ),
                        [],
                    )
                )

                for target in targets:

                    key = (
                        method_key(
                            target
                        ),
                        runtime_class,
                    )

                    if key in seen:
                        continue

                    seen.add(
                        key
                    )

                    result.append(
                        CallState(
                            candidate=target,
                            runtime_class=(
                                runtime_class
                            ),
                            depth=0,
                        )
                    )

        # Receiver calls, including inherited methods.
        for (
            receiver,
            called_method,
        ) in extract_receiver_calls(
            code
        ):

            receiver_type = (
                local_types.get(
                    receiver
                )
            )

            if receiver_type is None:
                continue

            runtime_classes = (
                resolve_simple_class(
                    receiver_type,
                    simple_class_index,
                )
            )

            for runtime_class in (
                runtime_classes
            ):

                targets = (
                    resolve_method_in_hierarchy(
                        runtime_class=(
                            runtime_class
                        ),
                        method_name=(
                            called_method
                        ),
                        class_method_index=(
                            class_method_index
                        ),
                        ancestor_graph=(
                            ancestor_graph
                        ),
                    )
                )

                for target in targets:

                    key = (
                        method_key(
                            target
                        ),
                        runtime_class,
                    )

                    if key in seen:
                        continue

                    seen.add(
                        key
                    )

                    result.append(
                        CallState(
                            candidate=target,
                            runtime_class=(
                                runtime_class
                            ),
                            depth=0,
                        )
                    )

    return result


# =========================================================
# Production expansion
# =========================================================

def expand_state(
    state: CallState,
    class_method_index,
    ancestor_graph,
) -> list[
    CallState
]:

    caller = (
        state.candidate
    )

    runtime_class = (
        state.runtime_class
    )

    next_states = []

    seen = set()

    called_names = (
        extract_called_names(
            caller.method.code
        )
    )

    for called_name in sorted(
        called_names
    ):

        # -------------------------------------------------
        # 1. Dynamic override on original runtime class.
        # Example:
        # parent optimize() -> doOptimize()
        # dispatches to LMOptimizer.doOptimize()
        # -------------------------------------------------

        override_targets = (
            resolve_virtual_override(
                runtime_class=(
                    runtime_class
                ),
                method_name=(
                    called_name
                ),
                class_method_index=(
                    class_method_index
                ),
            )
        )

        for target in (
            override_targets
        ):

            key = (
                method_key(
                    target
                ),
                runtime_class,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            next_states.append(
                CallState(
                    candidate=target,
                    runtime_class=(
                        runtime_class
                    ),
                    depth=(
                        state.depth + 1
                    ),
                )
            )

        # -------------------------------------------------
        # 2. Normal same-declaring-class /
        # inherited lookup.
        # -------------------------------------------------

        declaring_class = (
            caller.class_name
        )

        normal_targets = (
            resolve_method_in_hierarchy(
                runtime_class=(
                    declaring_class
                ),
                method_name=(
                    called_name
                ),
                class_method_index=(
                    class_method_index
                ),
                ancestor_graph=(
                    ancestor_graph
                ),
            )
        )

        for target in normal_targets:

            key = (
                method_key(
                    target
                ),
                runtime_class,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            next_states.append(
                CallState(
                    candidate=target,
                    runtime_class=(
                        runtime_class
                    ),
                    depth=(
                        state.depth + 1
                    ),
                )
            )

    return next_states


# =========================================================
# Main diagnostic
# =========================================================

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
        retriever
        .extract_program_methods(
            source_root
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

    class_info_index = (
        build_class_info_index(
            program_methods
        )
    )

    ancestor_graph = (
        build_ancestor_graph(
            class_info_index,
            simple_class_index,
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
        f"Call-Chain v3 Feasibility: "
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
            f"[{gt.start_line}-{gt.end_line}]"
        )

    all_entry_states = []

    helper_names = set()

    for failing_test in (
        failing_tests
    ):

        source_file = (
            failing_test.source_file
        )

        if (
            source_file is None
            or not source_file.exists()
        ):

            continue

        test_methods = (
            extract_test_methods(
                source_file
            )
        )

        helpers = (
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

        helper_names.update(
            helper.method_name
            for helper in helpers
        )

        entry_states = (
            infer_test_entry_states(
                helper_methods=(
                    helpers
                ),
                simple_class_index=(
                    simple_class_index
                ),
                class_method_index=(
                    class_method_index
                ),
                ancestor_graph=(
                    ancestor_graph
                ),
            )
        )

        all_entry_states.extend(
            entry_states
        )

        print()

        print(
            f"Failing test: "
            f"{failing_test.full_name}"
        )

        print(
            "  Test helper closure:"
        )

        for helper in helpers:

            print(
                f"    {helper.method_name}"
            )

        print(
            "  Entry methods:"
        )

        if not entry_states:

            print(
                "    (none)"
            )

        for state in entry_states:

            inherited_marker = (
                ""
                if (
                    state.candidate.class_name
                    == state.runtime_class
                )
                else (
                    " "
                    f"[runtime="
                    f"{state.runtime_class}]"
                )
            )

            print(
                "    "
                f"{state.candidate.class_name}."
                f"{state.candidate.method.name} "
                f"[{state.candidate.method.start_line}-"
                f"{state.candidate.method.end_line}]"
                f"{inherited_marker}"
            )

    # -----------------------------------------------------
    # Deduplicate initial states
    # -----------------------------------------------------

    state_map = {}

    for state in (
        all_entry_states
    ):

        key = (
            method_key(
                state.candidate
            ),
            state.runtime_class,
        )

        state_map[
            key
        ] = state

    entry_states = list(
        state_map.values()
    )

    queue = deque(
        entry_states
    )

    visited_states = set()

    reachable_methods = {}

    reachable_by_depth = (
        defaultdict(list)
    )

    first_gt_depth = None

    while queue:

        state = (
            queue.popleft()
        )

        state_key = (
            method_key(
                state.candidate
            ),
            state.runtime_class,
        )

        if state_key in (
            visited_states
        ):
            continue

        visited_states.add(
            state_key
        )

        candidate_key = (
            method_key(
                state.candidate
            )
        )

        reachable_methods[
            candidate_key
        ] = (
            state.candidate
        )

        reachable_by_depth[
            state.depth
        ].append(
            state
        )

        if (
            first_gt_depth
            is None
            and candidate_key
            in gt_keys
        ):

            first_gt_depth = (
                state.depth
            )

        if (
            state.depth
            >= production_depth
        ):
            continue

        for next_state in (
            expand_state(
                state=state,
                class_method_index=(
                    class_method_index
                ),
                ancestor_graph=(
                    ancestor_graph
                ),
            )
        ):

            next_key = (
                method_key(
                    next_state.candidate
                ),
                next_state.runtime_class,
            )

            if (
                next_key
                not in visited_states
            ):

                queue.append(
                    next_state
                )

    print()
    print("=" * 100)
    print(
        "Production Expansion"
    )
    print("=" * 100)

    cumulative = set()

    for depth in range(
        0,
        production_depth + 1,
    ):

        states = (
            reachable_by_depth[
                depth
            ]
        )

        method_keys = {
            method_key(
                state.candidate
            )
            for state in states
        }

        cumulative.update(
            method_keys
        )

        gt_here = (
            gt_keys
            & method_keys
        )

        gt_cumulative = (
            gt_keys
            & cumulative
        )

        print()
        print("-" * 100)

        print(
            f"Production depth {depth}"
        )

        print(
            f"  New states: "
            f"{len(states)}"
        )

        print(
            f"  New unique methods: "
            f"{len(method_keys)}"
        )

        print(
            f"  Cumulative unique methods: "
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

        for state in states:

            key = method_key(
                state.candidate
            )

            marker = (
                " <GT>"
                if key in gt_keys
                else ""
            )

            runtime_marker = ""

            if (
                state.runtime_class
                != state.candidate.class_name
            ):

                runtime_marker = (
                    " "
                    f"[runtime="
                    f"{state.runtime_class}]"
                )

            print(
                "  "
                f"{state.candidate.class_name}."
                f"{state.candidate.method.name} "
                f"[{state.candidate.method.start_line}-"
                f"{state.candidate.method.end_line}]"
                f"{runtime_marker}"
                f"{marker}"
            )

    print()
    print("=" * 100)
    print(
        "Feasibility Result"
    )
    print("=" * 100)

    if (
        first_gt_depth
        is None
    ):

        print(
            "GT reachable: NO"
        )

    else:

        print(
            "GT reachable: YES"
        )

        print(
            "First GT production depth: "
            f"{first_gt_depth}"
        )

    print(
        f"Test helpers visited: "
        f"{len(helper_names)}"
    )

    print(
        f"Production entry states: "
        f"{len(entry_states)}"
    )

    print(
        "Total unique production "
        f"methods reachable: "
        f"{len(reachable_methods)}"
    )

    print(
        "Reachable fraction: "
        f"{len(reachable_methods) / len(program_methods):.4f}"
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
        default=5,
    )

    return parser.parse_args()


def main():

    args = (
        parse_args()
    )

    inspect_bug(
        project=(
            args.project
        ),
        bug_id=(
            args.bug_id
        ),
        test_helper_depth=(
            args.test_helper_depth
        ),
        production_depth=(
            args.production_depth
        ),
    )


if __name__ == "__main__":
    main()