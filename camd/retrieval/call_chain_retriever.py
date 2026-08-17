from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from camd.context.method_extractor import (
    extract_java_methods,
)
from camd.evaluation.failing_test_extractor import (
    FailingTest,
)
from camd.retrieval.program_method_retriever import (
    ProgramMethod,
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
class CallChainCandidate:
    candidate: ProgramMethod

    depth: int
    runtime_class: str

    evidence_type: str

    originating_test: str | None = None


@dataclass
class _CallState:
    candidate: ProgramMethod
    runtime_class: str
    depth: int
    originating_test: str | None


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
            result.add(name)

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

        type_name = clean_type_name(
            match.group(1)
        )

        variable_name = (
            match.group(2)
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


class CallChainRetriever:

    def __init__(
        self,
        max_test_helper_depth: int = 3,
        max_production_depth: int = 5,
    ):

        self.max_test_helper_depth = (
            max_test_helper_depth
        )

        self.max_production_depth = (
            max_production_depth
        )

    # =========================================================
    # Program indexes
    # =========================================================

    @staticmethod
    def _build_class_method_index(
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

    @staticmethod
    def _build_simple_class_index(
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

    @staticmethod
    def _build_class_file_index(
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

    @staticmethod
    def _parse_class_info(
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

        for match in (
            CLASS_DECLARATION_PATTERN
            .finditer(
                text
            )
        ):

            if (
                match.group("class")
                != simple_name
            ):
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
                    implements_text
                    .split(",")
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

                if (
                    match.group("class")
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
                        extends_text
                        .split(",")
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

    def _build_class_info_index(
        self,
        methods: list[ProgramMethod],
    ):

        class_files = (
            self._build_class_file_index(
                methods
            )
        )

        result = {}

        for (
            class_name,
            source_file,
        ) in class_files.items():

            try:

                result[
                    class_name
                ] = (
                    self._parse_class_info(
                        class_name,
                        source_file,
                    )
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

    @staticmethod
    def _resolve_simple_class(
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

            for candidates in (
                simple_class_index.values()
            ):

                if clean in candidates:
                    return [
                        clean
                    ]

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

    def _build_ancestor_graph(
        self,
        class_info_index,
        simple_class_index,
    ):

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
                    self._resolve_simple_class(
                        raw_parent,
                        simple_class_index,
                        preferred_package=(
                            package
                        ),
                    )
                )

                for parent in resolved:

                    if parent not in parents:
                        parents.append(
                            parent
                        )

            graph[
                class_name
            ] = parents

        return graph

    @staticmethod
    def _ancestor_closure(
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
                    [],
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

    @staticmethod
    def _extract_test_methods(
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

    def _build_test_helper_closure(
        self,
        failing_method_name: str,
        test_methods,
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

            if (
                depth
                >=
                self.max_test_helper_depth
            ):
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

    def _resolve_method_in_hierarchy(
        self,
        runtime_class: str,
        method_name: str,
        class_method_index,
        ancestor_graph,
    ) -> list[ProgramMethod]:

        search_classes = [
            runtime_class
        ]

        search_classes.extend(
            self._ancestor_closure(
                runtime_class,
                ancestor_graph,
            )
        )

        for class_name in (
            search_classes
        ):

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

    @staticmethod
    def _resolve_virtual_override(
        runtime_class: str,
        method_name: str,
        class_method_index,
    ) -> list[ProgramMethod]:

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
    # Test entry points
    # =========================================================

    def _infer_entry_states(
        self,
        helper_methods: list[TestMethod],
        simple_class_index,
        class_method_index,
        ancestor_graph,
        originating_test: str,
    ) -> list[_CallState]:

        result = []

        seen = set()

        for helper in helper_methods:

            code = helper.code

            local_types = (
                extract_local_types(
                    code
                )
            )

            # -----------------------------
            # Constructors
            # -----------------------------

            for raw_type in (
                extract_new_classes(
                    code
                )
            ):

                for runtime_class in (
                    self._resolve_simple_class(
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

                        seen.add(key)

                        result.append(
                            _CallState(
                                candidate=(
                                    target
                                ),
                                runtime_class=(
                                    runtime_class
                                ),
                                depth=0,
                                originating_test=(
                                    originating_test
                                ),
                            )
                        )

            # -----------------------------
            # Typed receiver calls
            # -----------------------------

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

                for runtime_class in (
                    self._resolve_simple_class(
                        receiver_type,
                        simple_class_index,
                    )
                ):

                    targets = (
                        self._resolve_method_in_hierarchy(
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

                        seen.add(key)

                        result.append(
                            _CallState(
                                candidate=(
                                    target
                                ),
                                runtime_class=(
                                    runtime_class
                                ),
                                depth=0,
                                originating_test=(
                                    originating_test
                                ),
                            )
                        )

        return result

    # =========================================================
    # Production expansion
    # =========================================================

    def _expand_state(
        self,
        state: _CallState,
        class_method_index,
        ancestor_graph,
    ) -> list[_CallState]:

        caller = (
            state.candidate
        )

        runtime_class = (
            state.runtime_class
        )

        result = []

        seen = set()

        for called_name in sorted(
            extract_called_names(
                caller.method.code
            )
        ):

            # -----------------------------
            # Runtime virtual dispatch
            # -----------------------------

            override_targets = (
                self._resolve_virtual_override(
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

                seen.add(key)

                result.append(
                    _CallState(
                        candidate=target,
                        runtime_class=(
                            runtime_class
                        ),
                        depth=(
                            state.depth
                            + 1
                        ),
                        originating_test=(
                            state
                            .originating_test
                        ),
                    )
                )

            # -----------------------------
            # Declaring-class /
            # ancestor resolution
            # -----------------------------

            targets = (
                self._resolve_method_in_hierarchy(
                    runtime_class=(
                        caller.class_name
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

            for target in targets:

                key = (
                    method_key(
                        target
                    ),
                    runtime_class,
                )

                if key in seen:
                    continue

                seen.add(key)

                result.append(
                    _CallState(
                        candidate=target,
                        runtime_class=(
                            runtime_class
                        ),
                        depth=(
                            state.depth
                            + 1
                        ),
                        originating_test=(
                            state
                            .originating_test
                        ),
                    )
                )

        return result

    # =========================================================
    # Public API
    # =========================================================

    def retrieve(
        self,
        program_methods: list[
            ProgramMethod
        ],
        failing_tests: list[
            FailingTest
        ],
    ) -> list[
        CallChainCandidate
    ]:

        if not program_methods:
            return []

        if not failing_tests:
            return []

        simple_class_index = (
            self._build_simple_class_index(
                program_methods
            )
        )

        class_method_index = (
            self._build_class_method_index(
                program_methods
            )
        )

        class_info_index = (
            self._build_class_info_index(
                program_methods
            )
        )

        ancestor_graph = (
            self._build_ancestor_graph(
                class_info_index,
                simple_class_index,
            )
        )

        initial_states = []

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
                self._extract_test_methods(
                    source_file
                )
            )

            helpers = (
                self._build_test_helper_closure(
                    failing_method_name=(
                        failing_test
                        .method_name
                    ),
                    test_methods=(
                        test_methods
                    ),
                )
            )

            entry_states = (
                self._infer_entry_states(
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
                    originating_test=(
                        failing_test.full_name
                    ),
                )
            )

            initial_states.extend(
                entry_states
            )

        # -----------------------------
        # Deduplicate entry states
        # -----------------------------

        entry_map = {}

        for state in (
            initial_states
        ):

            key = (
                method_key(
                    state.candidate
                ),
                state.runtime_class,
            )

            previous = (
                entry_map.get(
                    key
                )
            )

            if (
                previous is None
                or state.depth
                < previous.depth
            ):

                entry_map[
                    key
                ] = state

        queue = deque(
            entry_map.values()
        )

        visited_states = set()

        best_candidates: dict[
            MethodKey,
            CallChainCandidate,
        ] = {}

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

            if (
                state_key
                in visited_states
            ):
                continue

            visited_states.add(
                state_key
            )

            key = method_key(
                state.candidate
            )

            evidence_type = (
                "production_entry"
                if state.depth == 0
                else (
                    "call_chain"
                )
            )

            previous = (
                best_candidates.get(
                    key
                )
            )

            candidate_record = (
                CallChainCandidate(
                    candidate=(
                        state.candidate
                    ),
                    depth=(
                        state.depth
                    ),
                    runtime_class=(
                        state.runtime_class
                    ),
                    evidence_type=(
                        evidence_type
                    ),
                    originating_test=(
                        state
                        .originating_test
                    ),
                )
            )

            if (
                previous is None
                or state.depth
                < previous.depth
            ):

                best_candidates[
                    key
                ] = candidate_record

            if (
                state.depth
                >=
                self.max_production_depth
            ):
                continue

            for next_state in (
                self._expand_state(
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

        results = list(
            best_candidates.values()
        )

        results.sort(
            key=lambda item: (
                item.depth,
                item.candidate.class_name,
                item.candidate.method.start_line,
                item.candidate.method.name,
            )
        )

        return results