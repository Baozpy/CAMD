from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from camd.context.method_extractor import (
    JavaMethod,
    extract_java_methods,
)
from camd.retrieval.failure_trace_parser import (
    FailureTrace,
)


TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
)


JAVA_KEYWORDS = {
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
    "null",
    "true",
    "false",
}


FRAMEWORK_PREFIXES = (
    "junit.",
    "org.junit.",
    "java.",
    "javax.",
    "sun.",
    "jdk.",
    "org.apache.tools.ant.",
    "org.gradle.",
)


@dataclass(frozen=True)
class ProgramMethod:
    project: str
    bug_id: int

    class_name: str
    source_file: str

    method: JavaMethod


@dataclass
class RetrievedMethod:
    candidate: ProgramMethod

    score: float
    base_score: float

    direct_method_reference: float
    class_reference: float
    name_overlap: float
    test_name_overlap: float
    lexical_overlap: float

    stack_score: float
    stack_exact_match: float
    stack_class_match: float
    stack_depth: int | None


def split_identifier(
    text: str,
) -> list[str]:

    if not text:
        return []

    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text,
    )

    text = text.replace(
        "_",
        " ",
    )

    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(
            text
        )
        if token
    ]


def tokenize(
    text: str,
) -> set[str]:

    tokens = set()

    for raw_token in TOKEN_PATTERN.findall(
        text or ""
    ):
        for token in split_identifier(
            raw_token
        ):
            if (
                len(token) >= 2
                and token not in JAVA_KEYWORDS
            ):
                tokens.add(
                    token
                )

    return tokens


def jaccard(
    left: set[str],
    right: set[str],
) -> float:

    if not left or not right:
        return 0.0

    union = left | right

    if not union:
        return 0.0

    return (
        len(left & right)
        / len(union)
    )


class ProgramMethodRetriever:

    def __init__(
        self,
        project: str,
        bug_id: int,
        use_stack_trace: bool = False,
        stack_weight: float = 0.50,
    ):
        self.project = project
        self.bug_id = bug_id

        self.use_stack_trace = (
            use_stack_trace
        )

        self.stack_weight = float(
            stack_weight
        )

    # =========================================================
    # Program-wide extraction
    # =========================================================

    def extract_program_methods(
        self,
        source_root: Path,
    ) -> list[ProgramMethod]:

        methods = []

        java_files = sorted(
            source_root.rglob(
                "*.java"
            )
        )

        for source_file in java_files:

            try:
                extracted = (
                    extract_java_methods(
                        source_file
                    )
                )

            except Exception as exc:

                print(
                    "Skipping parser failure: "
                    f"{source_file}"
                )

                print(
                    f"  Reason: {exc}"
                )

                continue

            try:
                relative = (
                    source_file.relative_to(
                        source_root
                    )
                )

                class_name = (
                    ".".join(
                        relative
                        .with_suffix("")
                        .parts
                    )
                )

            except ValueError:

                class_name = (
                    source_file.stem
                )

            for method in extracted:

                methods.append(
                    ProgramMethod(
                        project=self.project,
                        bug_id=self.bug_id,
                        class_name=class_name,
                        source_file=str(
                            source_file
                        ),
                        method=method,
                    )
                )

        return methods

    # =========================================================
    # V1 signals
    # =========================================================

    @staticmethod
    def _direct_method_reference(
        method: JavaMethod,
        failing_text: str,
    ) -> float:

        if not method.name:
            return 0.0

        pattern = re.compile(
            rf"\b{re.escape(method.name)}\s*\("
        )

        return (
            1.0
            if pattern.search(
                failing_text
            )
            else 0.0
        )

    @staticmethod
    def _class_reference(
        class_name: str,
        failing_text: str,
    ) -> float:

        simple_class_name = (
            class_name.split(".")[-1]
        )

        if (
            simple_class_name
            and simple_class_name
            in failing_text
        ):
            return 1.0

        if (
            class_name
            and class_name
            in failing_text
        ):
            return 1.0

        return 0.0

    @staticmethod
    def _name_overlap(
        method: JavaMethod,
        query_tokens: set[str],
    ) -> float:

        method_tokens = set(
            split_identifier(
                method.name
            )
        )

        return jaccard(
            method_tokens,
            query_tokens,
        )

    @staticmethod
    def _test_name_overlap(
        method: JavaMethod,
        failing_test_names: list[str],
    ) -> float:

        method_tokens = set(
            split_identifier(
                method.name
            )
        )

        if not method_tokens:
            return 0.0

        best_score = 0.0

        for test_name in (
            failing_test_names
        ):

            test_method_name = (
                test_name.split(
                    "::"
                )[-1]
            )

            test_tokens = set(
                split_identifier(
                    test_method_name
                )
            )

            score = jaccard(
                method_tokens,
                test_tokens,
            )

            best_score = max(
                best_score,
                score,
            )

        return best_score

    @staticmethod
    def _lexical_overlap(
        method: JavaMethod,
        query_tokens: set[str],
    ) -> float:

        method_tokens = tokenize(
            method.code
        )

        return jaccard(
            method_tokens,
            query_tokens,
        )

    # =========================================================
    # Stack trace
    # =========================================================

    @staticmethod
    def _normalize_class_name(
        class_name: str,
    ) -> str:

        if not class_name:
            return ""

        # Inner / anonymous class:
        # Foo$Bar -> Foo
        return class_name.split(
            "$",
            1,
        )[0]

    @staticmethod
    def _is_framework_frame(
        class_name: str,
    ) -> bool:

        return class_name.startswith(
            FRAMEWORK_PREFIXES
        )

    @classmethod
    def _production_frames(
        cls,
        traces: list[FailureTrace],
    ) -> list[
        tuple[
            str,
            str,
            str | None,
            int | None,
            int,
        ]
    ]:

        output = []

        for trace in traces:

            test_class = (
                cls._normalize_class_name(
                    trace.test_class
                )
            )

            production_depth = 0

            for frame in trace.frames:

                frame_class = (
                    cls._normalize_class_name(
                        frame.class_name
                    )
                )

                if cls._is_framework_frame(
                    frame_class
                ):
                    continue

                # Exclude frames belonging to
                # the failing test class itself.
                if (
                    frame_class
                    == test_class
                ):
                    continue

                output.append(
                    (
                        frame_class,
                        frame.method_name,
                        frame.file_name,
                        frame.line_number,
                        production_depth,
                    )
                )

                production_depth += 1

        return output

    @classmethod
    def _stack_signal(
        cls,
        candidate: ProgramMethod,
        traces: list[FailureTrace],
    ) -> tuple[
        float,
        float,
        float,
        int | None,
    ]:

        if not traces:

            return (
                0.0,
                0.0,
                0.0,
                None,
            )

        candidate_class = (
            cls._normalize_class_name(
                candidate.class_name
            )
        )

        candidate_method = (
            candidate.method.name
        )

        candidate_start = (
            candidate.method.start_line
        )

        candidate_end = (
            candidate.method.end_line
        )

        best_score = 0.0
        best_exact = 0.0
        best_class = 0.0
        best_depth = None

        production_frames = (
            cls._production_frames(
                traces
            )
        )

        for (
            frame_class,
            frame_method,
            frame_file,
            frame_line,
            depth,
        ) in production_frames:

            if (
                frame_class
                != candidate_class
            ):
                continue

            # -------------------------------------------------
            # Strongest signal:
            # stack frame line is physically inside
            # the candidate method.
            # -------------------------------------------------

            line_match = (
                frame_line is not None
                and candidate_start
                is not None
                and candidate_end
                is not None
                and (
                    candidate_start
                    <= frame_line
                    <= candidate_end
                )
            )

            method_name_match = (
                frame_method
                == candidate_method
            )

            if line_match:

                score = (
                    1.0
                    / (
                        1.0
                        + depth
                    )
                )

                exact_score = score
                class_score = 0.0

            elif method_name_match:

                # Method names agree, but we either
                # do not have a usable source line
                # or the line could not be aligned.
                score = (
                    0.75
                    / (
                        1.0
                        + depth
                    )
                )

                exact_score = score
                class_score = 0.0

            else:

                # Same-class-only evidence is intentionally
                # very weak because a class may contain
                # dozens or hundreds of methods.
                score = (
                    0.05
                    / (
                        1.0
                        + depth
                    )
                )

                exact_score = 0.0
                class_score = score

            if score > best_score:

                best_score = score
                best_exact = exact_score
                best_class = class_score
                best_depth = depth

        return (
            best_score,
            best_exact,
            best_class,
            best_depth,
        )


    # =========================================================
    # Ranking
    # =========================================================

    def rank(
        self,
        methods: list[ProgramMethod],
        failing_test_names: list[str],
        failing_test_text: str,
        failure_traces: list[FailureTrace] | None = None,
    ) -> list[RetrievedMethod]:

        query_text = "\n".join(
            failing_test_names
        )

        query_text += "\n"
        query_text += (
            failing_test_text or ""
        )

        query_tokens = tokenize(
            query_text
        )

        traces = (
            failure_traces
            if failure_traces is not None
            else []
        )

        ranked = []

        for candidate in methods:

            method = (
                candidate.method
            )

            direct_method_reference = (
                self._direct_method_reference(
                    method=method,
                    failing_text=query_text,
                )
            )

            class_reference = (
                self._class_reference(
                    class_name=(
                        candidate.class_name
                    ),
                    failing_text=query_text,
                )
            )

            name_overlap = (
                self._name_overlap(
                    method=method,
                    query_tokens=query_tokens,
                )
            )

            test_name_overlap = (
                self._test_name_overlap(
                    method=method,
                    failing_test_names=(
                        failing_test_names
                    ),
                )
            )

            lexical_overlap = (
                self._lexical_overlap(
                    method=method,
                    query_tokens=query_tokens,
                )
            )

            base_score = (
                0.35
                * direct_method_reference
                + 0.20
                * class_reference
                + 0.20
                * name_overlap
                + 0.10
                * test_name_overlap
                + 0.15
                * lexical_overlap
            )

            (
                stack_score,
                stack_exact_match,
                stack_class_match,
                stack_depth,
            ) = (
                self._stack_signal(
                    candidate=(
                        candidate
                    ),
                    traces=traces,
                )
                if self.use_stack_trace
                else (
                    0.0,
                    0.0,
                    0.0,
                    None,
                )
            )

            score = (
                base_score
                + self.stack_weight
                * stack_score
            )

            ranked.append(
                RetrievedMethod(
                    candidate=candidate,
                    score=score,
                    base_score=base_score,
                    direct_method_reference=(
                        direct_method_reference
                    ),
                    class_reference=(
                        class_reference
                    ),
                    name_overlap=(
                        name_overlap
                    ),
                    test_name_overlap=(
                        test_name_overlap
                    ),
                    lexical_overlap=(
                        lexical_overlap
                    ),
                    stack_score=(
                        stack_score
                    ),
                    stack_exact_match=(
                        stack_exact_match
                    ),
                    stack_class_match=(
                        stack_class_match
                    ),
                    stack_depth=(
                        stack_depth
                    ),
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.score,
                -item.stack_score,
                -item.base_score,
                item.candidate.class_name,
                item.candidate.method.start_line,
                item.candidate.method.name,
            )
        )

        return ranked