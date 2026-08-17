from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FAILURE_HEADER_PATTERN = re.compile(
    r"^---\s+(.+?)::(.+?)\s*$"
)

STACK_FRAME_PATTERN = re.compile(
    r"^\s*at\s+"
    r"(?P<class>[A-Za-z_$][A-Za-z0-9_.$]*)"
    r"\."
    r"(?P<method>[A-Za-z_$<>][A-Za-z0-9_$<>]*)"
    r"\("
    r"(?P<location>[^)]*)"
    r"\)"
)


@dataclass(frozen=True)
class StackFrame:
    class_name: str
    method_name: str

    file_name: str | None
    line_number: int | None

    depth: int


@dataclass
class FailureTrace:
    test_class: str
    test_method: str

    exception_line: str | None

    frames: list[StackFrame]

    raw_text: str


class FailureTraceParser:

    def parse_file(
        self,
        path: Path,
    ) -> list[FailureTrace]:

        if not path.exists():
            return []

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return self.parse_text(
            text
        )

    def parse_text(
        self,
        text: str,
    ) -> list[FailureTrace]:

        lines = text.splitlines()

        traces: list[
            FailureTrace
        ] = []

        current_header = None
        current_lines: list[str] = []

        def flush() -> None:

            nonlocal current_header
            nonlocal current_lines

            if current_header is None:
                current_lines = []
                return

            (
                test_class,
                test_method,
            ) = current_header

            traces.append(
                self._build_trace(
                    test_class=(
                        test_class
                    ),
                    test_method=(
                        test_method
                    ),
                    lines=(
                        current_lines
                    ),
                )
            )

            current_header = None
            current_lines = []

        for line in lines:

            match = (
                FAILURE_HEADER_PATTERN
                .match(line)
            )

            if match:

                flush()

                current_header = (
                    match.group(1).strip(),
                    match.group(2).strip(),
                )

                continue

            if current_header is not None:

                current_lines.append(
                    line
                )

        flush()

        return traces

    def _build_trace(
        self,
        test_class: str,
        test_method: str,
        lines: list[str],
    ) -> FailureTrace:

        exception_line = None

        frames = []

        for line in lines:

            stripped = (
                line.strip()
            )

            if (
                exception_line is None
                and stripped
                and not stripped.startswith(
                    "at "
                )
                and not stripped.startswith(
                    "Caused by:"
                )
            ):
                exception_line = stripped

            frame_match = (
                STACK_FRAME_PATTERN
                .match(line)
            )

            if not frame_match:
                continue

            class_name = (
                frame_match.group(
                    "class"
                )
            )

            method_name = (
                frame_match.group(
                    "method"
                )
            )

            location = (
                frame_match.group(
                    "location"
                )
            )

            (
                file_name,
                line_number,
            ) = self._parse_location(
                location
            )

            frames.append(
                StackFrame(
                    class_name=(
                        class_name
                    ),
                    method_name=(
                        method_name
                    ),
                    file_name=(
                        file_name
                    ),
                    line_number=(
                        line_number
                    ),
                    depth=len(
                        frames
                    ),
                )
            )

        return FailureTrace(
            test_class=test_class,
            test_method=test_method,
            exception_line=(
                exception_line
            ),
            frames=frames,
            raw_text="\n".join(
                lines
            ),
        )

    @staticmethod
    def _parse_location(
        location: str,
    ) -> tuple[
        str | None,
        int | None,
    ]:

        location = (
            location.strip()
        )

        if ":" not in location:

            if (
                location
                and location
                not in {
                    "Native Method",
                    "Unknown Source",
                }
            ):
                return (
                    location,
                    None,
                )

            return (
                None,
                None,
            )

        filename, line_text = (
            location.rsplit(
                ":",
                1,
            )
        )

        try:

            line_number = int(
                line_text
            )

        except ValueError:

            line_number = None

        return (
            filename,
            line_number,
        )