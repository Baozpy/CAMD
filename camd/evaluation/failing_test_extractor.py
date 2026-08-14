import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser


JAVA_LANGUAGE = Language(
    tsjava.language()
)


@dataclass
class FailingTest:
    full_name: str
    class_name: str
    method_name: str

    source_file: Path | None
    start_line: int | None
    end_line: int | None
    code: str | None


class FailingTestExtractor:

    FAILING_TEST_PATTERN = re.compile(
        r"^\s*-\s+(.+?)::(.+?)\s*$"
    )

    def __init__(
        self,
        checkout_dir: Path,
    ):
        self.checkout_dir = checkout_dir

        self.parser = Parser(
            JAVA_LANGUAGE
        )

    def run_defects4j_test(
        self,
    ) -> list[tuple[str, str]]:

        result = subprocess.run(
            [
                "defects4j",
                "test",
            ],
            cwd=self.checkout_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        output = (
            result.stdout
            + "\n"
            + result.stderr
        )

        failing_tests = []

        for line in output.splitlines():

            match = (
                self.FAILING_TEST_PATTERN
                .match(line)
            )

            if not match:
                continue

            class_name = (
                match.group(1).strip()
            )

            method_name = (
                match.group(2).strip()
            )

            failing_tests.append(
                (
                    class_name,
                    method_name,
                )
            )

        return failing_tests

    def extract(
        self,
    ) -> list[FailingTest]:

        failing_names = (
            self.run_defects4j_test()
        )

        results = []

        for class_name, method_name in (
            failing_names
        ):

            source_file = (
                self._find_test_source(
                    class_name
                )
            )

            if source_file is None:

                results.append(
                    FailingTest(
                        full_name=(
                            f"{class_name}::"
                            f"{method_name}"
                        ),
                        class_name=class_name,
                        method_name=method_name,
                        source_file=None,
                        start_line=None,
                        end_line=None,
                        code=None,
                    )
                )

                continue

            method_info = (
                self._extract_test_method(
                    source_file=source_file,
                    method_name=method_name,
                )
            )

            if method_info is None:

                results.append(
                    FailingTest(
                        full_name=(
                            f"{class_name}::"
                            f"{method_name}"
                        ),
                        class_name=class_name,
                        method_name=method_name,
                        source_file=source_file,
                        start_line=None,
                        end_line=None,
                        code=None,
                    )
                )

                continue

            start_line, end_line, code = (
                method_info
            )

            results.append(
                FailingTest(
                    full_name=(
                        f"{class_name}::"
                        f"{method_name}"
                    ),
                    class_name=class_name,
                    method_name=method_name,
                    source_file=source_file,
                    start_line=start_line,
                    end_line=end_line,
                    code=code,
                )
            )

        return results

    def _find_test_source(
        self,
        class_name: str,
    ) -> Path | None:

        relative_path = (
            Path(
                *class_name.split(".")
            )
            .with_suffix(".java")
        )

        candidates = list(
            self.checkout_dir.rglob(
                relative_path.name
            )
        )

        for candidate in candidates:

            normalized = str(
                candidate
            ).replace(
                "\\",
                "/",
            )

            expected_suffix = str(
                relative_path
            ).replace(
                "\\",
                "/",
            )

            if normalized.endswith(
                expected_suffix
            ):
                return candidate

        return None

    def _extract_test_method(
        self,
        source_file: Path,
        method_name: str,
    ) -> (
        tuple[int, int, str]
        | None
    ):

        source_bytes = (
            source_file.read_bytes()
        )

        tree = self.parser.parse(
            source_bytes
        )

        target_node = (
            self._find_method_node(
                node=tree.root_node,
                source_bytes=source_bytes,
                method_name=method_name,
            )
        )

        if target_node is None:
            return None

        start_line = (
            target_node.start_point[0]
            + 1
        )

        end_line = (
            target_node.end_point[0]
            + 1
        )

        code = source_bytes[
            target_node.start_byte:
            target_node.end_byte
        ].decode(
            "utf-8"
        )

        return (
            start_line,
            end_line,
            code,
        )

    def _find_method_node(
        self,
        node,
        source_bytes: bytes,
        method_name: str,
    ):

        if node.type == "method_declaration":

            name_node = (
                node.child_by_field_name(
                    "name"
                )
            )

            if name_node is not None:

                current_name = (
                    source_bytes[
                        name_node.start_byte:
                        name_node.end_byte
                    ].decode(
                        "utf-8"
                    )
                )

                if (
                    current_name
                    == method_name
                ):
                    return node

        for child in node.children:

            result = (
                self._find_method_node(
                    node=child,
                    source_bytes=source_bytes,
                    method_name=method_name,
                )
            )

            if result is not None:
                return result

        return None