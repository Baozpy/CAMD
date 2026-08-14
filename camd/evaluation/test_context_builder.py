from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_java as tsjava


JAVA_LANGUAGE = Language(
    tsjava.language()
)


@dataclass
class TestHelperMethod:
    name: str
    start_line: int
    end_line: int
    code: str


@dataclass
class ExpandedTestContext:
    test_name: str
    test_code: str
    helpers: list[TestHelperMethod]

    def to_text(self) -> str:

        parts = []

        parts.append(
            "FAILING TEST"
        )

        parts.append(
            "=" * 70
        )

        parts.append(
            self.test_code
        )

        if self.helpers:

            parts.append("")
            parts.append(
                "DIRECT TEST HELPERS"
            )

            parts.append(
                "=" * 70
            )

            for helper in self.helpers:

                parts.append(
                    f"Helper: "
                    f"{helper.name} "
                    f"({helper.start_line}-"
                    f"{helper.end_line})"
                )

                parts.append("")
                parts.append(
                    helper.code
                )

                parts.append("")
                parts.append(
                    "-" * 70
                )

        return "\n".join(
            parts
        )


class TestContextBuilder:

    def __init__(
        self,
    ):
        self.parser = Parser(
            JAVA_LANGUAGE
        )

    def build(
        self,
        source_file: Path,
        test_method_name: str,
    ) -> ExpandedTestContext:

        source_bytes = (
            source_file.read_bytes()
        )

        tree = self.parser.parse(
            source_bytes
        )

        methods = (
            self._extract_methods(
                tree.root_node,
                source_bytes,
            )
        )

        test_method = next(
            (
                method
                for method in methods
                if (
                    method["name"]
                    == test_method_name
                )
            ),
            None,
        )

        if test_method is None:

            raise ValueError(
                f"Test method not found: "
                f"{test_method_name}"
            )

        called_methods = (
            self._extract_method_calls(
                test_method["node"],
                source_bytes,
            )
        )

        helpers = []

        seen = set()

        for call_name in called_methods:

            if call_name in seen:
                continue

            seen.add(
                call_name
            )

            helper_candidates = [
                method
                for method in methods
                if (
                    method["name"]
                    == call_name
                    and method["name"]
                    != test_method_name
                )
            ]

            if len(
                helper_candidates
            ) != 1:
                continue

            helper = (
                helper_candidates[0]
            )

            helpers.append(
                TestHelperMethod(
                    name=helper["name"],
                    start_line=(
                        helper["start_line"]
                    ),
                    end_line=(
                        helper["end_line"]
                    ),
                    code=(
                        helper["code"]
                    ),
                )
            )

        return ExpandedTestContext(
            test_name=(
                test_method_name
            ),
            test_code=(
                test_method["code"]
            ),
            helpers=helpers,
        )

    def _extract_methods(
        self,
        root_node,
        source_bytes: bytes,
    ) -> list[dict]:

        methods = []

        def visit(
            node,
        ):

            if (
                node.type
                == "method_declaration"
            ):

                name_node = (
                    node.child_by_field_name(
                        "name"
                    )
                )

                if name_node is not None:

                    name = source_bytes[
                        name_node.start_byte:
                        name_node.end_byte
                    ].decode(
                        "utf-8"
                    )

                    code = source_bytes[
                        node.start_byte:
                        node.end_byte
                    ].decode(
                        "utf-8"
                    )

                    methods.append(
                        {
                            "name": name,
                            "start_line": (
                                node.start_point[0]
                                + 1
                            ),
                            "end_line": (
                                node.end_point[0]
                                + 1
                            ),
                            "code": code,
                            "node": node,
                        }
                    )

            for child in node.children:

                visit(
                    child
                )

        visit(
            root_node
        )

        return methods

    def _extract_method_calls(
        self,
        method_node,
        source_bytes: bytes,
    ) -> list[str]:

        calls = []

        def visit(
            node,
        ):

            if (
                node.type
                == "method_invocation"
            ):

                name_node = (
                    node.child_by_field_name(
                        "name"
                    )
                )

                if (
                    name_node
                    is not None
                ):

                    name = source_bytes[
                        name_node.start_byte:
                        name_node.end_byte
                    ].decode(
                        "utf-8"
                    )

                    calls.append(
                        name
                    )

            for child in node.children:

                visit(
                    child
                )

        visit(
            method_node
        )

        return calls