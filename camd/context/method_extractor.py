from dataclasses import dataclass
from pathlib import Path

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser


JAVA_LANGUAGE = Language(
    tsjava.language()
)


@dataclass
class JavaMethod:
    name: str
    start_line: int
    end_line: int
    code: str
    node_type: str
    parameter_count: int


class JavaMethodExtractor:

    def __init__(self):
        self.parser = Parser(
            JAVA_LANGUAGE
        )

    def extract(
        self,
        source_file: Path,
    ) -> list[JavaMethod]:

        source_bytes = (
            source_file.read_bytes()
        )

        tree = self.parser.parse(
            source_bytes
        )

        methods: list[JavaMethod] = []

        self._walk(
            node=tree.root_node,
            source_bytes=source_bytes,
            methods=methods,
        )

        return methods

    def _walk(
        self,
        node,
        source_bytes: bytes,
        methods: list[JavaMethod],
    ) -> None:

        if node.type in {
            "method_declaration",
            "constructor_declaration",
        }:

            method = self._build_method(
                node=node,
                source_bytes=source_bytes,
            )

            if method is not None:
                methods.append(
                    method
                )

        for child in node.children:

            self._walk(
                node=child,
                source_bytes=source_bytes,
                methods=methods,
            )

    def _build_method(
        self,
        node,
        source_bytes: bytes,
    ) -> JavaMethod | None:

        name_node = (
            node.child_by_field_name(
                "name"
            )
        )

        if name_node is None:
            return None

        name = source_bytes[
            name_node.start_byte:
            name_node.end_byte
        ].decode(
            "utf-8"
        )

        parameters_node = (
            node.child_by_field_name(
                "parameters"
            )
        )

        parameter_count = (
            self._count_parameters(
                parameters_node
            )
        )

        code = source_bytes[
            node.start_byte:
            node.end_byte
        ].decode(
            "utf-8"
        )

        return JavaMethod(
            name=name,
            start_line=(
                node.start_point[0]
                + 1
            ),
            end_line=(
                node.end_point[0]
                + 1
            ),
            code=code,
            node_type=node.type,
            parameter_count=(
                parameter_count
            ),
        )

    @staticmethod
    def _count_parameters(
        parameters_node,
    ) -> int:

        if parameters_node is None:
            return 0

        parameter_types = {
            "formal_parameter",
            "spread_parameter",
            "receiver_parameter",
        }

        count = 0

        for child in (
            parameters_node.named_children
        ):

            if child.type in parameter_types:
                count += 1

        return count


def extract_java_methods(
    source_file: Path,
) -> list[JavaMethod]:

    extractor = (
        JavaMethodExtractor()
    )

    return extractor.extract(
        source_file
    )