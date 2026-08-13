from dataclasses import dataclass

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from camd.context.method_extractor import (
    JavaMethod,
)


JAVA_LANGUAGE = Language(
    tsjava.language()
)


@dataclass
class MethodCall:
    caller: str
    callee: str
    line: int
    argument_count: int

    qualifier: str | None


class JavaCallExtractor:

    def __init__(self):

        self.parser = Parser(
            JAVA_LANGUAGE
        )

    def extract_calls(
        self,
        method: JavaMethod,
    ) -> list[MethodCall]:

        source_bytes = (
            method.code.encode(
                "utf-8"
            )
        )

        tree = self.parser.parse(
            source_bytes
        )

        calls: list[MethodCall] = []

        self._walk(
            node=tree.root_node,
            source_bytes=source_bytes,
            caller=method.name,
            method_start_line=(
                method.start_line
            ),
            calls=calls,
        )

        return calls

    def _walk(
        self,
        node,
        source_bytes: bytes,
        caller: str,
        method_start_line: int,
        calls: list[MethodCall],
    ) -> None:

        if node.type == "method_invocation":

            name_node = (
                node.child_by_field_name(
                    "name"
                )
            )

            arguments_node = (
                node.child_by_field_name(
                    "arguments"
                )
            )

            object_node = (
                node.child_by_field_name(
                    "object"
                )
            )

            if name_node is not None:

                callee = source_bytes[
                    name_node.start_byte:
                    name_node.end_byte
                ].decode(
                    "utf-8"
                )

                argument_count = (
                    self._count_arguments(
                        arguments_node
                    )
                )

                qualifier = None

                if object_node is not None:

                    qualifier = source_bytes[
                        object_node.start_byte:
                        object_node.end_byte
                    ].decode(
                        "utf-8"
                    )

                absolute_line = (
                    method_start_line
                    + node.start_point[0]
                )

                calls.append(
                    MethodCall(
                        caller=caller,
                        callee=callee,
                        line=absolute_line,
                        argument_count=(
                            argument_count
                        ),
                        qualifier=qualifier,
                    )
                )

        for child in node.children:

            self._walk(
                node=child,
                source_bytes=source_bytes,
                caller=caller,
                method_start_line=(
                    method_start_line
                ),
                calls=calls,
            )

    @staticmethod
    def _count_arguments(
        arguments_node,
    ) -> int:

        if arguments_node is None:
            return 0

        return len(
            arguments_node.named_children
        )


def extract_method_calls(
    method: JavaMethod,
) -> list[MethodCall]:

    extractor = (
        JavaCallExtractor()
    )

    return extractor.extract_calls(
        method
    )