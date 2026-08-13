import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from camd.context.method_extractor import (
    JavaMethod,
)
from camd.static.models import (
    StaticEvidence,
)


JAVA_LANGUAGE = Language(
    tsjava.language()
)


class JavaASTAnalyzer:

    CONDITION_NODE_TYPES = {
        "if_statement",
        "switch_expression",
        "switch_statement",
        "ternary_expression",
    }

    LOOP_NODE_TYPES = {
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
    }

    COMPARISON_OPERATORS = {
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
    }

    def __init__(self):

        self.parser = Parser(
            JAVA_LANGUAGE
        )

    def analyze(
        self,
        method: JavaMethod,
    ) -> StaticEvidence:

        source_bytes = (
            method.code.encode(
                "utf-8"
            )
        )

        tree = self.parser.parse(
            source_bytes
        )

        evidence = StaticEvidence(
            method_name=method.name,
            start_line=method.start_line,
            end_line=method.end_line,
        )

        self._walk(
            node=tree.root_node,
            source_bytes=source_bytes,
            evidence=evidence,
        )

        evidence.method_calls = (
            self._deduplicate(
                evidence.method_calls
            )
        )

        evidence.comparisons = (
            self._deduplicate(
                evidence.comparisons
            )
        )

        evidence.numeric_literals = (
            self._deduplicate(
                evidence.numeric_literals
            )
        )

        evidence.null_checks = (
            self._deduplicate(
                evidence.null_checks
            )
        )

        evidence.thrown_exceptions = (
            self._deduplicate(
                evidence.thrown_exceptions
            )
        )

        return evidence

    def _walk(
        self,
        node,
        source_bytes: bytes,
        evidence: StaticEvidence,
    ) -> None:

        if (
            node.type
            in self.CONDITION_NODE_TYPES
        ):
            evidence.condition_count += 1

        if (
            node.type
            in self.LOOP_NODE_TYPES
        ):
            evidence.loop_count += 1

        if node.type == "return_statement":
            evidence.return_count += 1

        if node.type == "throw_statement":
            evidence.throw_count += 1

            exception_text = (
                self._node_text(
                    node,
                    source_bytes,
                )
            )

            evidence.thrown_exceptions.append(
                exception_text
            )

        if node.type == "method_invocation":

            name_node = (
                node.child_by_field_name(
                    "name"
                )
            )

            if name_node is not None:

                method_name = (
                    self._node_text(
                        name_node,
                        source_bytes,
                    )
                )

                evidence.method_calls.append(
                    method_name
                )

        if node.type == "binary_expression":

            text = self._node_text(
                node,
                source_bytes,
            )

            if self._contains_comparison(
                node
            ):
                evidence.comparisons.append(
                    text
                )

                if "null" in text:
                    evidence.null_checks.append(
                        text
                    )

        if node.type in {
            "decimal_integer_literal",
            "hex_integer_literal",
            "octal_integer_literal",
            "binary_integer_literal",
            "decimal_floating_point_literal",
            "hex_floating_point_literal",
        }:

            literal = (
                self._node_text(
                    node,
                    source_bytes,
                )
            )

            evidence.numeric_literals.append(
                literal
            )

        for child in node.children:

            self._walk(
                node=child,
                source_bytes=source_bytes,
                evidence=evidence,
            )

    def _contains_comparison(
        self,
        node,
    ) -> bool:

        for child in node.children:

            if (
                child.type
                in self.COMPARISON_OPERATORS
            ):
                return True

        return False

    @staticmethod
    def _node_text(
        node,
        source_bytes: bytes,
    ) -> str:

        return source_bytes[
            node.start_byte:
            node.end_byte
        ].decode(
            "utf-8"
        )

    @staticmethod
    def _deduplicate(
        values: list[str],
    ) -> list[str]:

        seen = set()
        result = []

        for value in values:

            if value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result