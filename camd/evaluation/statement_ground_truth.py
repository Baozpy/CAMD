from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_java


@dataclass
class StatementRegion:
    start_line: int
    end_line: int
    node_type: str
    anchor_lines: list[int]


RELEVANT_NODE_TYPES = {
    "assert_statement",
    "break_statement",
    "continue_statement",
    "expression_statement",
    "local_variable_declaration",
    "return_statement",
    "throw_statement",
    "if_statement",
}

# RELEVANT_NODE_TYPES = {
#     "assert_statement",
#     "break_statement",
#     "continue_statement",
#     "do_statement",
#     "enhanced_for_statement",
#     "expression_statement",
#     "for_statement",
#     "if_statement",
#     "local_variable_declaration",
#     "return_statement",
#     "switch_expression",
#     "switch_statement",
#     "synchronized_statement",
#     "throw_statement",
#     "try_statement",
#     "while_statement",
# }


class StatementGroundTruthBuilder:

    def __init__(
        self,
        anchor_tolerance: int = 2,
    ):
        self.anchor_tolerance = anchor_tolerance

        self.language = Language(
            tree_sitter_java.language()
        )

        self.parser = Parser(
            self.language
        )

    @staticmethod
    def _node_start_line(node) -> int:
        return node.start_point[0] + 1

    @staticmethod
    def _node_end_line(node) -> int:
        return node.end_point[0] + 1

    def _collect_relevant_nodes(
        self,
        node,
        output: list,
    ) -> None:

        if node.type in RELEVANT_NODE_TYPES:
            output.append(node)

        for child in node.children:
            self._collect_relevant_nodes(
                child,
                output,
            )

    @staticmethod
    def _distance_to_range(
        line: int,
        start_line: int,
        end_line: int,
    ) -> int:

        if start_line <= line <= end_line:
            return 0

        if line < start_line:
            return start_line - line

        return line - end_line

    def build(
        self,
        source_file: Path,
        ground_truth_lines: list[int],
    ) -> list[StatementRegion]:

        source_bytes = source_file.read_bytes()

        tree = self.parser.parse(
            source_bytes
        )

        nodes = []

        self._collect_relevant_nodes(
            tree.root_node,
            nodes,
        )

        regions = []

        for anchor_line in ground_truth_lines:

            nearby = []

            for node in nodes:

                start_line = (
                    self._node_start_line(
                        node
                    )
                )

                end_line = (
                    self._node_end_line(
                        node
                    )
                )

                distance = (
                    self._distance_to_range(
                        anchor_line,
                        start_line,
                        end_line,
                    )
                )

                if (
                    distance
                    <= self.anchor_tolerance
                ):
                    nearby.append(
                        (
                            distance,
                            end_line - start_line,
                            start_line,
                            end_line,
                            node.type,
                        )
                    )

            if not nearby:
                continue

            nearby.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )

            best_distance = nearby[0][0]

            # Keep only nodes at the closest
            # anchor distance. Among overlapping
            # AST nodes this prevents a very broad
            # method/control region from dominating.
            closest = [
                item
                for item in nearby
                if item[0] == best_distance
            ]

            # Prefer smaller / more specific nodes,
            # but preserve all equally close useful
            # statements when necessary.
            min_span = min(
                item[1]
                for item in closest
            )

            selected = [
                item
                for item in closest
                if (
                    item[1]
                    <= min_span + 2
                )
            ]

            for (
                _,
                _,
                start_line,
                end_line,
                node_type,
            ) in selected:

                regions.append(
                    StatementRegion(
                        start_line=start_line,
                        end_line=end_line,
                        node_type=node_type,
                        anchor_lines=[
                            anchor_line
                        ],
                    )
                )

        return self._merge_regions(
            regions
        )

    @staticmethod
    def _merge_regions(
        regions: list[StatementRegion],
    ) -> list[StatementRegion]:

        merged = {}

        for region in regions:

            key = (
                region.start_line,
                region.end_line,
                region.node_type,
            )

            if key not in merged:

                merged[key] = (
                    StatementRegion(
                        start_line=(
                            region.start_line
                        ),
                        end_line=(
                            region.end_line
                        ),
                        node_type=(
                            region.node_type
                        ),
                        anchor_lines=[],
                    )
                )

            merged[key].anchor_lines.extend(
                region.anchor_lines
            )

        output = list(
            merged.values()
        )

        for region in output:

            region.anchor_lines = sorted(
                set(
                    region.anchor_lines
                )
            )

        output.sort(
            key=lambda item: (
                item.start_line,
                item.end_line,
            )
        )

        return output