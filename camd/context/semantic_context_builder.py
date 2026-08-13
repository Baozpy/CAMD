from dataclasses import dataclass

from camd.context.context_builder import (
    ContextBuilder,
)
from camd.context.context_selector import (
    ContextSelector,
    SelectedContextMethod,
)
from camd.context.method_extractor import (
    JavaMethod,
)


@dataclass
class SemanticMethodContext:
    target: JavaMethod
    selected_methods: list[
        SelectedContextMethod
    ]


class SemanticContextBuilder:

    def __init__(
        self,
        methods: list[JavaMethod],
        top_k_callees: int = 3,
        top_k_callers: int = 2,
    ):

        self.context_builder = (
            ContextBuilder(
                methods=methods
            )
        )

        self.selector = ContextSelector(
            top_k_callees=top_k_callees,
            top_k_callers=top_k_callers,
        )

    def build(
        self,
        target: JavaMethod,
    ) -> SemanticMethodContext:

        structural_context = (
            self.context_builder.build(
                target
            )
        )

        selected_methods = (
            self.selector.select(
                target=target,
                callees=(
                    structural_context.callees
                ),
                callers=(
                    structural_context.callers
                ),
            )
        )

        return SemanticMethodContext(
            target=target,
            selected_methods=(
                selected_methods
            ),
        )