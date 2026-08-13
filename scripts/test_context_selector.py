from pathlib import Path

from camd.context.method_extractor import (
    extract_java_methods,
)
from camd.context.semantic_context_builder import (
    SemanticContextBuilder,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1b"
    / "src"
    / "main"
    / "java"
    / "org"
    / "apache"
    / "commons"
    / "lang3"
    / "math"
    / "NumberUtils.java"
)


def main():

    methods = extract_java_methods(
        SOURCE_FILE
    )

    target = next(
        method
        for method in methods
        if method.name
        == "createNumber"
    )

    builder = (
        SemanticContextBuilder(
            methods=methods,
            top_k_callees=3,
            top_k_callers=2,
        )
    )

    context = builder.build(
        target
    )

    print("=" * 70)
    print("CAMD - Context Selector Test")
    print("=" * 70)

    print(
        f"Target: "
        f"{target.name}"
    )

    print()

    for item in (
        context.selected_methods
    ):

        print(
            f"{item.relation}: "
            f"{item.method.name}"
        )

        print(
            f"Score: "
            f"{item.relevance_score:.2f}"
        )

        print(
            f"Reason: "
            f"{item.reason}"
        )

        print()


if __name__ == "__main__":
    main()