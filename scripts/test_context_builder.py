from pathlib import Path

from camd.context.context_builder import (
    ContextBuilder,
)
from camd.context.method_extractor import (
    extract_java_methods,
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

    target_methods = [
        method
        for method in methods
        if method.name
        == "createNumber"
    ]

    if not target_methods:

        raise ValueError(
            "createNumber not found."
        )

    target = target_methods[0]

    builder = ContextBuilder(
        methods
    )

    context = builder.build(
        target
    )

    print("=" * 70)
    print("CAMD - Context Builder Test")
    print("=" * 70)

    print(
        f"Target: "
        f"{context.target.name}"
    )

    print(
        f"Lines: "
        f"{context.target.start_line}-"
        f"{context.target.end_line}"
    )

    print()

    print("Direct callees:")

    if not context.callees:
        print("  None")

    for method in context.callees:

        print(
            f"  {method.name} "
            f"({method.start_line}-"
            f"{method.end_line})"
        )

    print()

    print("Direct callers:")

    if not context.callers:
        print("  None")

    for method in context.callers:

        print(
            f"  {method.name} "
            f"({method.start_line}-"
            f"{method.end_line})"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()