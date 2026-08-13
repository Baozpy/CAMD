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


def print_method_context(
    builder,
    method,
):

    context = builder.build(
        method
    )

    print(
        f"Target: "
        f"{method.name}"
        f"/{method.parameter_count}"
    )

    print(
        f"Lines: "
        f"{method.start_line}-"
        f"{method.end_line}"
    )

    print("Callees:")

    if not context.callees:
        print("  None")

    for callee in context.callees:

        print(
            f"  {callee.name}"
            f"/{callee.parameter_count} "
            f"({callee.start_line}-"
            f"{callee.end_line})"
        )

    print("Callers:")

    if not context.callers:
        print("  None")

    for caller in context.callers:

        print(
            f"  {caller.name}"
            f"/{caller.parameter_count} "
            f"({caller.start_line}-"
            f"{caller.end_line})"
        )

    print("-" * 70)


def main():

    methods = extract_java_methods(
        SOURCE_FILE
    )

    builder = ContextBuilder(
        methods
    )

    print("=" * 70)
    print(
        "CAMD - Overload Resolution Test"
    )
    print("=" * 70)

    target_names = {
        "toInt",
        "min",
        "max",
    }

    for method in methods:

        if (
            method.name
            not in target_names
        ):
            continue

        print_method_context(
            builder=builder,
            method=method,
        )


if __name__ == "__main__":
    main()