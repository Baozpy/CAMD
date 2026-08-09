from pathlib import Path

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

    print("=" * 60)
    print("CAMD - Java Method Extraction")
    print("=" * 60)

    print(
        f"Total methods: {len(methods)}"
    )

    print()

    for method in methods:

        print(
            f"{method.name}: "
            f"{method.start_line}-"
            f"{method.end_line}"
        )

    print()

    target = [
        method
        for method in methods
        if method.name == "createNumber"
    ]

    if target:

        method = target[0]

        print("=" * 60)
        print("Target method: createNumber")
        print("=" * 60)

        print(
            f"Lines: "
            f"{method.start_line}-"
            f"{method.end_line}"
        )

        print()

        print(method.code)

    else:

        print(
            "createNumber was not found."
        )


if __name__ == "__main__":
    main()