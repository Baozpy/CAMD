from pathlib import Path

from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
)
from camd.evaluation.test_context_builder import (
    TestContextBuilder,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CHECKOUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_10b"
)


def main():

    failing_extractor = (
        FailingTestExtractor(
            checkout_dir=CHECKOUT_DIR
        )
    )

    failing_tests = (
        failing_extractor.extract()
    )

    print("=" * 100)
    print(
        "CAMD - Expanded Failing "
        "Test Context"
    )
    print("=" * 100)

    builder = (
        TestContextBuilder()
    )

    for test in failing_tests:

        print()
        print(
            f"Test: "
            f"{test.full_name}"
        )

        if (
            test.source_file
            is None
        ):

            print(
                "No source file found."
            )

            continue

        try:

            expanded = (
                builder.build(
                    source_file=(
                        test.source_file
                    ),
                    test_method_name=(
                        test.method_name
                    ),
                )
            )

        except ValueError as exc:

            print(
                str(exc)
            )

            continue

        print()
        print(
            expanded.to_text()
        )

        print()
        print(
            "=" * 100
        )


if __name__ == "__main__":
    main()