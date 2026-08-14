from pathlib import Path

from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
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
    / "Lang_1b"
)


def main():

    extractor = (
        FailingTestExtractor(
            checkout_dir=CHECKOUT_DIR
        )
    )

    failing_tests = (
        extractor.extract()
    )

    print("=" * 80)
    print(
        "CAMD - Failing Test Extraction"
    )
    print("=" * 80)

    print(
        f"Failing tests: "
        f"{len(failing_tests)}"
    )

    print()

    for test in failing_tests:

        print(
            f"Test: "
            f"{test.full_name}"
        )

        print(
            f"Source file: "
            f"{test.source_file}"
        )

        print(
            f"Lines: "
            f"{test.start_line}-"
            f"{test.end_line}"
        )

        print()

        if test.code:

            print(
                test.code
            )

        else:

            print(
                "Test source code "
                "was not extracted."
            )

        print()
        print("-" * 80)


if __name__ == "__main__":
    main()