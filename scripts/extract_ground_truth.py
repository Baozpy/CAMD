from pathlib import Path

from camd.evaluation.diff_ground_truth import (
    extract_changed_ranges,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RELATIVE_SOURCE = Path(
    "src/main/java/"
    "org/apache/commons/lang3/math/"
    "NumberUtils.java"
)

BUGGY_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1b"
    / RELATIVE_SOURCE
)

FIXED_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1f"
    / RELATIVE_SOURCE
)


def main():

    print("=" * 60)
    print("CAMD - Defects4J Ground Truth")
    print("=" * 60)

    print("Project: Lang")
    print("Bug ID: 1")

    print(f"\nBuggy file:\n{BUGGY_FILE}")
    print(f"\nFixed file:\n{FIXED_FILE}")

    ranges = extract_changed_ranges(
        buggy_file=BUGGY_FILE,
        fixed_file=FIXED_FILE,
    )

    print("\nChanged buggy-side ranges:")

    for changed_range in ranges:
        print(
            f"  {changed_range.start_line}"
            f"-{changed_range.end_line}"
            f" [{changed_range.change_type}]"
        )

    print("\nTotal changed ranges:", len(ranges))

    print("=" * 60)


if __name__ == "__main__":
    main()