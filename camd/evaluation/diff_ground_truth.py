from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class ChangedRange:
    start_line: int
    end_line: int
    change_type: str


def extract_changed_ranges(
    buggy_file: Path,
    fixed_file: Path,
) -> list[ChangedRange]:

    if not buggy_file.exists():
        raise FileNotFoundError(
            f"Buggy file not found: {buggy_file}"
        )

    if not fixed_file.exists():
        raise FileNotFoundError(
            f"Fixed file not found: {fixed_file}"
        )

    buggy_lines = buggy_file.read_text(
        encoding="utf-8"
    ).splitlines()

    fixed_lines = fixed_file.read_text(
        encoding="utf-8"
    ).splitlines()

    if buggy_lines == fixed_lines:
        raise ValueError(
            "Buggy and fixed files are identical. "
            "Check the checkout paths."
        )

    matcher = SequenceMatcher(
        a=buggy_lines,
        b=fixed_lines,
        autojunk=False,
    )

    changed_ranges: list[ChangedRange] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            continue

        if tag == "insert":
            # New lines exist only in the fixed version.
            # Anchor them to the nearest buggy-side line.
            anchor = i1 + 1

            if anchor > len(buggy_lines):
                anchor = len(buggy_lines)

            changed_ranges.append(
                ChangedRange(
                    start_line=anchor,
                    end_line=anchor,
                    change_type="insert",
                )
            )

        elif tag in {"replace", "delete"}:
            changed_ranges.append(
                ChangedRange(
                    start_line=i1 + 1,
                    end_line=i2,
                    change_type=tag,
                )
            )

    return changed_ranges