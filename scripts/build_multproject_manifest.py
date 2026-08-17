from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "multi_project_manifest.json"
)

DEFAULT_PROJECTS = [
    "Lang",
    "Math",
    "Chart",
    "Time",
    "Mockito",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Defects4J active-bug manifest "
            "for CAMD multi-project experiments."
        )
    )

    parser.add_argument(
        "--projects",
        nargs="+",
        default=DEFAULT_PROJECTS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--limit-per-project",
        type=int,
        default=None,
        help=(
            "Optional maximum number of active bugs "
            "to include per project. "
            "Default: include all active bugs."
        ),
    )

    return parser.parse_args()


def run_command(
    command: list[str],
) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr
        )

    return result.stdout.strip()


def get_active_bug_ids(
    project: str,
) -> list[int]:
    output = run_command(
        [
            "defects4j",
            "bids",
            "-p",
            project,
        ]
    )

    bug_ids: list[int] = []

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        # Support either one ID per line or
        # whitespace-separated output.
        for token in line.split():
            token = token.strip()

            if token.isdigit():
                bug_ids.append(
                    int(token)
                )

    bug_ids = sorted(
        set(bug_ids)
    )

    if not bug_ids:
        raise RuntimeError(
            f"No active bug IDs found for {project}."
        )

    return bug_ids


def get_project_metadata(
    project: str,
) -> dict[str, Any]:
    output = run_command(
        [
            "defects4j",
            "info",
            "-p",
            project,
        ]
    )

    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]

    return {
        "project": project,
        "info_preview": lines[:20],
    }


def choose_bug_ids(
    bug_ids: list[int],
    limit: int | None,
) -> list[int]:
    if limit is None:
        return bug_ids

    if limit <= 0:
        raise ValueError(
            "--limit-per-project must be > 0."
        )

    return bug_ids[:limit]


def build_project_entry(
    project: str,
    limit: int | None,
) -> dict[str, Any]:
    active_bug_ids = get_active_bug_ids(
        project
    )

    selected_bug_ids = choose_bug_ids(
        active_bug_ids,
        limit,
    )

    metadata = get_project_metadata(
        project
    )

    return {
        "project": project,

        "active_bug_count": len(
            active_bug_ids
        ),

        "active_bug_ids": (
            active_bug_ids
        ),

        "selected_bug_count": len(
            selected_bug_ids
        ),

        "selected_bug_ids": (
            selected_bug_ids
        ),

        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()

    print()
    print("=" * 100)
    print("CAMD Multi-Project Defects4J Manifest")
    print("=" * 100)

    projects: dict[
        str,
        dict[str, Any]
    ] = {}

    total_active = 0
    total_selected = 0

    for project in args.projects:
        print()
        print(
            f"[{project}] Reading Defects4J metadata..."
        )

        entry = build_project_entry(
            project,
            args.limit_per_project,
        )

        projects[
            project
        ] = entry

        total_active += (
            entry[
                "active_bug_count"
            ]
        )

        total_selected += (
            entry[
                "selected_bug_count"
            ]
        )

        print(
            "  Active bugs: "
            f"{entry['active_bug_count']}"
        )

        print(
            "  Selected: "
            f"{entry['selected_bug_count']}"
        )

        print(
            "  IDs: "
            f"{entry['selected_bug_ids']}"
        )

    manifest = {
        "name": (
            "CAMD Multi-Project "
            "Defects4J Benchmark"
        ),

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source_of_truth": (
            "Defects4J active bug IDs "
            "reported by `defects4j bids`."
        ),

        "projects": (
            projects
        ),

        "summary": {
            "project_count": len(
                projects
            ),

            "total_active_bugs": (
                total_active
            ),

            "total_selected_bugs": (
                total_selected
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)

    print(
        "Projects: "
        f"{len(projects)}"
    )

    print(
        "Total active bugs: "
        f"{total_active}"
    )

    print(
        "Total selected bugs: "
        f"{total_selected}"
    )

    print()
    print("Saved:")
    print(
        args.output
    )


if __name__ == "__main__":
    main()