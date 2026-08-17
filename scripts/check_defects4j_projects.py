from __future__ import annotations

import argparse
import subprocess

from camd.evaluation.project_registry import (
    get_project_config,
    list_supported_projects,
)


def run_defects4j_info(
    project: str,
) -> tuple[int, str, str]:
    process = subprocess.run(
        [
            "defects4j",
            "info",
            "-p",
            project,
        ],
        capture_output=True,
        text=True,
    )

    return (
        process.returncode,
        process.stdout,
        process.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Defects4J availability for "
            "CAMD multi-project experiments."
        )
    )

    parser.add_argument(
        "--projects",
        nargs="+",
        default=list_supported_projects(),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("CAMD Defects4J Multi-Project Check")
    print("=" * 100)

    success = []
    failed = []

    for project in args.projects:
        config = get_project_config(
            project
        )

        print()
        print(
            f"[{project}] "
            f"{config.display_name}"
        )

        code, stdout, stderr = (
            run_defects4j_info(
                project
            )
        )

        if code == 0:
            success.append(
                project
            )

            print("  Status: OK")

            lines = [
                line.strip()
                for line in stdout.splitlines()
                if line.strip()
            ]

            for line in lines[:8]:
                print(
                    f"  {line}"
                )

        else:
            failed.append(
                project
            )

            print("  Status: FAILED")

            message = (
                stderr.strip()
                or stdout.strip()
            )

            if message:
                print(
                    f"  {message}"
                )

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)

    print(
        "Available:",
        success,
    )

    print(
        "Failed:",
        failed,
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()