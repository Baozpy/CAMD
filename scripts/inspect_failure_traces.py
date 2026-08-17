from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from camd.retrieval.failure_trace_parser import (
    FailureTraceParser,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        required=True,
    )

    parser.add_argument(
        "--bug-id",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    checkout = (
        PROJECT_ROOT
        / "data"
        / "defects4j"
        / "checkouts"
        / f"{args.project}_{args.bug_id}b"
    )

    if not checkout.exists():

        print(
            f"Checking out "
            f"{args.project}-{args.bug_id}b..."
        )

        subprocess.run(
            [
                "defects4j",
                "checkout",
                "-p",
                args.project,
                "-v",
                f"{args.bug_id}b",
                "-w",
                str(checkout),
            ],
            check=True,
        )

    print(
        "Running defects4j test..."
    )

    subprocess.run(
        [
            "defects4j",
            "test",
        ],
        cwd=checkout,
        check=False,
    )

    failing_file = (
        checkout
        / "failing_tests"
    )

    print()
    print(
        f"Failure file: "
        f"{failing_file}"
    )

    print(
        f"Exists: "
        f"{failing_file.exists()}"
    )

    parser = (
        FailureTraceParser()
    )

    traces = (
        parser.parse_file(
            failing_file
        )
    )

    print()
    print("=" * 100)
    print("Failure Trace Inspection")
    print("=" * 100)

    print(
        f"Parsed failures: "
        f"{len(traces)}"
    )

    for trace_index, trace in enumerate(
        traces,
        start=1,
    ):

        print()
        print(
            f"[Failure {trace_index}]"
        )

        print(
            f"Test: "
            f"{trace.test_class}"
            f"::{trace.test_method}"
        )

        print(
            f"Exception: "
            f"{trace.exception_line}"
        )

        print(
            f"Stack frames: "
            f"{len(trace.frames)}"
        )

        print()

        for frame in (
            trace.frames[
                :args.max_frames
            ]
        ):

            print(
                f"  depth={frame.depth:<3} "
                f"{frame.class_name}"
                f".{frame.method_name} "
                f"{frame.file_name}:"
                f"{frame.line_number}"
            )


if __name__ == "__main__":
    main()