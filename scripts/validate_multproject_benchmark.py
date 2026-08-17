from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BENCHMARK = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "fse_ase_benchmark_v1.json"
)

DEFAULT_CHECKOUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "fse_ase_v1"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "defects4j"
    / "fse_ase_benchmark_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen CAMD FSE/ASE multi-project "
            "Defects4J benchmark before running expensive LLM evaluation."
        )
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
    )

    parser.add_argument(
        "--checkout-root",
        type=Path,
        default=DEFAULT_CHECKOUT_ROOT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--keep-checkouts",
        action="store_true",
        help=(
            "Keep checked-out projects after validation. "
            "Default behavior removes each checkout after validation."
        ),
    )

    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help=(
            "Skip `defects4j compile`. "
            "Useful only for debugging the validation script."
        ),
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based benchmark entry index to start from.",
    )

    parser.add_argument(
        "--max-bugs",
        type=int,
        default=None,
        help="Optional maximum number of bugs to validate.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry benchmark entries that were previously "
            "recorded as failed. Successful entries remain skipped."
        ),
    )

    return parser.parse_args()


def load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    started = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_sec": round(
                time.time() - started,
                3,
            ),
            "timed_out": False,
        }

    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": (
                exc.stdout.decode()
                if isinstance(exc.stdout, bytes)
                else exc.stdout
            ) or "",
            "stderr": (
                exc.stderr.decode()
                if isinstance(exc.stderr, bytes)
                else exc.stderr
            ) or "",
            "duration_sec": round(
                time.time() - started,
                3,
            ),
            "timed_out": True,
        }


def cleanup_checkout(
    path: Path,
) -> None:
    if path.exists():
        shutil.rmtree(
            path,
            ignore_errors=True,
        )


def checkout_bug(
    project: str,
    bug_id: int,
    checkout_dir: Path,
) -> dict[str, Any]:
    cleanup_checkout(
        checkout_dir
    )

    checkout_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_command(
        [
            "defects4j",
            "checkout",
            "-p",
            project,
            "-v",
            f"{bug_id}b",
            "-w",
            str(checkout_dir),
        ],
        timeout=600,
    )


def compile_bug(
    checkout_dir: Path,
) -> dict[str, Any]:
    return run_command(
        [
            "defects4j",
            "compile",
        ],
        cwd=checkout_dir,
        timeout=900,
    )


def export_property(
    checkout_dir: Path,
    prop: str,
) -> dict[str, Any]:
    result = run_command(
        [
            "defects4j",
            "export",
            "-p",
            prop,
        ],
        cwd=checkout_dir,
        timeout=120,
    )

    value = result[
        "stdout"
    ].strip()

    result[
        "value"
    ] = value

    return result


def parse_lines(
    text: str,
) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def validate_entry(
    entry: dict[str, Any],
    *,
    checkout_root: Path,
    skip_compile: bool,
    keep_checkout: bool,
) -> dict[str, Any]:
    project = entry[
        "project"
    ]

    bug_id = int(
        entry[
            "bug_id"
        ]
    )

    benchmark_id = entry[
        "benchmark_id"
    ]

    checkout_dir = (
        checkout_root
        / f"{project}_{bug_id}b"
    )

    result: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "project": project,
        "bug_id": bug_id,
        "checkout_dir": str(
            checkout_dir
        ),
        "checkout": {
            "status": "pending",
        },
        "compile": {
            "status": "pending",
        },
        "failing_tests": {
            "status": "pending",
            "items": [],
        },
        "classes_modified": {
            "status": "pending",
            "items": [],
        },
        "success": False,
    }

    checkout = checkout_bug(
        project,
        bug_id,
        checkout_dir,
    )

    result[
        "checkout"
    ] = checkout

    result[
        "checkout"
    ][
        "status"
    ] = (
        "success"
        if checkout[
            "returncode"
        ] == 0
        else "failed"
    )

    if checkout[
        "returncode"
    ] != 0:
        if not keep_checkout:
            cleanup_checkout(
                checkout_dir
            )

        return result

    if skip_compile:
        result[
            "compile"
        ] = {
            "status": "skipped",
        }

    else:
        compile_result = compile_bug(
            checkout_dir
        )

        compile_result[
            "status"
        ] = (
            "success"
            if compile_result[
                "returncode"
            ] == 0
            else "failed"
        )

        result[
            "compile"
        ] = compile_result

    failing_tests = export_property(
        checkout_dir,
        "tests.trigger",
    )

    failing_tests[
        "items"
    ] = parse_lines(
        failing_tests[
            "value"
        ]
    )

    failing_tests[
        "status"
    ] = (
        "success"
        if (
            failing_tests[
                "returncode"
            ] == 0
            and len(
                failing_tests[
                    "items"
                ]
            ) > 0
        )
        else "failed"
    )

    result[
        "failing_tests"
    ] = failing_tests

    modified = export_property(
        checkout_dir,
        "classes.modified",
    )

    modified[
        "items"
    ] = parse_lines(
        modified[
            "value"
        ]
    )

    modified[
        "status"
    ] = (
        "success"
        if (
            modified[
                "returncode"
            ] == 0
            and len(
                modified[
                    "items"
                ]
            ) > 0
        )
        else "failed"
    )

    result[
        "classes_modified"
    ] = modified

    compile_ok = (
        skip_compile
        or result[
            "compile"
        ][
            "status"
        ] == "success"
    )

    result[
        "success"
    ] = (
        result[
            "checkout"
        ][
            "status"
        ] == "success"
        and compile_ok
        and result[
            "failing_tests"
        ][
            "status"
        ] == "success"
        and result[
            "classes_modified"
        ][
            "status"
        ] == "success"
    )

    if not keep_checkout:
        cleanup_checkout(
            checkout_dir
        )

    return result


def summarize(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(
        results
    )

    successful = sum(
        1
        for row in results
        if row[
            "success"
        ]
    )

    by_project: dict[
        str,
        dict[str, Any]
    ] = {}

    for row in results:
        project = row[
            "project"
        ]

        if project not in by_project:
            by_project[
                project
            ] = {
                "total": 0,
                "successful": 0,
            }

        by_project[
            project
        ][
            "total"
        ] += 1

        if row[
            "success"
        ]:
            by_project[
                project
            ][
                "successful"
            ] += 1

    for project, stats in by_project.items():
        total_p = stats[
            "total"
        ]

        success_p = stats[
            "successful"
        ]

        stats[
            "processable_rate"
        ] = (
            success_p
            / total_p
            if total_p
            else 0.0
        )

    return {
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "processable_rate": (
            successful
            / total
            if total
            else 0.0
        ),
        "by_project": by_project,
    }


def save_progress(
    *,
    benchmark: dict[str, Any],
    results: list[dict[str, Any]],
    output: Path,
) -> None:
    payload = {
        "benchmark": benchmark[
            "name"
        ],
        "selection_policy": benchmark[
            "selection_policy"
        ],
        "summary": summarize(
            results
        ),
        "results": results,
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    args = parse_args()

    benchmark = load_json(
        args.benchmark
    )

    all_entries = benchmark[
        "entries"
    ]

    if args.start_index < 1:
        raise ValueError(
            "--start-index must be >= 1."
        )

    selected_entries = all_entries[
        args.start_index - 1:
    ]

    if args.max_bugs is not None:
        if args.max_bugs <= 0:
            raise ValueError(
                "--max-bugs must be > 0."
            )

        selected_entries = selected_entries[
            :args.max_bugs
        ]

    existing_results: list[
        dict[str, Any]
    ] = []

    if args.output.exists():
        try:
            existing_payload = load_json(
                args.output
            )

            existing_results = (
                existing_payload.get(
                    "results",
                    []
                )
            )

            if not isinstance(
                existing_results,
                list,
            ):
                raise ValueError(
                    "Existing validation file contains "
                    "an invalid 'results' field."
                )

        except Exception as exc:
            raise RuntimeError(
                "Existing validation output could not "
                f"be loaded safely:\n{args.output}"
            ) from exc

    results_by_id: dict[
        str,
        dict[str, Any]
    ] = {
        row[
            "benchmark_id"
        ]: row
        for row in existing_results
        if (
            isinstance(row, dict)
            and "benchmark_id" in row
        )
    }

    completed_success_ids = {
        row[
            "benchmark_id"
        ]
        for row in existing_results
        if (
            isinstance(row, dict)
            and "benchmark_id" in row
            and row.get(
                "success",
                False,
            )
        )
    }

    completed_failed_ids = {
        row[
            "benchmark_id"
        ]
        for row in existing_results
        if (
            isinstance(row, dict)
            and "benchmark_id" in row
            and not row.get(
                "success",
                False,
            )
        )
    }

    if args.retry_failed:
        entries_to_run = [
            entry
            for entry in selected_entries
            if entry[
                "benchmark_id"
            ] not in completed_success_ids
        ]
    else:
        completed_ids = (
            completed_success_ids
            | completed_failed_ids
        )

        entries_to_run = [
            entry
            for entry in selected_entries
            if entry[
                "benchmark_id"
            ] not in completed_ids
        ]

    print()
    print("=" * 100)
    print(
        "CAMD Multi-Project Benchmark Validation"
    )
    print("=" * 100)

    print(
        "Frozen benchmark entries: "
        f"{len(all_entries)}"
    )

    print(
        "Entries in current selection: "
        f"{len(selected_entries)}"
    )

    print(
        "Previously recorded results: "
        f"{len(existing_results)}"
    )

    print(
        "Previously successful: "
        f"{len(completed_success_ids)}"
    )

    print(
        "Previously failed: "
        f"{len(completed_failed_ids)}"
    )

    print(
        "Retry failed enabled: "
        f"{args.retry_failed}"
    )

    print(
        "Entries remaining to validate: "
        f"{len(entries_to_run)}"
    )

    print(
        f"Keep checkouts: {args.keep_checkouts}"
    )

    print(
        f"Compile enabled: {not args.skip_compile}"
    )

    benchmark_order = {
        entry[
            "benchmark_id"
        ]: index
        for index, entry in enumerate(
            all_entries
        )
    }

    def ordered_results() -> list[
        dict[str, Any]
    ]:
        return sorted(
            results_by_id.values(),
            key=lambda row: benchmark_order.get(
                row[
                    "benchmark_id"
                ],
                10**9,
            ),
        )

    if not entries_to_run:
        results = ordered_results()

        summary = summarize(
            results
        )

        save_progress(
            benchmark=benchmark,
            results=results,
            output=args.output,
        )

        print()
        print(
            "No unfinished entries in the "
            "current selection."
        )

        print()
        print("=" * 100)
        print("Validation Summary")
        print("=" * 100)

        print(
            f"Total recorded: "
            f"{summary['total']}"
        )

        print(
            f"Successful: "
            f"{summary['successful']}"
        )

        print(
            f"Failed: "
            f"{summary['failed']}"
        )

        print(
            "Processable rate: "
            f"{summary['processable_rate']:.4f}"
        )

        print()

        for project, stats in (
            summary[
                "by_project"
            ].items()
        ):
            print(
                f"{project}: "
                f"{stats['successful']}/"
                f"{stats['total']} "
                f"({stats['processable_rate']:.4f})"
            )

        print()
        print("Saved:")
        print(
            args.output
        )

        return

    for run_index, entry in enumerate(
        entries_to_run,
        start=1,
    ):
        benchmark_id = entry[
            "benchmark_id"
        ]

        global_index = (
            benchmark_order[
                benchmark_id
            ]
            + 1
        )

        previous_row = results_by_id.get(
            benchmark_id
        )

        is_retry = (
            previous_row is not None
            and not previous_row.get(
                "success",
                False,
            )
        )

        print()
        print(
            f"[{run_index}/"
            f"{len(entries_to_run)}] "
            f"{benchmark_id} "
            f"(benchmark "
            f"{global_index}/"
            f"{len(all_entries)})"
        )

        if is_retry:
            print(
                "  Mode: RETRY PREVIOUS FAILURE"
            )

        row = validate_entry(
            entry,
            checkout_root=args.checkout_root,
            skip_compile=args.skip_compile,
            keep_checkout=args.keep_checkouts,
        )

        if is_retry:
            row[
                "retry"
            ] = {
                "was_retried": True,
                "previous_success": (
                    previous_row.get(
                        "success",
                        False,
                    )
                ),
                "previous_checkout_status": (
                    previous_row.get(
                        "checkout",
                        {}
                    ).get(
                        "status"
                    )
                ),
                "reason": (
                    "Infrastructure issue corrected; "
                    "failed benchmark entry revalidated."
                ),
            }

        results_by_id[
            benchmark_id
        ] = row

        print(
            "  Checkout: "
            f"{row['checkout']['status']}"
        )

        print(
            "  Compile: "
            f"{row['compile']['status']}"
        )

        print(
            "  Failing tests: "
            f"{row['failing_tests']['status']}"
        )

        print(
            "  Modified classes: "
            f"{row['classes_modified']['status']}"
        )

        print(
            "  Overall: "
            f"{'SUCCESS' if row['success'] else 'FAILED'}"
        )

        current_results = ordered_results()

        save_progress(
            benchmark=benchmark,
            results=current_results,
            output=args.output,
        )

    results = ordered_results()

    summary = summarize(
        results
    )

    print()
    print("=" * 100)
    print("Validation Summary")
    print("=" * 100)

    print(
        f"Total recorded: "
        f"{summary['total']}"
    )

    print(
        f"Successful: "
        f"{summary['successful']}"
    )

    print(
        f"Failed: "
        f"{summary['failed']}"
    )

    print(
        "Processable rate: "
        f"{summary['processable_rate']:.4f}"
    )

    print()

    for project, stats in (
        summary[
            "by_project"
        ].items()
    ):
        print(
            f"{project}: "
            f"{stats['successful']}/"
            f"{stats['total']} "
            f"({stats['processable_rate']:.4f})"
        )

    print()
    print("Saved:")
    print(
        args.output
    )



if __name__ == "__main__":
    main()