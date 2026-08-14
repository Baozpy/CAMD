import argparse
import json
from pathlib import Path

from camd.evaluation.experiment_runner import (
    Defects4JExperimentRunner,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run CAMD experiments "
            "on Defects4J bugs."
        )
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--bug-start",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--bug-end",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse an existing per-bug "
            "summary.json instead of "
            "rerunning the experiment."
        ),
    )

    return parser.parse_args()


def load_existing_summary(
    project: str,
    bug_id: int,
):

    summary_file = (
        PROJECT_ROOT
        / "results"
        / "defects4j"
        / f"{project}_{bug_id}"
        / "summary.json"
    )

    if not summary_file.exists():
        return None

    with summary_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def aggregate_summaries(
    summaries: list[dict],
) -> dict:

    if not summaries:
        return {}

    total = len(summaries)

    stages = [
        "b1_method_only",
        "b4_static_aware",
        "camd_multi_agent",
    ]

    aggregate = {
        "total_bugs": total,
        "bugs": [
            item["bug_id"]
            for item in summaries
        ],
        "stages": {},
    }

    for stage in stages:

        valid_items = [
            item
            for item in summaries
            if (
                stage in item
                and item[stage]
            )
        ]

        if not valid_items:

            aggregate["stages"][
                stage
            ] = {
                "mrr": 0.0,
                "top_1_accuracy": 0.0,
                "top_3_accuracy": 0.0,
                "top_5_accuracy": 0.0,
                "evaluated_bugs": 0,
            }

            continue

        stage_total = len(
            valid_items
        )

        rr_values = [
            item[stage]["rr"]
            for item in valid_items
        ]

        top_1 = [
            item[stage]["top_1"]
            for item in valid_items
        ]

        top_3 = [
            item[stage]["top_3"]
            for item in valid_items
        ]

        top_5 = [
            item[stage]["top_5"]
            for item in valid_items
        ]

        aggregate["stages"][
            stage
        ] = {
            "mrr": (
                sum(rr_values)
                / stage_total
            ),

            "top_1_accuracy": (
                sum(top_1)
                / stage_total
            ),

            "top_3_accuracy": (
                sum(top_3)
                / stage_total
            ),

            "top_5_accuracy": (
                sum(top_5)
                / stage_total
            ),

            "evaluated_bugs": (
                stage_total
            ),
        }

    return aggregate


def print_aggregate(
    aggregate: dict,
) -> None:

    print()
    print("=" * 100)
    print(
        "CAMD Aggregate Results"
    )
    print("=" * 100)

    print(
        f"Total bugs: "
        f"{aggregate['total_bugs']}"
    )

    print()

    for stage, metrics in (
        aggregate["stages"].items()
    ):

        print(stage)

        print(
            f"  Evaluated bugs: "
            f"{metrics['evaluated_bugs']}"
        )

        print(
            f"  MRR: "
            f"{metrics['mrr']:.4f}"
        )

        print(
            f"  Top-1: "
            f"{metrics['top_1_accuracy']:.4f}"
        )

        print(
            f"  Top-3: "
            f"{metrics['top_3_accuracy']:.4f}"
        )

        print(
            f"  Top-5: "
            f"{metrics['top_5_accuracy']:.4f}"
        )

        print()


def main():

    args = parse_args()

    summaries = []

    for bug_id in range(
        args.bug_start,
        args.bug_end + 1,
    ):

        if args.skip_existing:

            existing_summary = (
                load_existing_summary(
                    project=args.project,
                    bug_id=bug_id,
                )
            )

            if (
                existing_summary
                is not None
            ):

                print()
                print("=" * 100)

                print(
                    f"SKIPPING: "
                    f"{args.project}-"
                    f"{bug_id}"
                )

                print(
                    "Existing summary found."
                )

                print("=" * 100)

                summaries.append(
                    existing_summary
                )

                continue

        try:

            runner = (
                Defects4JExperimentRunner(
                    project_root=(
                        PROJECT_ROOT
                    ),
                    project=args.project,
                    bug_id=bug_id,
                    top_k=args.top_k,
                )
            )

            summary = (
                runner.run()
            )

            summaries.append(
                summary
            )

        except Exception as exc:

            print()
            print(
                "=" * 100
            )

            print(
                f"FAILED: "
                f"{args.project}-"
                f"{bug_id}"
            )

            print(
                str(exc)
            )

            print(
                "=" * 100
            )

    aggregate = (
        aggregate_summaries(
            summaries
        )
    )

    if not aggregate:

        print(
            "No experiments "
            "completed successfully."
        )

        return

    output_file = (
        PROJECT_ROOT
        / "results"
        / "defects4j"
        / (
            f"{args.project}_"
            f"{args.bug_start}_"
            f"{args.bug_end}_summary.json"
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "project": (
                    args.project
                ),

                "bug_start": (
                    args.bug_start
                ),

                "bug_end": (
                    args.bug_end
                ),

                "skip_existing": (
                    args.skip_existing
                ),

                "per_bug": (
                    summaries
                ),

                "aggregate": (
                    aggregate
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_aggregate(
        aggregate
    )

    print(
        f"Summary saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()