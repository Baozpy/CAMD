import argparse
import json
from pathlib import Path

from camd.evaluation.line_experiment_runner import (
    LineExperimentRunner,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "defects4j"
)


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    path: Path,
    data: dict,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def aggregate_one_metric(
    results: list[dict],
    metric_name: str,
) -> dict:

    total = len(results)

    if total == 0:
        return {
            "evaluated_bugs": 0,
            "mrr": 0.0,
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "top_10_accuracy": 0.0,
        }

    metrics = [
        item["metrics"][metric_name]
        for item in results
    ]

    return {
        "evaluated_bugs": total,

        "mrr": (
            sum(
                item["reciprocal_rank"]
                for item in metrics
            )
            / total
        ),

        "top_1_accuracy": (
            sum(
                item["top_1"]
                for item in metrics
            )
            / total
        ),

        "top_3_accuracy": (
            sum(
                item["top_3"]
                for item in metrics
            )
            / total
        ),

        "top_5_accuracy": (
            sum(
                item["top_5"]
                for item in metrics
            )
            / total
        ),

        "top_10_accuracy": (
            sum(
                item["top_10"]
                for item in metrics
            )
            / total
        ),
    }


def aggregate_metrics(
    results: list[dict],
) -> dict:

    return {
        "exact": (
            aggregate_one_metric(
                results,
                "exact",
            )
        ),

        "statement": (
            aggregate_one_metric(
                results,
                "statement",
            )
        ),

        "region_2": (
            aggregate_one_metric(
                results,
                "region_2",
            )
        ),
    }

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        default="Lang",
    )

    parser.add_argument(
        "--bug-start",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--bug-end",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )

    args = parser.parse_args()

    final_summary_file = (
        RESULTS_ROOT
        / "Lang_1_20_final_summary.json"
    )

    if not final_summary_file.exists():
        raise RuntimeError(
            "Missing method-level final summary:\n"
            f"{final_summary_file}"
        )

    final_summary = load_json(
        final_summary_file
    )

    records_by_bug = {
        item["bug_id"]: item
        for item
        in final_summary["per_bug"]
    }

    runner = LineExperimentRunner(
        project_root=PROJECT_ROOT,
        top_k=args.top_k,
    )

    completed = []
    failed = []

    for bug_id in range(
        args.bug_start,
        args.bug_end + 1,
    ):

        if bug_id not in records_by_bug:
            print(
                f"\nSkipping {args.project}-{bug_id}: "
                f"not in valid bug set."
            )
            continue

        bug_record = (
            records_by_bug[bug_id]
        )

        output_file = (
            RESULTS_ROOT
            / f"{args.project}_{bug_id}"
            / "line_localization_oracle.json"
        )

        if (
            args.skip_existing
            and output_file.exists()
        ):
            print(
                f"\nSkipping existing "
                f"{args.project}-{bug_id}"
            )

            result = load_json(
                output_file
            )

            completed.append(
                result
            )

            continue

        print()
        print("=" * 100)
        print(
            f"Line Localization: "
            f"{args.project}-{bug_id}"
        )
        print("=" * 100)

        try:

            result = (
                runner.run_oracle_bug(
                    bug_record
                )
            )

            completed.append(
                result
            )

            metrics = result["metrics"]

            print()
            print("Exact Patch-Line")

            print(
                f"  First hit rank: "
                f"{metrics['exact']['first_hit_rank']}"
            )

            print(
                f"  RR: "
                f"{metrics['exact']['reciprocal_rank']:.4f}"
            )

            print(
                f"  Top-1: "
                f"{metrics['exact']['top_1']}"
            )

            print()
            print("AST Statement")

            print(
                f"  First hit rank: "
                f"{metrics['statement']['first_hit_rank']}"
            )

            print(
                f"  RR: "
                f"{metrics['statement']['reciprocal_rank']:.4f}"
            )

            print(
                f"  Top-1: "
                f"{metrics['statement']['top_1']}"
            )

            print(
                f"  Top-3: "
                f"{metrics['statement']['top_3']}"
            )

            print(
                f"  Top-5: "
                f"{metrics['statement']['top_5']}"
            )

            print()
            print("±2 Line Region")

            print(
                f"  First hit rank: "
                f"{metrics['region_2']['first_hit_rank']}"
            )

            print(
                f"  RR: "
                f"{metrics['region_2']['reciprocal_rank']:.4f}"
            )

            # metrics = (
            #     result["metrics"]
            # )///////////////////////////////////////

            # print(
            #     f"First hit rank: "
            #     f"{metrics['first_hit_rank']}"
            # )

            # print(
            #     f"RR: "
            #     f"{metrics['reciprocal_rank']:.4f}"
            # )

            # print(
            #     f"Top-1: "
            #     f"{metrics['top_1']}"
            # )

            # print(
            #     f"Top-3: "
            #     f"{metrics['top_3']}"
            # )

            # print(
            #     f"Top-5: "
            #     f"{metrics['top_5']}"
            # )

        except Exception as exc:

            failed.append(
                {
                    "bug_id": bug_id,
                    "error": str(exc),
                }
            )

            print()
            print(
                f"FAILED: "
                f"{args.project}-{bug_id}"
            )

            print(exc)

    aggregate = (
        aggregate_metrics(
            completed
        )
    )

    summary = {
        "project": args.project,

        "bug_start": (
            args.bug_start
        ),

        "bug_end": (
            args.bug_end
        ),

        "mode": (
            "oracle_method_line_localization"
        ),

        "top_k": args.top_k,

        "evaluated_bug_ids": [
            item["bug_id"]
            for item in completed
        ],

        "failed": failed,

        "per_bug": completed,

        "aggregate": aggregate,
    }

    output_summary = (
        RESULTS_ROOT
        / (
            f"{args.project}_"
            f"{args.bug_start}_"
            f"{args.bug_end}_"
            f"line_oracle_summary.json"
        )
    )

    save_json(
        output_summary,
        summary,
    )

    print()
    print("=" * 100)
    print(
        "Oracle-Method Line Localization "
        "Aggregate"
    )
    print("=" * 100)

    print(
        f"Evaluated bugs: "
        f"{aggregate['statement']['evaluated_bugs']}"
    )
    print()
    print("Exact Patch-Line")

    print(
        f"  MRR: "
        f"{aggregate['exact']['mrr']:.4f}"
    )

    print(
        f"  Top-1: "
        f"{aggregate['exact']['top_1_accuracy']:.4f}"
    )

    print(
        f"  Top-3: "
        f"{aggregate['exact']['top_3_accuracy']:.4f}"
    )

    print(
        f"  Top-5: "
        f"{aggregate['exact']['top_5_accuracy']:.4f}"
    )

    print(
        f"  Top-10: "
        f"{aggregate['exact']['top_10_accuracy']:.4f}"
    )

    print()
    print("AST Statement")

    print(
        f"  MRR: "
        f"{aggregate['statement']['mrr']:.4f}"
    )

    print(
        f"  Top-1: "
        f"{aggregate['statement']['top_1_accuracy']:.4f}"
    )

    print(
        f"  Top-3: "
        f"{aggregate['statement']['top_3_accuracy']:.4f}"
    )

    print(
        f"  Top-5: "
        f"{aggregate['statement']['top_5_accuracy']:.4f}"
    )

    print(
        f"  Top-10: "
        f"{aggregate['statement']['top_10_accuracy']:.4f}"
    )

    print()
    print("±2 Line Region")

    print(
        f"  MRR: "
        f"{aggregate['region_2']['mrr']:.4f}"
    )

    print(
        f"  Top-1: "
        f"{aggregate['region_2']['top_1_accuracy']:.4f}"
    )

    print(
        f"  Top-3: "
        f"{aggregate['region_2']['top_3_accuracy']:.4f}"
    )

    print(
        f"  Top-5: "
        f"{aggregate['region_2']['top_5_accuracy']:.4f}"
    )

    print(
        f"  Top-10: "
        f"{aggregate['region_2']['top_10_accuracy']:.4f}"
    )


    print(
        f"Failed bugs: "
        f"{[item['bug_id'] for item in failed]}"
    )

    print()
    print(
        "Summary saved to:"
    )
    print(
        output_summary
    )


if __name__ == "__main__":
    main()