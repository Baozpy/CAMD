import argparse
import json
from pathlib import Path

from camd.evaluation.adaptive_candidate_runner import (
    AdaptiveCandidateRunner,
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        default="Lang",
    )

    parser.add_argument(
        "--bug-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--initial-top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--expanded-top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    final_summary_file = (
        RESULTS_ROOT
        / "Lang_1_20_final_summary.json"
    )

    if not final_summary_file.exists():

        raise RuntimeError(
            "Missing final method-level "
            f"summary: {final_summary_file}"
        )

    final_summary = (
        load_json(
            final_summary_file
        )
    )

    records_by_bug = {
        item["bug_id"]: item
        for item
        in final_summary[
            "per_bug"
        ]
    }

    if args.bug_id not in records_by_bug:

        raise RuntimeError(
            f"{args.project}-{args.bug_id} "
            "is not in the valid bug set."
        )

    runner = (
        AdaptiveCandidateRunner(
            project_root=PROJECT_ROOT,
            threshold=args.threshold,
            initial_top_k=(
                args.initial_top_k
            ),
            expanded_top_k=(
                args.expanded_top_k
            ),
        )
    )

    print()
    print("=" * 100)
    print(
        "CAMD Adaptive Candidate Expansion"
    )
    print("=" * 100)

    print(
        f"Bug: "
        f"{args.project}-{args.bug_id}"
    )

    print(
        f"Threshold: "
        f"{args.threshold}"
    )

    print(
        f"Top-K: "
        f"{args.initial_top_k} "
        f"→ {args.expanded_top_k}"
    )

    result = (
        runner.run_bug(
            records_by_bug[
                args.bug_id
            ]
        )
    )

    print()
    print(
        f"Triggered: "
        f"{result.get('triggered')}"
    )

    print(
        f"Initial best: "
        f"{result.get('initial_best_method')} "
        f"("
        f"{result.get('initial_best_score')}"
        f")"
    )

    if result.get(
        "expanded"
    ):

        print(
            f"New candidates evaluated: "
            f"{result.get('new_candidates_evaluated')}"
        )

        print(
            f"Final best: "
            f"{result.get('final_best_method')} "
            f"("
            f"{result.get('final_best_score')}"
            f")"
        )

        print()
        print(
            "Final ranking:"
        )

        for item in result.get(
            "final_ranking",
            [],
        ):

            print(
                f"#{item['rank']:<2} "
                f"{item['method']:<35} "
                f"judge="
                f"{item['judge_score']:.4f} "
                f"B4="
                f"{item['b4_rank']}"
            )

    else:

        print(
            f"Expanded: "
            f"{result.get('expanded', False)}"
        )

        if result.get(
            "reason"
        ):
            print(
                f"Reason: "
                f"{result['reason']}"
            )


if __name__ == "__main__":
    main()