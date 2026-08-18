from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from camd.verification.frozen_candidate_loader import (
    FrozenCandidateLoader,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "defects4j"
    / "fse_ase_frozen_candidate_pools.json"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "candidate_recovery_analysis.json"
)


BUDGETS = [
    10,
    20,
    50,
    100,
]


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def gt_key(
    gt,
):
    return (
        gt.class_name,
        gt.source_file,
        gt.start_line,
        gt.end_line,
    )


def candidate_key(
    candidate,
):
    return (
        candidate.class_name,
        candidate.source_file,
        candidate.start_line,
        candidate.end_line,
    )


def case_has_gt(
    case,
) -> bool:

    gt_keys = {
        gt_key(gt)
        for gt in case.ground_truth
    }

    candidate_keys = {
        candidate_key(candidate)
        for candidate in case.candidates
    }

    return bool(
        gt_keys
        & candidate_keys
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    loader = FrozenCandidateLoader(
        args.manifest
    )

    benchmark_ids = (
        loader.benchmark_ids(
            only_successful=True,
            only_method_applicable=True,
        )
    )

    records = []

    coverage_counts = {
        budget: 0
        for budget in BUDGETS
    }

    first_recovery_counts = defaultdict(
        int
    )

    project_stats = defaultdict(
        lambda: {
            "total": 0,
            "recall_at_10": 0,
            "recall_at_20": 0,
            "recall_at_50": 0,
            "recall_at_100": 0,
            "recovered_20": 0,
            "recovered_50": 0,
            "recovered_100": 0,
            "never_recovered": 0,
        }
    )

    for benchmark_id in benchmark_ids:

        recall_by_budget = {}

        cases = {}

        for budget in BUDGETS:

            case = loader.load_case(
                benchmark_id,
                budget=budget,
            )

            cases[
                budget
            ] = case

            recalled = (
                case_has_gt(
                    case
                )
            )

            recall_by_budget[
                budget
            ] = recalled

            if recalled:
                coverage_counts[
                    budget
                ] += 1

        base_case = (
            cases[10]
        )

        project = (
            base_case.project
        )

        project_stats[
            project
        ][
            "total"
        ] += 1

        for budget in BUDGETS:

            if recall_by_budget[
                budget
            ]:

                project_stats[
                    project
                ][
                    f"recall_at_{budget}"
                ] += 1

        first_recovery = None

        if recall_by_budget[10]:

            first_recovery = 10

        else:

            for budget in [
                20,
                50,
                100,
            ]:

                if recall_by_budget[
                    budget
                ]:

                    first_recovery = (
                        budget
                    )

                    project_stats[
                        project
                    ][
                        f"recovered_{budget}"
                    ] += 1

                    break

            if first_recovery is None:

                project_stats[
                    project
                ][
                    "never_recovered"
                ] += 1

        if first_recovery is None:

            first_recovery_counts[
                "never"
            ] += 1

        else:

            first_recovery_counts[
                str(
                    first_recovery
                )
            ] += 1

        records.append(
            {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": (
                    project
                ),
                "recall_at_10": (
                    recall_by_budget[10]
                ),
                "recall_at_20": (
                    recall_by_budget[20]
                ),
                "recall_at_50": (
                    recall_by_budget[50]
                ),
                "recall_at_100": (
                    recall_by_budget[100]
                ),
                "first_recovery_budget": (
                    first_recovery
                ),
            }
        )

    total = len(
        records
    )

    base_misses = [
        record
        for record in records
        if not record[
            "recall_at_10"
        ]
    ]

    print("=" * 100)
    print(
        "CAMD Candidate Recovery Analysis"
    )
    print("=" * 100)

    print(
        "Total cases:",
        total,
    )

    print()

    print("=" * 100)
    print("Recall by Budget")
    print("=" * 100)

    for budget in BUDGETS:

        count = (
            coverage_counts[
                budget
            ]
        )

        print(
            f"@{budget}:",
            f"{count}/{total}",
            f"= {count / total:.4f}",
        )

    print()

    print("=" * 100)
    print(
        "Recovery of @10 Misses"
    )
    print("=" * 100)

    print(
        "@10 misses:",
        len(
            base_misses
        ),
    )

    recovered_20 = [
        record
        for record in base_misses
        if (
            record[
                "first_recovery_budget"
            ]
            == 20
        )
    ]

    recovered_50 = [
        record
        for record in base_misses
        if (
            record[
                "first_recovery_budget"
            ]
            == 50
        )
    ]

    recovered_100 = [
        record
        for record in base_misses
        if (
            record[
                "first_recovery_budget"
            ]
            == 100
        )
    ]

    never = [
        record
        for record in base_misses
        if (
            record[
                "first_recovery_budget"
            ]
            is None
        )
    ]

    print(
        "Recovered first at @20:",
        len(
            recovered_20
        ),
    )

    print(
        "Recovered first at @50:",
        len(
            recovered_50
        ),
    )

    print(
        "Recovered first at @100:",
        len(
            recovered_100
        ),
    )

    print(
        "Still missed at @100:",
        len(
            never
        ),
    )

    for label, subset in [
        (
            "Recovered @20",
            recovered_20,
        ),
        (
            "Recovered @50",
            recovered_50,
        ),
        (
            "Recovered @100",
            recovered_100,
        ),
        (
            "Never recovered",
            never,
        ),
    ]:

        print()
        print(
            label + ":"
        )

        if not subset:
            print(
                "  None"
            )
            continue

        for record in subset:

            print(
                " ",
                record[
                    "benchmark_id"
                ],
            )

    print()

    print("=" * 100)
    print(
        "Per-project Recovery"
    )
    print("=" * 100)

    for project in sorted(
        project_stats
    ):

        stats = (
            project_stats[
                project
            ]
        )

        print()

        print(
            project
        )

        print(
            "  Total:",
            stats[
                "total"
            ],
        )

        print(
            "  Recall @10:",
            stats[
                "recall_at_10"
            ],
        )

        print(
            "  Recall @20:",
            stats[
                "recall_at_20"
            ],
        )

        print(
            "  Recall @50:",
            stats[
                "recall_at_50"
            ],
        )

        print(
            "  Recall @100:",
            stats[
                "recall_at_100"
            ],
        )

        print(
            "  First recovered @20:",
            stats[
                "recovered_20"
            ],
        )

        print(
            "  First recovered @50:",
            stats[
                "recovered_50"
            ],
        )

        print(
            "  First recovered @100:",
            stats[
                "recovered_100"
            ],
        )

        print(
            "  Never recovered:",
            stats[
                "never_recovered"
            ],
        )

    # =========================================================
    # Potential ceilings
    # =========================================================

    print()

    print("=" * 100)
    print(
        "Potential End-to-End Ceilings"
    )
    print("=" * 100)

    for budget in BUDGETS:

        count = (
            coverage_counts[
                budget
            ]
        )

        print(
            f"Perfect verifier with K={budget}:",
            f"{count}/{total}",
            f"= {count / total:.4f}",
        )

    report = {
        "total_cases": (
            total
        ),

        "coverage": {
            str(budget): (
                coverage_counts[
                    budget
                ]
            )
            for budget in BUDGETS
        },

        "base_10_misses": (
            len(
                base_misses
            )
        ),

        "recovery": {
            "first_at_20": [
                record[
                    "benchmark_id"
                ]
                for record
                in recovered_20
            ],

            "first_at_50": [
                record[
                    "benchmark_id"
                ]
                for record
                in recovered_50
            ],

            "first_at_100": [
                record[
                    "benchmark_id"
                ]
                for record
                in recovered_100
            ],

            "never_recovered": [
                record[
                    "benchmark_id"
                ]
                for record
                in never
            ],
        },

        "per_project": {
            project: dict(
                stats
            )
            for project, stats
            in project_stats.items()
        },

        "records": (
            records
        ),
    }

    save_json(
        args.output,
        report,
    )

    print()

    print("=" * 100)
    print("Saved")
    print("=" * 100)

    print(
        args.output
    )


if __name__ == "__main__":
    main()