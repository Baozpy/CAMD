from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
    / "recovered_candidate_positions.json"
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


def candidate_identifier(
    candidate,
) -> str:

    return (
        f"{candidate.class_name}."
        f"{candidate.method_name}"
        f"[{candidate.start_line}-"
        f"{candidate.end_line}]"
    )


def gt_identifier(
    gt,
) -> str:

    return (
        f"{gt.class_name}."
        f"{gt.method_name}"
        f"[{gt.start_line}-"
        f"{gt.end_line}]"
    )


def get_matching_gt_candidates(
    case,
):

    gt_keys = {
        gt_key(gt)
        for gt in case.ground_truth
    }

    return [
        candidate
        for candidate in case.candidates
        if candidate_key(candidate)
        in gt_keys
    ]


def get_admission_sources(
    candidate,
):

    sources = []

    if getattr(
        candidate,
        "from_base",
        False,
    ):
        sources.append(
            "base"
        )

    if getattr(
        candidate,
        "from_stack",
        False,
    ):
        sources.append(
            "stack"
        )

    if getattr(
        candidate,
        "from_call",
        False,
    ):
        sources.append(
            "call"
        )

    if not sources:
        sources.append(
            "unknown"
        )

    return sources


def candidate_record(
    candidate,
):

    return {
        "identifier": (
            candidate_identifier(
                candidate
            )
        ),

        "class_name": (
            candidate.class_name
        ),

        "method_name": (
            candidate.method_name
        ),

        "source_file": (
            candidate.source_file
        ),

        "start_line": (
            candidate.start_line
        ),

        "end_line": (
            candidate.end_line
        ),

        "pool_position": (
            candidate.pool_position
        ),

        "base_rank": getattr(
            candidate,
            "base_rank",
            None,
        ),

        "base_score": getattr(
            candidate,
            "base_score",
            None,
        ),

        "from_base": bool(
            getattr(
                candidate,
                "from_base",
                False,
            )
        ),

        "from_stack": bool(
            getattr(
                candidate,
                "from_stack",
                False,
            )
        ),

        "stack_depth": getattr(
            candidate,
            "stack_depth",
            None,
        ),

        "from_call": bool(
            getattr(
                candidate,
                "from_call",
                False,
            )
        ),

        "call_depth": getattr(
            candidate,
            "call_depth",
            None,
        ),

        "admission_sources": (
            get_admission_sources(
                candidate
            )
        ),
    }


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

    source_counter = Counter()

    first_recovery_counter = Counter()

    project_recovery = defaultdict(
        Counter
    )

    recovered_pool_positions = []

    recovered_base_ranks = []

    for benchmark_id in benchmark_ids:

        cases = {}

        matching_by_budget = {}

        for budget in BUDGETS:

            case = loader.load_case(
                benchmark_id,
                budget=budget,
            )

            cases[
                budget
            ] = case

            matching_by_budget[
                budget
            ] = (
                get_matching_gt_candidates(
                    case
                )
            )

        # We only care about bugs missed at @10.
        if matching_by_budget[10]:
            continue

        base_case = cases[10]

        project = (
            base_case.project
        )

        first_recovery_budget = None

        for budget in [
            20,
            50,
            100,
        ]:

            if matching_by_budget[
                budget
            ]:

                first_recovery_budget = (
                    budget
                )

                break

        gt_methods = [
            {
                "identifier": (
                    gt_identifier(
                        gt
                    )
                ),
                "class_name": (
                    gt.class_name
                ),
                "method_name": (
                    gt.method_name
                ),
                "source_file": (
                    gt.source_file
                ),
                "start_line": (
                    gt.start_line
                ),
                "end_line": (
                    gt.end_line
                ),
            }
            for gt
            in base_case.ground_truth
        ]

        budget_records = {}

        for budget in BUDGETS:

            matches = (
                matching_by_budget[
                    budget
                ]
            )

            budget_records[
                str(budget)
            ] = {
                "recalled": bool(
                    matches
                ),

                "candidate_count": (
                    len(
                        cases[
                            budget
                        ].candidates
                    )
                ),

                "matching_gt_candidates": [
                    candidate_record(
                        candidate
                    )
                    for candidate
                    in matches
                ],
            }

        record = {
            "benchmark_id": (
                benchmark_id
            ),

            "project": (
                project
            ),

            "bug_id": (
                base_case.bug_id
            ),

            "ground_truth": (
                gt_methods
            ),

            "first_recovery_budget": (
                first_recovery_budget
            ),

            "budgets": (
                budget_records
            ),
        }

        records.append(
            record
        )

        if first_recovery_budget is None:

            first_recovery_counter[
                "never"
            ] += 1

            project_recovery[
                project
            ][
                "never"
            ] += 1

            continue

        first_recovery_counter[
            str(
                first_recovery_budget
            )
        ] += 1

        project_recovery[
            project
        ][
            str(
                first_recovery_budget
            )
        ] += 1

        first_candidates = (
            matching_by_budget[
                first_recovery_budget
            ]
        )

        for candidate in (
            first_candidates
        ):

            recovered_pool_positions.append(
                candidate.pool_position
            )

            base_rank = getattr(
                candidate,
                "base_rank",
                None,
            )

            if base_rank is not None:

                recovered_base_ranks.append(
                    base_rank
                )

            for source in (
                get_admission_sources(
                    candidate
                )
            ):

                source_counter[
                    source
                ] += 1

    # =========================================================
    # Split cases
    # =========================================================

    recovered = [
        record
        for record in records
        if record[
            "first_recovery_budget"
        ]
        is not None
    ]

    never_recovered = [
        record
        for record in records
        if record[
            "first_recovery_budget"
        ]
        is None
    ]

    # =========================================================
    # Header
    # =========================================================

    print("=" * 100)
    print(
        "CAMD Recovered Candidate Position Analysis"
    )
    print("=" * 100)

    print(
        "@10 retrieval misses:",
        len(
            records
        ),
    )

    print(
        "Recovered by @100:",
        len(
            recovered
        ),
    )

    print(
        "Still absent @100:",
        len(
            never_recovered
        ),
    )

    # =========================================================
    # Recovery summary
    # =========================================================

    print()
    print("=" * 100)
    print(
        "First Recovery Budget"
    )
    print("=" * 100)

    print(
        "@20:",
        first_recovery_counter[
            "20"
        ],
    )

    print(
        "@50:",
        first_recovery_counter[
            "50"
        ],
    )

    print(
        "@100:",
        first_recovery_counter[
            "100"
        ],
    )

    print(
        "Never:",
        first_recovery_counter[
            "never"
        ],
    )

    # =========================================================
    # Individual recovered cases
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Recovered Cases"
    )
    print("=" * 100)

    for record in recovered:

        budget = (
            record[
                "first_recovery_budget"
            ]
        )

        matches = (
            record[
                "budgets"
            ][
                str(
                    budget
                )
            ][
                "matching_gt_candidates"
            ]
        )

        print()
        print(
            record[
                "benchmark_id"
            ],
            f"(first recovered @{budget})",
        )

        for match in matches:

            print(
                "  GT:",
                match[
                    "identifier"
                ],
            )

            print(
                "    pool_position:",
                match[
                    "pool_position"
                ],
            )

            print(
                "    base_rank:",
                match[
                    "base_rank"
                ],
            )

            print(
                "    base_score:",
                match[
                    "base_score"
                ],
            )

            print(
                "    sources:",
                ", ".join(
                    match[
                        "admission_sources"
                    ]
                ),
            )

            print(
                "    stack_depth:",
                match[
                    "stack_depth"
                ],
            )

            print(
                "    call_depth:",
                match[
                    "call_depth"
                ],
            )

    # =========================================================
    # Aggregate recovered positions
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Recovered Candidate Position Statistics"
    )
    print("=" * 100)

    if recovered_pool_positions:

        sorted_positions = sorted(
            recovered_pool_positions
        )

        print(
            "Observed recovered GT candidates:",
            len(
                sorted_positions
            ),
        )

        print(
            "Min pool position:",
            min(
                sorted_positions
            ),
        )

        print(
            "Max pool position:",
            max(
                sorted_positions
            ),
        )

        mean_position = (
            sum(
                sorted_positions
            )
            / len(
                sorted_positions
            )
        )

        print(
            "Mean pool position:",
            f"{mean_position:.2f}",
        )

        midpoint = (
            len(
                sorted_positions
            )
            // 2
        )

        if (
            len(
                sorted_positions
            )
            % 2
            == 1
        ):

            median_position = (
                sorted_positions[
                    midpoint
                ]
            )

        else:

            median_position = (
                (
                    sorted_positions[
                        midpoint - 1
                    ]
                    + sorted_positions[
                        midpoint
                    ]
                )
                / 2
            )

        print(
            "Median pool position:",
            median_position,
        )

    if recovered_base_ranks:

        sorted_base_ranks = sorted(
            recovered_base_ranks
        )

        print()
        print(
            "GT candidates with observable "
            "base_rank:",
            len(
                sorted_base_ranks
            ),
        )

        print(
            "Min base rank:",
            min(
                sorted_base_ranks
            ),
        )

        print(
            "Max base rank:",
            max(
                sorted_base_ranks
            ),
        )

        print(
            "Mean base rank:",
            f"{sum(sorted_base_ranks) / len(sorted_base_ranks):.2f}",
        )

    # =========================================================
    # Admission source
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Admission Evidence for Recovered GTs"
    )
    print("=" * 100)

    if not source_counter:

        print(
            "No observable admission-source metadata."
        )

    else:

        for source in [
            "base",
            "stack",
            "call",
            "unknown",
        ]:

            if source_counter[
                source
            ]:

                print(
                    source,
                    ":",
                    source_counter[
                        source
                    ],
                )

    # =========================================================
    # Per-project
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Per-project Recovery Depth"
    )
    print("=" * 100)

    for project in sorted(
        project_recovery
    ):

        stats = (
            project_recovery[
                project
            ]
        )

        print()
        print(
            project
        )

        print(
            "  first @20:",
            stats[
                "20"
            ],
        )

        print(
            "  first @50:",
            stats[
                "50"
            ],
        )

        print(
            "  first @100:",
            stats[
                "100"
            ],
        )

        print(
            "  never:",
            stats[
                "never"
            ],
        )

    # =========================================================
    # Never recovered
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Still Missing at @100"
    )
    print("=" * 100)

    print(
        "Important: the frozen candidate pools do not "
        "contain retrieval metadata for methods that were "
        "never admitted. Therefore base rank, base score, "
        "and exact failure reason cannot be inferred here."
    )

    for record in never_recovered:

        print()
        print(
            record[
                "benchmark_id"
            ]
        )

        for gt in (
            record[
                "ground_truth"
            ]
        ):

            print(
                "  GT:",
                gt[
                    "identifier"
                ],
            )

        print(
            "  Status: absent from frozen "
            "@10/@20/@50/@100 candidate pools"
        )

    # =========================================================
    # Save
    # =========================================================

    report = {
        "configuration": {
            "budgets": (
                BUDGETS
            ),
        },

        "summary": {
            "base_10_misses": (
                len(
                    records
                )
            ),

            "recovered_by_100": (
                len(
                    recovered
                )
            ),

            "never_recovered": (
                len(
                    never_recovered
                )
            ),

            "first_recovery": {
                "20": (
                    first_recovery_counter[
                        "20"
                    ]
                ),
                "50": (
                    first_recovery_counter[
                        "50"
                    ]
                ),
                "100": (
                    first_recovery_counter[
                        "100"
                    ]
                ),
                "never": (
                    first_recovery_counter[
                        "never"
                    ]
                ),
            },

            "admission_sources": (
                dict(
                    source_counter
                )
            ),
        },

        "recovered_pool_positions": (
            recovered_pool_positions
        ),

        "recovered_base_ranks": (
            recovered_base_ranks
        ),

        "per_project": {
            project: dict(
                stats
            )
            for project, stats
            in project_recovery.items()
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