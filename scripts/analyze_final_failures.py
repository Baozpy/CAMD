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


DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "final_verifier_summary.json"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "final_failure_analysis.json"
)


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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


def format_candidate(
    item,
) -> str:

    if not item:
        return "N/A"

    return (
        f"{item.get('class_name', '')}."
        f"{item.get('method_name', '')}"
        f"[{item.get('start_line', '?')}-"
        f"{item.get('end_line', '?')}]"
    )


def ground_truth_methods(
    case,
):

    methods = []

    for gt in case.ground_truth:

        methods.append(
            {
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
        )

    return methods


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    loader = FrozenCandidateLoader(
        args.manifest
    )

    summary = load_json(
        args.summary
    )

    records = summary[
        "records"
    ]

    # =========================================================
    # Main failure categories
    # =========================================================

    retrieval_failures = []

    detector_ranking_failures = []

    detector_successes = []

    judge_regressions = []

    judge_corrections = []

    judge_rank_improvements = []

    judge_rank_regressions = []

    unchanged = []

    per_project = defaultdict(
        lambda: {
            "total": 0,
            "retrieval_failure": 0,
            "retrieval_success": 0,
            "detector_top1_correct": 0,
            "detector_top1_wrong_given_recall": 0,
            "judge_top1_correct": 0,
            "judge_top1_wrong_given_recall": 0,
            "judge_corrected": 0,
            "judge_regressed": 0,
        }
    )

    for record in records:

        benchmark_id = (
            record[
                "benchmark_id"
            ]
        )

        project = (
            record[
                "project"
            ]
        )

        case = loader.load_case(
            benchmark_id,
            budget=args.budget,
        )

        gt_methods = (
            ground_truth_methods(
                case
            )
        )

        detector_rank = (
            record.get(
                "detector_best_gt_rank"
            )
        )

        judge_rank = (
            record.get(
                "judge_best_gt_rank"
            )
        )

        candidate_recall = bool(
            record.get(
                "candidate_recall",
                False,
            )
        )

        shortlist_recall = bool(
            record.get(
                "shortlist_recall",
                False,
            )
        )

        detector_top1 = (
            detector_rank == 1
        )

        judge_top1 = (
            judge_rank == 1
        )

        base_item = {
            "benchmark_id": (
                benchmark_id
            ),
            "project": (
                project
            ),
            "bug_id": (
                record.get(
                    "bug_id"
                )
            ),
            "candidate_recall": (
                candidate_recall
            ),
            "shortlist_recall": (
                shortlist_recall
            ),
            "detector_best_gt_rank": (
                detector_rank
            ),
            "judge_best_gt_rank": (
                judge_rank
            ),
            "ground_truth": (
                gt_methods
            ),
            "detector_top_candidate": (
                record.get(
                    "detector_top_candidate"
                )
            ),
            "judge_top_candidate": (
                record.get(
                    "judge_top_candidate"
                )
            ),
        }

        stats = per_project[
            project
        ]

        stats[
            "total"
        ] += 1

        # =====================================================
        # Retrieval decomposition
        # =====================================================

        if not candidate_recall:

            stats[
                "retrieval_failure"
            ] += 1

            retrieval_failures.append(
                base_item
            )

        else:

            stats[
                "retrieval_success"
            ] += 1

            if detector_top1:

                stats[
                    "detector_top1_correct"
                ] += 1

                detector_successes.append(
                    base_item
                )

            else:

                stats[
                    "detector_top1_wrong_given_recall"
                ] += 1

                detector_ranking_failures.append(
                    base_item
                )

            if judge_top1:

                stats[
                    "judge_top1_correct"
                ] += 1

            else:

                stats[
                    "judge_top1_wrong_given_recall"
                ] += 1

        # =====================================================
        # Judge transition analysis
        # =====================================================

        if (
            not detector_top1
            and judge_top1
        ):

            stats[
                "judge_corrected"
            ] += 1

            judge_corrections.append(
                base_item
            )

        elif (
            detector_top1
            and not judge_top1
        ):

            stats[
                "judge_regressed"
            ] += 1

            judge_regressions.append(
                base_item
            )

        # =====================================================
        # Rank movement
        # =====================================================

        if (
            detector_rank is not None
            and judge_rank is not None
        ):

            if judge_rank < detector_rank:

                judge_rank_improvements.append(
                    base_item
                )

            elif judge_rank > detector_rank:

                judge_rank_regressions.append(
                    base_item
                )

            else:

                unchanged.append(
                    base_item
                )

    # =========================================================
    # Overall counts
    # =========================================================

    total = len(
        records
    )

    retrieval_success_count = (
        total
        - len(
            retrieval_failures
        )
    )

    detector_correct_count = sum(
        1
        for record in records
        if (
            record.get(
                "detector_best_gt_rank"
            )
            == 1
        )
    )

    judge_correct_count = sum(
        1
        for record in records
        if (
            record.get(
                "judge_best_gt_rank"
            )
            == 1
        )
    )

    print("=" * 100)
    print("CAMD Final Failure Analysis")
    print("=" * 100)

    print(
        "Total method-applicable bugs:",
        total,
    )

    print()
    print("=" * 100)
    print("Failure Decomposition")
    print("=" * 100)

    print(
        "Retrieval failures:",
        len(
            retrieval_failures
        ),
    )

    print(
        "Retrieval successes:",
        retrieval_success_count,
    )

    print(
        "Detector Top-1 correct:",
        detector_correct_count,
    )

    print(
        "Detector ranking failures "
        "given retrieval success:",
        len(
            detector_ranking_failures
        ),
    )

    print(
        "Judge Top-1 correct:",
        judge_correct_count,
    )

    print(
        "Judge corrections:",
        len(
            judge_corrections
        ),
    )

    print(
        "Judge regressions:",
        len(
            judge_regressions
        ),
    )

    # =========================================================
    # Retrieval failures
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Retrieval Failures"
    )
    print("=" * 100)

    for item in (
        retrieval_failures
    ):

        print()
        print(
            item[
                "benchmark_id"
            ]
        )

        print(
            "  Ground truth:"
        )

        for gt in (
            item[
                "ground_truth"
            ]
        ):

            print(
                "   ",
                format_candidate(
                    gt
                ),
            )

    # =========================================================
    # Detector failures
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Detector Ranking Failures "
        "(GT was retrieved)"
    )
    print("=" * 100)

    for item in (
        detector_ranking_failures
    ):

        print()
        print(
            item[
                "benchmark_id"
            ]
        )

        print(
            "  Best GT Detector rank:",
            item[
                "detector_best_gt_rank"
            ],
        )

        print(
            "  Best GT Judge rank:",
            item[
                "judge_best_gt_rank"
            ],
        )

        print(
            "  Detector Top-1:",
            format_candidate(
                item[
                    "detector_top_candidate"
                ]
            ),
        )

        print(
            "  Judge Top-1:",
            format_candidate(
                item[
                    "judge_top_candidate"
                ]
            ),
        )

        print(
            "  Ground truth:"
        )

        for gt in (
            item[
                "ground_truth"
            ]
        ):

            print(
                "   ",
                format_candidate(
                    gt
                ),
            )

    # =========================================================
    # Judge regression
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Judge Regressions"
    )
    print("=" * 100)

    if not judge_regressions:

        print(
            "None"
        )

    for item in (
        judge_regressions
    ):

        print()
        print(
            item[
                "benchmark_id"
            ]
        )

        print(
            "  Detector GT rank:",
            item[
                "detector_best_gt_rank"
            ],
        )

        print(
            "  Judge GT rank:",
            item[
                "judge_best_gt_rank"
            ],
        )

        print(
            "  Detector Top-1:",
            format_candidate(
                item[
                    "detector_top_candidate"
                ]
            ),
        )

        print(
            "  Judge Top-1:",
            format_candidate(
                item[
                    "judge_top_candidate"
                ]
            ),
        )

        print(
            "  Ground truth:"
        )

        for gt in (
            item[
                "ground_truth"
            ]
        ):

            print(
                "   ",
                format_candidate(
                    gt
                ),
            )

    # =========================================================
    # Rank improvements
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Judge Rank Improvements"
    )
    print("=" * 100)

    if not judge_rank_improvements:

        print(
            "None"
        )

    for item in (
        judge_rank_improvements
    ):

        print(
            item[
                "benchmark_id"
            ],
            ":",
            item[
                "detector_best_gt_rank"
            ],
            "->",
            item[
                "judge_best_gt_rank"
            ],
        )

    # =========================================================
    # Per-project decomposition
    # =========================================================

    print()
    print("=" * 100)
    print(
        "Per-project Failure Decomposition"
    )
    print("=" * 100)

    for project in sorted(
        per_project
    ):

        stats = (
            per_project[
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
            "  Retrieval success:",
            stats[
                "retrieval_success"
            ],
        )

        print(
            "  Retrieval failure:",
            stats[
                "retrieval_failure"
            ],
        )

        print(
            "  Detector Top-1 correct:",
            stats[
                "detector_top1_correct"
            ],
        )

        print(
            "  Detector ranking failure:",
            stats[
                "detector_top1_wrong_given_recall"
            ],
        )

        print(
            "  Judge Top-1 correct:",
            stats[
                "judge_top1_correct"
            ],
        )

        print(
            "  Judge corrected:",
            stats[
                "judge_corrected"
            ],
        )

        print(
            "  Judge regressed:",
            stats[
                "judge_regressed"
            ],
        )

    # =========================================================
    # Bottleneck shares
    # =========================================================

    total_end_to_end_failures = (
        total
        - detector_correct_count
    )

    print()
    print("=" * 100)
    print(
        "Detector End-to-End Failure Attribution"
    )
    print("=" * 100)

    print(
        "Total Detector Top-1 failures:",
        total_end_to_end_failures,
    )

    if total_end_to_end_failures:

        retrieval_share = (
            len(
                retrieval_failures
            )
            / total_end_to_end_failures
        )

        detector_share = (
            len(
                detector_ranking_failures
            )
            / total_end_to_end_failures
        )

        print(
            "Due to retrieval:",
            f"{len(retrieval_failures)}/"
            f"{total_end_to_end_failures}",
            f"= {retrieval_share:.4f}",
        )

        print(
            "Due to Detector ranking:",
            f"{len(detector_ranking_failures)}/"
            f"{total_end_to_end_failures}",
            f"= {detector_share:.4f}",
        )

    # =========================================================
    # Save machine-readable report
    # =========================================================

    report = {
        "configuration": {
            "budget": (
                args.budget
            ),
            "total_cases": (
                total
            ),
        },

        "overall": {
            "retrieval_failures": (
                len(
                    retrieval_failures
                )
            ),
            "retrieval_successes": (
                retrieval_success_count
            ),
            "detector_top1_correct": (
                detector_correct_count
            ),
            "detector_ranking_failures_given_recall": (
                len(
                    detector_ranking_failures
                )
            ),
            "judge_top1_correct": (
                judge_correct_count
            ),
            "judge_corrections": (
                len(
                    judge_corrections
                )
            ),
            "judge_regressions": (
                len(
                    judge_regressions
                )
            ),
        },

        "retrieval_failures": (
            retrieval_failures
        ),

        "detector_ranking_failures": (
            detector_ranking_failures
        ),

        "judge_corrections": (
            judge_corrections
        ),

        "judge_regressions": (
            judge_regressions
        ),

        "judge_rank_improvements": (
            judge_rank_improvements
        ),

        "judge_rank_regressions": (
            judge_rank_regressions
        ),

        "per_project": {
            project: dict(stats)
            for project, stats
            in per_project.items()
        },
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