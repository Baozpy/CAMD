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
    / "fse_ase_retrieval_dev_frozen_candidate_pools.json"
)


DEFAULT_DETECTOR_DIR = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "detector"
)


DEFAULT_VERIFIER_DIR = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "verifier_dev"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "verifier_dev_summary.json"
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


def candidate_key(
    item,
):
    return (
        item["class_name"],
        item["source_file"],
        int(item["start_line"]),
        int(item["end_line"]),
    )


def get_gt_keys(
    case,
):
    return {
        (
            gt.class_name,
            gt.source_file,
            gt.start_line,
            gt.end_line,
        )
        for gt in case.ground_truth
    }


def get_best_gt_rank(
    ranking,
    gt_keys,
    rank_field: str,
):

    ranks = []

    for item in ranking:

        if (
            candidate_key(item)
            in gt_keys
        ):
            ranks.append(
                int(
                    item[
                        rank_field
                    ]
                )
            )

    if not ranks:
        return None

    return min(
        ranks
    )


def reciprocal_rank(
    rank,
) -> float:

    if rank is None:
        return 0.0

    return 1.0 / rank


def metric_hits(
    records,
    field: str,
    k: int,
) -> int:

    return sum(
        (
            record[field]
            is not None
            and record[field] <= k
        )
        for record in records
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--detector-dir",
        type=Path,
        default=DEFAULT_DETECTOR_DIR,
    )

    parser.add_argument(
        "--verifier-dir",
        type=Path,
        default=DEFAULT_VERIFIER_DIR,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=10,
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
    missing = []

    for benchmark_id in benchmark_ids:

        case = loader.load_case(
            benchmark_id,
            budget=args.budget,
        )

        detector_path = (
            args.detector_dir
            / (
                f"{benchmark_id}"
                f"_budget{args.budget}.json"
            )
        )

        verifier_path = (
            args.verifier_dir
            / (
                f"{benchmark_id}"
                f"_budget{args.budget}"
                f"_top{args.verify_top_k}.json"
            )
        )

        if (
            not detector_path.exists()
            or not verifier_path.exists()
        ):
            missing.append(
                benchmark_id
            )
            continue

        detector_data = (
            load_json(
                detector_path
            )
        )

        verifier_data = (
            load_json(
                verifier_path
            )
        )

        if not verifier_data.get(
            "completed",
            False,
        ):
            missing.append(
                benchmark_id
            )
            continue

        gt_keys = get_gt_keys(
            case
        )

        detector_ranking = (
            detector_data[
                "ranking"
            ]
        )

        judge_ranking = (
            verifier_data.get(
                "judge_ranking",
                [],
            )
        )

        detector_best_gt_rank = (
            get_best_gt_rank(
                detector_ranking,
                gt_keys,
                "detector_rank",
            )
        )

        judge_best_gt_rank = (
            get_best_gt_rank(
                judge_ranking,
                gt_keys,
                "judge_rank",
            )
        )

        shortlist_keys = {
            candidate_key(
                item
            )
            for item
            in detector_ranking[
                :args.verify_top_k
            ]
        }

        shortlist_recall = bool(
            gt_keys
            & shortlist_keys
        )

        detector_top1 = (
            detector_best_gt_rank
            == 1
        )

        judge_top1 = (
            judge_best_gt_rank
            == 1
        )

        if (
            not detector_top1
            and judge_top1
        ):
            transition = (
                "corrected"
            )

        elif (
            detector_top1
            and not judge_top1
        ):
            transition = (
                "regressed"
            )

        elif (
            detector_top1
            and judge_top1
        ):
            transition = (
                "retained_correct"
            )

        else:
            transition = (
                "retained_wrong"
            )

        detector_top_candidate = (
            detector_ranking[0]
            if detector_ranking
            else None
        )

        judge_top_candidate = (
            judge_ranking[0]
            if judge_ranking
            else None
        )

        records.append(
            {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": (
                    case.project
                ),
                "bug_id": (
                    case.bug_id
                ),
                "candidate_recall": (
                    bool(
                        case.candidate_recall
                    )
                ),
                "shortlist_recall": (
                    shortlist_recall
                ),
                "detector_best_gt_rank": (
                    detector_best_gt_rank
                ),
                "judge_best_gt_rank": (
                    judge_best_gt_rank
                ),
                "detector_top1": (
                    detector_top1
                ),
                "judge_top1": (
                    judge_top1
                ),
                "transition": (
                    transition
                ),
                "detector_top_candidate": (
                    detector_top_candidate
                ),
                "judge_top_candidate": (
                    judge_top_candidate
                ),
            }
        )

    # =========================================================
    # Basic completeness
    # =========================================================

    print("=" * 100)
    print("CAMD Verifier Dev Evaluation")
    print("=" * 100)

    print(
        "Expected cases:",
        len(benchmark_ids),
    )

    print(
        "Completed cases:",
        len(records),
    )

    print(
        "Missing/incomplete:",
        len(missing),
    )

    if missing:
        print(
            "Missing:",
            ", ".join(
                missing
            ),
        )

    if not records:
        return

    total = len(
        records
    )

    # =========================================================
    # Retrieval / shortlist coverage
    # =========================================================

    retrieval_recalled = sum(
        record[
            "candidate_recall"
        ]
        for record in records
    )

    shortlist_recalled = sum(
        record[
            "shortlist_recall"
        ]
        for record in records
    )

    print()
    print("=" * 100)
    print("Candidate Coverage")
    print("=" * 100)

    print(
        "Retriever recall:",
        f"{retrieval_recalled}/{total}",
        f"= {retrieval_recalled / total:.4f}",
    )

    print(
        "Detector Top-K shortlist recall:",
        f"{shortlist_recalled}/{total}",
        f"= {shortlist_recalled / total:.4f}",
    )

    # =========================================================
    # Detector vs Judge
    # =========================================================

    print()
    print("=" * 100)
    print("Detector vs Judge End-to-End")
    print("=" * 100)

    detector_metrics = {}
    judge_metrics = {}

    for k in [
        1,
        3,
        5,
        10,
    ]:

        detector_hits = (
            metric_hits(
                records,
                "detector_best_gt_rank",
                k,
            )
        )

        judge_hits = (
            metric_hits(
                records,
                "judge_best_gt_rank",
                k,
            )
        )

        detector_metrics[
            f"top{k}"
        ] = (
            detector_hits
            / total
        )

        judge_metrics[
            f"top{k}"
        ] = (
            judge_hits
            / total
        )

        print(
            f"Top-{k}:"
        )

        print(
            "  Detector:",
            f"{detector_hits}/{total}",
            f"= {detector_hits / total:.4f}",
        )

        print(
            "  Judge:   ",
            f"{judge_hits}/{total}",
            f"= {judge_hits / total:.4f}",
        )

        print(
            "  Delta:   ",
            f"{(judge_hits - detector_hits) / total:+.4f}",
        )

    detector_mrr = sum(
        reciprocal_rank(
            record[
                "detector_best_gt_rank"
            ]
        )
        for record in records
    ) / total

    judge_mrr = sum(
        reciprocal_rank(
            record[
                "judge_best_gt_rank"
            ]
        )
        for record in records
    ) / total

    print()
    print(
        "MRR:"
    )

    print(
        "  Detector:",
        f"{detector_mrr:.4f}",
    )

    print(
        "  Judge:   ",
        f"{judge_mrr:.4f}",
    )

    print(
        "  Delta:   ",
        f"{judge_mrr - detector_mrr:+.4f}",
    )

    # =========================================================
    # Conditional verification
    # =========================================================

    conditional_records = [
        record
        for record in records
        if record[
            "shortlist_recall"
        ]
    ]

    conditional_total = len(
        conditional_records
    )

    print()
    print("=" * 100)
    print(
        "Conditional on GT in Detector Shortlist"
    )
    print("=" * 100)

    print(
        "Cases:",
        conditional_total,
    )

    conditional_metrics = {}

    for k in [
        1,
        3,
        5,
        10,
    ]:

        detector_hits = (
            metric_hits(
                conditional_records,
                "detector_best_gt_rank",
                k,
            )
        )

        judge_hits = (
            metric_hits(
                conditional_records,
                "judge_best_gt_rank",
                k,
            )
        )

        conditional_metrics[
            f"detector_top{k}"
        ] = (
            detector_hits
            / conditional_total
        )

        conditional_metrics[
            f"judge_top{k}"
        ] = (
            judge_hits
            / conditional_total
        )

        print(
            f"Top-{k}: "
            f"Detector "
            f"{detector_hits}/"
            f"{conditional_total} "
            f"= "
            f"{detector_hits / conditional_total:.4f}"
            " | "
            f"Judge "
            f"{judge_hits}/"
            f"{conditional_total} "
            f"= "
            f"{judge_hits / conditional_total:.4f}"
        )

    conditional_detector_mrr = sum(
        reciprocal_rank(
            record[
                "detector_best_gt_rank"
            ]
        )
        for record
        in conditional_records
    ) / conditional_total

    conditional_judge_mrr = sum(
        reciprocal_rank(
            record[
                "judge_best_gt_rank"
            ]
        )
        for record
        in conditional_records
    ) / conditional_total

    print(
        "Conditional MRR:",
        f"Detector "
        f"{conditional_detector_mrr:.4f}",
        "|",
        f"Judge "
        f"{conditional_judge_mrr:.4f}",
    )

    # =========================================================
    # Corrections / regressions
    # =========================================================

    transition_counts = defaultdict(
        int
    )

    for record in records:
        transition_counts[
            record[
                "transition"
            ]
        ] += 1

    corrected = [
        record
        for record in records
        if record[
            "transition"
        ]
        == "corrected"
    ]

    regressed = [
        record
        for record in records
        if record[
            "transition"
        ]
        == "regressed"
    ]

    retained_correct = [
        record
        for record in records
        if record[
            "transition"
        ]
        == "retained_correct"
    ]

    retained_wrong = [
        record
        for record in records
        if record[
            "transition"
        ]
        == "retained_wrong"
    ]

    print()
    print("=" * 100)
    print("Top-1 Transition Analysis")
    print("=" * 100)

    print(
        "Corrected:",
        len(corrected),
    )

    print(
        "Regressed:",
        len(regressed),
    )

    print(
        "Retained correct:",
        len(retained_correct),
    )

    print(
        "Retained wrong:",
        len(retained_wrong),
    )

    print(
        "Net Top-1 gain:",
        len(corrected)
        - len(regressed),
    )

    if corrected:

        print()
        print("Corrected cases:")

        for record in corrected:

            print(
                " ",
                record[
                    "benchmark_id"
                ],
                "Detector rank=",
                record[
                    "detector_best_gt_rank"
                ],
                "-> Judge rank=",
                record[
                    "judge_best_gt_rank"
                ],
            )

    if regressed:

        print()
        print("Regressed cases:")

        for record in regressed:

            print(
                " ",
                record[
                    "benchmark_id"
                ],
                "Detector rank=",
                record[
                    "detector_best_gt_rank"
                ],
                "-> Judge rank=",
                record[
                    "judge_best_gt_rank"
                ],
            )

    # =========================================================
    # Rank movement
    # =========================================================

    print()
    print("=" * 100)
    print("GT Rank Movement")
    print("=" * 100)

    improved = []
    unchanged = []
    worsened = []

    for record in records:

        detector_rank = (
            record[
                "detector_best_gt_rank"
            ]
        )

        judge_rank = (
            record[
                "judge_best_gt_rank"
            ]
        )

        if (
            detector_rank is None
            or judge_rank is None
        ):
            continue

        if judge_rank < detector_rank:
            improved.append(
                record
            )

        elif judge_rank > detector_rank:
            worsened.append(
                record
            )

        else:
            unchanged.append(
                record
            )

    print(
        "Improved rank:",
        len(improved),
    )

    print(
        "Unchanged rank:",
        len(unchanged),
    )

    print(
        "Worsened rank:",
        len(worsened),
    )

    # =========================================================
    # Per-project
    # =========================================================

    print()
    print("=" * 100)
    print("Per-project Top-1")
    print("=" * 100)

    project_summary = {}

    projects = sorted(
        {
            record[
                "project"
            ]
            for record in records
        }
    )

    for project in projects:

        subset = [
            record
            for record in records
            if record[
                "project"
            ]
            == project
        ]

        n = len(
            subset
        )

        detector_hits = sum(
            record[
                "detector_top1"
            ]
            for record in subset
        )

        judge_hits = sum(
            record[
                "judge_top1"
            ]
            for record in subset
        )

        project_summary[
            project
        ] = {
            "n": n,
            "detector_top1": (
                detector_hits
                / n
            ),
            "judge_top1": (
                judge_hits
                / n
            ),
            "delta": (
                judge_hits
                - detector_hits
            ) / n,
        }

        print(
            project
        )

        print(
            "  Detector:",
            f"{detector_hits}/{n}",
            f"= {detector_hits / n:.4f}",
        )

        print(
            "  Judge:   ",
            f"{judge_hits}/{n}",
            f"= {judge_hits / n:.4f}",
        )

        print(
            "  Delta:   ",
            f"{(judge_hits - detector_hits) / n:+.4f}",
        )

    # =========================================================
    # Save machine-readable summary
    # =========================================================

    summary = {
        "configuration": {
            "budget": (
                args.budget
            ),
            "verify_top_k": (
                args.verify_top_k
            ),
            "total_cases": (
                total
            ),
        },

        "coverage": {
            "retrieval_recall_count": (
                retrieval_recalled
            ),
            "retrieval_recall": (
                retrieval_recalled
                / total
            ),
            "shortlist_recall_count": (
                shortlist_recalled
            ),
            "shortlist_recall": (
                shortlist_recalled
                / total
            ),
        },

        "detector": {
            **detector_metrics,
            "mrr": (
                detector_mrr
            ),
        },

        "judge": {
            **judge_metrics,
            "mrr": (
                judge_mrr
            ),
        },

        "conditional": {
            "n": (
                conditional_total
            ),
            **conditional_metrics,
            "detector_mrr": (
                conditional_detector_mrr
            ),
            "judge_mrr": (
                conditional_judge_mrr
            ),
        },

        "transitions": {
            "corrected": (
                len(corrected)
            ),
            "regressed": (
                len(regressed)
            ),
            "retained_correct": (
                len(
                    retained_correct
                )
            ),
            "retained_wrong": (
                len(
                    retained_wrong
                )
            ),
            "net_top1_gain": (
                len(corrected)
                - len(regressed)
            ),
            "corrected_cases": [
                record[
                    "benchmark_id"
                ]
                for record
                in corrected
            ],
            "regressed_cases": [
                record[
                    "benchmark_id"
                ]
                for record
                in regressed
            ],
        },

        "rank_movement": {
            "improved": (
                len(improved)
            ),
            "unchanged": (
                len(unchanged)
            ),
            "worsened": (
                len(worsened)
            ),
        },

        "per_project": (
            project_summary
        ),

        "records": (
            records
        ),
    }

    save_json(
        args.output,
        summary,
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