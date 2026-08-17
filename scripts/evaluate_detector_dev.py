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

DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "detector"
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=10,
    )

    return parser.parse_args()


def candidate_key(item):

    return (
        item["class_name"],
        item["source_file"],
        int(item["start_line"]),
        int(item["end_line"]),
    )


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

        result_path = (
            args.result_dir
            / (
                f"{benchmark_id}"
                f"_budget{args.budget}.json"
            )
        )

        if not result_path.exists():

            missing.append(
                benchmark_id
            )
            continue

        with result_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if (
            len(
                data.get(
                    "results",
                    [],
                )
            )
            != len(case.candidates)
        ):

            missing.append(
                benchmark_id
            )
            continue

        gt_keys = {
            (
                gt.class_name,
                gt.source_file,
                gt.start_line,
                gt.end_line,
            )
            for gt in case.ground_truth
        }

        best_gt_rank = None

        for item in data["ranking"]:

            if (
                candidate_key(item)
                in gt_keys
            ):

                rank = int(
                    item[
                        "detector_rank"
                    ]
                )

                if (
                    best_gt_rank is None
                    or rank
                    < best_gt_rank
                ):
                    best_gt_rank = rank

        records.append(
            {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": (
                    case.project
                ),
                "candidate_recall": (
                    bool(
                        case.candidate_recall
                    )
                ),
                "best_gt_rank": (
                    best_gt_rank
                ),
            }
        )

    print("=" * 100)
    print("CAMD Detector Dev Evaluation")
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
            ", ".join(missing),
        )

    if not records:
        return

    total = len(records)

    recalled_records = [
        record
        for record in records
        if record[
            "candidate_recall"
        ]
    ]

    recalled = len(
        recalled_records
    )

    print()
    print("=" * 100)
    print("Candidate Pool")
    print("=" * 100)

    print(
        "Candidate recall:",
        f"{recalled}/{total}",
        f"= {recalled / total:.4f}",
    )

    print()
    print("=" * 100)
    print("Detector End-to-End")
    print("=" * 100)

    for k in [1, 3, 5, 10]:

        hits = sum(
            (
                record[
                    "best_gt_rank"
                ]
                is not None
                and record[
                    "best_gt_rank"
                ]
                <= k
            )
            for record in records
        )

        print(
            f"Top-{k}:",
            f"{hits}/{total}",
            f"= {hits / total:.4f}",
        )

    mrr = sum(
        (
            1.0
            / record[
                "best_gt_rank"
            ]
        )
        if (
            record[
                "best_gt_rank"
            ]
            is not None
        )
        else 0.0
        for record in records
    ) / total

    print(
        "MRR:",
        f"{mrr:.4f}",
    )

    print()
    print("=" * 100)
    print("Detector Conditional on Retrieval Recall")
    print("=" * 100)

    for k in [1, 3, 5, 10]:

        hits = sum(
            (
                record[
                    "best_gt_rank"
                ]
                is not None
                and record[
                    "best_gt_rank"
                ]
                <= k
            )
            for record
            in recalled_records
        )

        print(
            f"Top-{k}:",
            f"{hits}/{recalled}",
            f"= {hits / recalled:.4f}",
        )

    conditional_mrr = sum(
        (
            1.0
            / record[
                "best_gt_rank"
            ]
        )
        if (
            record[
                "best_gt_rank"
            ]
            is not None
        )
        else 0.0
        for record
        in recalled_records
    ) / recalled

    print(
        "Conditional MRR:",
        f"{conditional_mrr:.4f}",
    )

    print()
    print("=" * 100)
    print("GT Rank Distribution")
    print("=" * 100)

    rank_counts = defaultdict(
        int
    )

    for record in records:

        rank = (
            record[
                "best_gt_rank"
            ]
        )

        if rank is None:
            rank_counts[
                "MISS"
            ] += 1
        else:
            rank_counts[
                str(rank)
            ] += 1

    numeric_ranks = sorted(
        (
            int(key)
            for key
            in rank_counts
            if key != "MISS"
        )
    )

    for rank in numeric_ranks:

        print(
            f"rank {rank}:",
            rank_counts[
                str(rank)
            ],
        )

    if rank_counts["MISS"]:

        print(
            "MISS:",
            rank_counts[
                "MISS"
            ],
        )

    print()
    print("=" * 100)
    print("Per-project Top-1")
    print("=" * 100)

    projects = sorted(
        {
            record["project"]
            for record in records
        }
    )

    for project in projects:

        subset = [
            record
            for record in records
            if (
                record[
                    "project"
                ]
                == project
            )
        ]

        hits = sum(
            (
                record[
                    "best_gt_rank"
                ]
                == 1
            )
            for record in subset
        )

        print(
            project,
            f"{hits}/{len(subset)}",
            f"= {hits / len(subset):.4f}",
        )


if __name__ == "__main__":
    main()