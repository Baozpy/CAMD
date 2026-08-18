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


DEFAULT_DETECTOR_DIR = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "detector"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "expansion_signal_analysis.json"
)


def load_json(path: Path):

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


def gt_key(gt):

    return (
        gt.class_name,
        gt.source_file,
        gt.start_line,
        gt.end_line,
    )


def candidate_key(candidate):

    return (
        candidate.class_name,
        candidate.source_file,
        candidate.start_line,
        candidate.end_line,
    )


def has_gt(case) -> bool:

    gt_keys = {
        gt_key(gt)
        for gt in case.ground_truth
    }

    candidate_keys = {
        candidate_key(candidate)
        for candidate in case.candidates
    }

    return bool(
        gt_keys & candidate_keys
    )


def summarize(values):

    if not values:
        return None

    ordered = sorted(
        values
    )

    n = len(
        ordered
    )

    if n % 2:

        median = (
            ordered[
                n // 2
            ]
        )

    else:

        median = (
            ordered[
                n // 2 - 1
            ]
            + ordered[
                n // 2
            ]
        ) / 2

    return {
        "n": n,
        "mean": (
            sum(values) / n
        ),
        "median": (
            median
        ),
        "min": (
            min(values)
        ),
        "max": (
            max(values)
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
        "--detector-dir",
        type=Path,
        default=DEFAULT_DETECTOR_DIR,
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

    groups = defaultdict(
        list
    )

    records = []

    for benchmark_id in benchmark_ids:

        case10 = loader.load_case(
            benchmark_id,
            budget=10,
        )

        case50 = loader.load_case(
            benchmark_id,
            budget=50,
        )

        case100 = loader.load_case(
            benchmark_id,
            budget=100,
        )

        recall10 = has_gt(
            case10
        )

        recall50 = has_gt(
            case50
        )

        recall100 = has_gt(
            case100
        )

        if recall10:

            group = (
                "retrieved_at_10"
            )

        elif recall50:

            group = (
                "recovered_at_50"
            )

        elif recall100:

            group = (
                "recovered_at_100"
            )

        else:

            group = (
                "never_recovered"
            )

        detector_path = (
            args.detector_dir
            / f"{benchmark_id}_budget10.json"
        )

        if not detector_path.exists():

            raise FileNotFoundError(
                detector_path
            )

        detector_data = load_json(
            detector_path
        )

        ranking = (
            detector_data[
                "ranking"
            ]
        )

        if not ranking:

            continue

        probabilities = [
            float(
                item[
                    "target_defect_probability"
                ]
            )
            for item in ranking
        ]

        p1 = (
            probabilities[0]
        )

        p2 = (
            probabilities[1]
            if len(
                probabilities
            ) >= 2
            else 0.0
        )

        margin = (
            p1 - p2
        )

        top3 = (
            probabilities[:3]
        )

        top3_mean = (
            sum(top3)
            / len(top3)
        )

        high_probability_count = sum(
            probability >= 0.5
            for probability
            in probabilities
        )

        record = {
            "benchmark_id": (
                benchmark_id
            ),
            "project": (
                case10.project
            ),
            "group": (
                group
            ),
            "p1": (
                p1
            ),
            "p2": (
                p2
            ),
            "margin": (
                margin
            ),
            "top3_mean": (
                top3_mean
            ),
            "high_probability_count": (
                high_probability_count
            ),
            "candidate_count": (
                len(
                    ranking
                )
            ),
        }

        records.append(
            record
        )

        groups[
            group
        ].append(
            record
        )

    print("=" * 100)
    print(
        "CAMD Expansion Signal Analysis"
    )
    print("=" * 100)

    group_order = [
        "retrieved_at_10",
        "recovered_at_50",
        "recovered_at_100",
        "never_recovered",
    ]

    summary = {}

    for group in group_order:

        subset = (
            groups[
                group
            ]
        )

        print()
        print("=" * 100)
        print(group)
        print("=" * 100)

        print(
            "Cases:",
            len(
                subset
            ),
        )

        p1_summary = summarize(
            [
                item["p1"]
                for item in subset
            ]
        )

        margin_summary = summarize(
            [
                item["margin"]
                for item in subset
            ]
        )

        top3_summary = summarize(
            [
                item["top3_mean"]
                for item in subset
            ]
        )

        high_count_summary = summarize(
            [
                item[
                    "high_probability_count"
                ]
                for item in subset
            ]
        )

        summary[
            group
        ] = {
            "p1": (
                p1_summary
            ),
            "margin": (
                margin_summary
            ),
            "top3_mean": (
                top3_summary
            ),
            "high_probability_count": (
                high_count_summary
            ),
        }

        print(
            "Top-1 probability:",
            p1_summary,
        )

        print(
            "Top1-Top2 margin:",
            margin_summary,
        )

        print(
            "Top-3 mean:",
            top3_summary,
        )

        print(
            "Candidates p>=0.5:",
            high_count_summary,
        )

    print()
    print("=" * 100)
    print(
        "Individual Retrieval Misses"
    )
    print("=" * 100)

    misses = [
        record
        for record in records
        if record[
            "group"
        ]
        != "retrieved_at_10"
    ]

    misses.sort(
        key=lambda item: (
            item[
                "group"
            ],
            item[
                "p1"
            ],
        )
    )

    for record in misses:

        print(
            f"{record['benchmark_id']:<12}",
            f"{record['group']:<20}",
            f"p1={record['p1']:.4f}",
            f"p2={record['p2']:.4f}",
            f"margin={record['margin']:.4f}",
            f"high>=0.5={record['high_probability_count']}",
        )

    report = {
        "summary": (
            summary
        ),
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