from __future__ import annotations

import argparse
import json
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
    / "detector"
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark-id",
        required=True,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--detector-result",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def candidate_key(
    item,
):
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

    case = loader.load_case(
        args.benchmark_id,
        budget=args.budget,
    )

    result_path = (
        args.detector_result
        if args.detector_result is not None
        else (
            DEFAULT_DETECTOR_DIR
            / (
                f"{args.benchmark_id}"
                f"_budget{args.budget}.json"
            )
        )
    )

    with result_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    gt_keys = {
        (
            gt.class_name,
            gt.source_file,
            gt.start_line,
            gt.end_line,
        )
        for gt in case.ground_truth
    }

    ranking = data["ranking"]

    gt_hits = []

    print("=" * 100)
    print("Detector Case Evaluation")
    print("=" * 100)

    print(
        "benchmark:",
        case.benchmark_id,
    )

    print(
        "budget:",
        case.budget,
    )

    print(
        "candidate count:",
        len(case.candidates),
    )

    print(
        "candidate recall:",
        case.candidate_recall,
    )

    print()

    print("Ground truth methods:")

    for gt in case.ground_truth:
        print(
            " ",
            gt.class_name,
            gt.method_name,
            f"[{gt.start_line}-{gt.end_line}]",
        )

    print()
    print("Ground-truth positions in Detector ranking:")

    for item in ranking:

        key = candidate_key(
            item
        )

        if key not in gt_keys:
            continue

        gt_hits.append(
            item
        )

        print(
            f" rank={item['detector_rank']}",
            f"p={item['target_defect_probability']:.4f}",
            (
                f"{item['class_name']}."
                f"{item['method_name']}"
                f"[{item['start_line']}-"
                f"{item['end_line']}]"
            ),
            f"pool={item['pool_position']}",
        )

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)

    if not case.candidate_recall:

        print(
            "GT is not present in the frozen candidate pool."
        )

        print(
            "Detector success is impossible for this case."
        )

        return

    if not gt_hits:

        print(
            "WARNING: frozen manifest says recall=True, "
            "but no GT candidate was found in Detector output."
        )

        return

    best_gt_rank = min(
        item["detector_rank"]
        for item in gt_hits
    )

    best_gt_probability = max(
        item["target_defect_probability"]
        for item in gt_hits
    )

    top1 = (
        best_gt_rank == 1
    )

    top3 = (
        best_gt_rank <= 3
    )

    top5 = (
        best_gt_rank <= 5
    )

    print(
        "Best GT Detector rank:",
        best_gt_rank,
    )

    print(
        "Best GT probability:",
        f"{best_gt_probability:.4f}",
    )

    print(
        "Detector Top-1:",
        top1,
    )

    print(
        "Detector Top-3:",
        top3,
    )

    print(
        "Detector Top-5:",
        top5,
    )


if __name__ == "__main__":
    main()