from __future__ import annotations

import argparse
import json
from pathlib import Path

from camd.llm.client import OpenAIClient
from camd.verification.detector import ProgramWideDetector
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

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "detector"
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


def result_matches_candidate(
    result,
    candidate,
) -> bool:

    return (
        int(result["pool_position"])
        == candidate.pool_position
        and result["class_name"]
        == candidate.class_name
        and result["source_file"]
        == candidate.source_file
        and int(result["start_line"])
        == candidate.start_line
        and int(result["end_line"])
        == candidate.end_line
    )


def make_payload(
    case,
    results,
):

    results = sorted(
        results,
        key=lambda item: (
            item["pool_position"]
        ),
    )

    ranking = [
        dict(item)
        for item in results
    ]

    ranking.sort(
        key=lambda item: (
            -item[
                "target_defect_probability"
            ],
            item["pool_position"],
        )
    )

    for rank, item in enumerate(
        ranking,
        start=1,
    ):
        item["detector_rank"] = rank

    return {
        "benchmark_id": (
            case.benchmark_id
        ),
        "project": (
            case.project
        ),
        "bug_id": (
            case.bug_id
        ),
        "budget": (
            case.budget
        ),
        "candidate_count": (
            len(case.candidates)
        ),
        "candidate_recall": (
            case.candidate_recall
        ),
        "include_retrieval_evidence": True,
        "completed": (
            len(results)
            == len(case.candidates)
        ),
        "results": (
            results
        ),
        "ranking": (
            ranking
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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

    benchmark_ids = (
        loader.benchmark_ids(
            only_successful=True,
            only_method_applicable=True,
        )
    )

    client = OpenAIClient()

    detector = ProgramWideDetector(
        llm_client=client,
        include_retrieval_evidence=True,
    )

    total_new_calls = 0

    print("=" * 100)
    print("CAMD Detector Dev Batch")
    print("=" * 100)

    print(
        "Cases:",
        len(benchmark_ids),
    )

    print(
        "Budget:",
        args.budget,
    )

    for case_index, benchmark_id in enumerate(
        benchmark_ids,
        start=1,
    ):

        case = loader.load_case(
            benchmark_id,
            budget=args.budget,
        )

        output_path = (
            args.output_dir
            / (
                f"{benchmark_id}"
                f"_budget{args.budget}.json"
            )
        )

        existing_results = []

        if output_path.exists():

            existing_data = (
                load_json(
                    output_path
                )
            )

            existing_results = (
                existing_data.get(
                    "results",
                    [],
                )
            )

        valid_existing = {}

        for result in existing_results:

            pool_position = (
                int(
                    result[
                        "pool_position"
                    ]
                )
            )

            candidate = next(
                (
                    item
                    for item
                    in case.candidates
                    if (
                        item.pool_position
                        == pool_position
                    )
                ),
                None,
            )

            if candidate is None:
                continue

            if result_matches_candidate(
                result,
                candidate,
            ):
                valid_existing[
                    pool_position
                ] = result

        print()
        print("=" * 100)
        print(
            f"[{case_index}/"
            f"{len(benchmark_ids)}] "
            f"{benchmark_id}"
        )
        print("=" * 100)

        print(
            "Candidates:",
            len(case.candidates),
        )

        print(
            "Existing results:",
            len(valid_existing),
        )

        if (
            len(valid_existing)
            == len(case.candidates)
        ):

            print(
                "Complete. Skipping."
            )
            continue

        results = list(
            valid_existing.values()
        )

        for candidate in (
            case.candidates
        ):

            if (
                candidate.pool_position
                in valid_existing
            ):
                continue

            print(
                f"Running pool "
                f"{candidate.pool_position}: "
                f"{candidate.class_name}."
                f"{candidate.method_name}"
                f"[{candidate.start_line}-"
                f"{candidate.end_line}]"
            )

            result = (
                detector.analyze_candidate(
                    case,
                    candidate,
                )
            )

            result_dict = (
                result.to_dict()
            )

            results.append(
                result_dict
            )

            total_new_calls += 1

            payload = (
                make_payload(
                    case,
                    results,
                )
            )

            save_json(
                output_path,
                payload,
            )

            print(
                "  p =",
                f"{result.target_defect_probability:.4f}",
            )

        payload = (
            make_payload(
                case,
                results,
            )
        )

        save_json(
            output_path,
            payload,
        )

        print(
            "Completed:",
            output_path,
        )

    print()
    print("=" * 100)
    print("Detector Dev Batch Complete")
    print("=" * 100)

    print(
        "New LLM calls:",
        total_new_calls,
    )


if __name__ == "__main__":
    main()