from __future__ import annotations

import argparse
import json
from pathlib import Path

from camd.llm.client import OpenAIClient
from camd.verification.detector import (
    ProgramWideDetector,
)
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


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "detector"
)


def save_json(
    path: Path,
    data,
) -> None:

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


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

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
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--include-retrieval-evidence",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    loader = FrozenCandidateLoader(
        args.manifest
    )

    case = loader.load_case(
        args.benchmark_id,
        budget=args.budget,
    )

    client = OpenAIClient()

    detector = ProgramWideDetector(
        llm_client=client,
        include_retrieval_evidence=(
            args.include_retrieval_evidence
        ),
    )

    output_path = (
        args.output
        if args.output is not None
        else (
            DEFAULT_OUTPUT_DIR
            / (
                f"{case.benchmark_id}"
                f"_budget{case.budget}.json"
            )
        )
    )

    print()
    print("=" * 100)
    print(
        "Frozen Program-Wide Detector"
    )
    print("=" * 100)

    print(
        f"Benchmark: "
        f"{case.benchmark_id}"
    )

    print(
        f"Budget: "
        f"{case.budget}"
    )

    print(
        f"Candidate count: "
        f"{len(case.candidates)}"
    )

    print(
        f"Candidate recall: "
        f"{case.candidate_recall}"
    )

    print(
        "Retrieval evidence: "
        f"{args.include_retrieval_evidence}"
    )

    results = []

    for index, candidate in enumerate(
        case.candidates,
        start=1,
    ):

        print()
        print("-" * 100)

        print(
            f"[{index}/"
            f"{len(case.candidates)}] "
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

        print(
            "Probability: "
            f"{result.target_defect_probability:.4f}"
        )

        print(
            "Hypothesis: "
            f"{result.hypothesis}"
        )

        payload = {
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
            "include_retrieval_evidence": (
                args.include_retrieval_evidence
            ),
            "results": (
                results
            ),
        }

        save_json(
            output_path,
            payload,
        )

    # ---------------------------------------------------------
    # Sort by Detector score
    # ---------------------------------------------------------

    ranked = sorted(
        results,
        key=lambda item: (
            -item[
                "target_defect_probability"
            ],
            item[
                "pool_position"
            ],
        ),
    )

    for rank, item in enumerate(
        ranked,
        start=1,
    ):

        item[
            "detector_rank"
        ] = rank

    payload = {
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
        "include_retrieval_evidence": (
            args.include_retrieval_evidence
        ),
        "results": (
            results
        ),
        "ranking": (
            ranked
        ),
    }

    save_json(
        output_path,
        payload,
    )

    print()
    print("=" * 100)
    print(
        "Detector Ranking"
    )
    print("=" * 100)

    for item in ranked:

        print(
            f"#{item['detector_rank']:<3} "
            f"p="
            f"{item['target_defect_probability']:.4f} "
            f"{item['class_name']}."
            f"{item['method_name']}"
            f"[{item['start_line']}-"
            f"{item['end_line']}] "
            f"(pool="
            f"{item['pool_position']})"
        )

    print()
    print(
        f"Saved:\n{output_path}"
    )


if __name__ == "__main__":
    main()