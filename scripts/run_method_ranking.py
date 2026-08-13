import json
from pathlib import Path

from camd.context.method_extractor import (
    extract_java_methods,
)
from camd.detectors.method_ranker import (
    MethodRanker,
)
from camd.llm.client import OpenAIClient


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1b"
    / "src"
    / "main"
    / "java"
    / "org"
    / "apache"
    / "commons"
    / "lang3"
    / "math"
    / "NumberUtils.java"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "method_ranking_Lang_1.jsonl"
)

GROUND_TRUTH_METHOD = (
    "createNumber"
)


def save_results(
    results,
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for rank, result in enumerate(
            results,
            start=1,
        ):

            record = {
                "rank": rank,
                "method": result.method_name,
                "start_line": (
                    result.start_line
                ),
                "end_line": (
                    result.end_line
                ),
                "is_suspicious": (
                    result.is_suspicious
                ),
                "suspicion_score": (
                    result.suspicion_score
                ),
                "defect_type": (
                    result.defect_type
                ),
                "reason": result.reason,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")


def find_ground_truth_rank(
    results,
    method_name: str,
):

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if (
            result.method_name
            == method_name
        ):
            return rank

    return None

def reciprocal_rank(
    rank,
) -> float:

    if rank is None:
        return 0.0

    return 1.0 / rank

def print_top_results(
    results,
    top_k: int = 10,
) -> None:

    print()
    print("=" * 80)
    print(
        f"Top-{min(top_k, len(results))} "
        f"Suspicious Methods"
    )
    print("=" * 80)

    for rank, result in enumerate(
        results[:top_k],
        start=1,
    ):

        print(
            f"#{rank:<3} "
            f"{result.method_name:<25} "
            f"{result.start_line:>4}-"
            f"{result.end_line:<4} "
            f"score="
            f"{result.suspicion_score:.2f}"
        )

        print(
            f"     suspicious="
            f"{result.is_suspicious}"
        )

        print(
            f"     type="
            f"{result.defect_type}"
        )

        print(
            f"     reason="
            f"{result.reason}"
        )

        print()


def print_evaluation(
    results,
) -> None:

    rank = find_ground_truth_rank(
        results=results,
        method_name=GROUND_TRUTH_METHOD,
    )

    print()
    print("=" * 80)
    print("CAMD Method Localization Evaluation")
    print("=" * 80)

    print(
        f"Ground-truth method: "
        f"{GROUND_TRUTH_METHOD}"
    )

    if rank is None:

        print(
            "Ground-truth method was not "
            "ranked."
        )

        return

    print(
        f"Ground-truth rank: {rank}"
    )
    print(
        f"Reciprocal rank: "
        f"{reciprocal_rank(rank):.4f}"
    )

    print(
        f"Top-1 localization: "
        f"{rank <= 1}"
    )

    print(
        f"Top-3 localization: "
        f"{rank <= 3}"
    )

    print(
        f"Top-5 localization: "
        f"{rank <= 5}"
    )

    print(
        f"Top-10 localization: "
        f"{rank <= 10}"
    )

    print("=" * 80)


def main():

    print("=" * 80)
    print(
        "CAMD - Method-Level "
        "Defect Ranking"
    )
    print("=" * 80)

    print("\nProject: Lang")
    print("Bug ID: 1")
    print("Version: buggy")

    print(
        f"\nSource file: "
        f"{SOURCE_FILE.name}"
    )

    methods = extract_java_methods(
        SOURCE_FILE
    )

    print(
        f"Extracted methods: "
        f"{len(methods)}"
    )

    client = OpenAIClient()

    ranker = MethodRanker(
        llm_client=client
    )

    print()
    print(
        "Running method-level "
        "defect analysis..."
    )
    print()

    results = ranker.rank_methods(
        methods
    )

    save_results(
        results
    )

    print_top_results(
        results=results,
        top_k=10,
    )

    print_evaluation(
        results
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()