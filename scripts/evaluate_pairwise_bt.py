from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Pairwise evaluation JSON produced by evaluate_pairwise_qlora.py",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--l2",
        type=float,
        default=1e-3,
    )

    return parser.parse_args()


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def tie_aware_ranks(
    candidate_scores: dict[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    sorted_items = sorted(
        candidate_scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    ranks: dict[str, float] = {}

    index = 0

    while index < len(sorted_items):
        score = sorted_items[index][1]
        end = index + 1

        while end < len(sorted_items):
            other_score = sorted_items[end][1]

            if abs(other_score - score) > tolerance:
                break

            end += 1

        first_rank = index + 1
        last_rank = end

        average_rank = (
            first_rank + last_rank
        ) / 2.0

        for tie_index in range(
            index,
            end,
        ):
            identity = sorted_items[
                tie_index
            ][0]

            ranks[identity] = average_rank

        index = end

    return ranks


def fit_bradley_terry(
    pair_results: list[dict[str, Any]],
    *,
    max_iters: int,
    learning_rate: float,
    l2: float,
) -> dict[str, float]:
    candidates = set()

    for pair in pair_results:
        candidates.add(
            pair["candidate_a"]["identity"]
        )
        candidates.add(
            pair["candidate_b"]["identity"]
        )

    strengths = {
        candidate: 0.0
        for candidate in candidates
    }

    for _ in range(max_iters):
        gradients = {
            candidate: 0.0
            for candidate in candidates
        }

        for pair in pair_results:
            a = pair["candidate_a"]["identity"]
            b = pair["candidate_b"]["identity"]

            p_a_observed = float(
                pair["p_a"]
            )

            score_diff = (
                strengths[a]
                - strengths[b]
            )

            p_a_model = sigmoid(
                score_diff
            )

            error = (
                p_a_observed
                - p_a_model
            )

            gradients[a] += error
            gradients[b] -= error

        for candidate in candidates:
            gradients[candidate] -= (
                l2
                * strengths[candidate]
            )

            strengths[candidate] += (
                learning_rate
                * gradients[candidate]
            )

        # Identifiability:
        # center scores around zero.
        mean_strength = (
            sum(strengths.values())
            / len(strengths)
        )

        for candidate in candidates:
            strengths[candidate] -= (
                mean_strength
            )

    return strengths


def main() -> None:
    args = parse_args()

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    pair_results = data[
        "pair_results"
    ]

    by_bug: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    candidate_info: dict[
        tuple[str, int],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    for pair in pair_results:
        key = (
            pair["project"],
            int(pair["bug_id"]),
        )

        by_bug[key].append(pair)

        candidate_a = pair[
            "candidate_a"
        ]
        candidate_b = pair[
            "candidate_b"
        ]

        candidate_info[
            key
        ][
            candidate_a["identity"]
        ] = candidate_a

        candidate_info[
            key
        ][
            candidate_b["identity"]
        ] = candidate_b

    bug_results = []

    reciprocal_ranks = []

    top1 = 0
    top3 = 0
    top5 = 0

    for key in sorted(by_bug):
        project, bug_id = key

        strengths = fit_bradley_terry(
            by_bug[key],
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )

        ranks = tie_aware_ranks(
            strengths
        )

        targets = [
            identity
            for identity, info
            in candidate_info[
                key
            ].items()
            if info[
                "is_target_defect"
            ]
        ]

        target_ranks = [
            ranks[identity]
            for identity in targets
            if identity in ranks
        ]

        if not target_ranks:
            continue

        best_target_rank = min(
            target_ranks
        )

        reciprocal_rank = (
            1.0 / best_target_rank
        )

        reciprocal_ranks.append(
            reciprocal_rank
        )

        if best_target_rank <= 1:
            top1 += 1

        if best_target_rank <= 3:
            top3 += 1

        if best_target_rank <= 5:
            top5 += 1

        ranking = []

        for identity in sorted(
            strengths,
            key=lambda x: (
                ranks[x],
                x,
            ),
        ):
            info = candidate_info[
                key
            ][identity]

            ranking.append(
                {
                    "identity": identity,
                    "score": strengths[
                        identity
                    ],
                    "rank": ranks[
                        identity
                    ],
                    "is_target_defect": bool(
                        info[
                            "is_target_defect"
                        ]
                    ),
                    "class_name": info[
                        "class_name"
                    ],
                    "method_name": info[
                        "method_name"
                    ],
                    "start_line": info[
                        "start_line"
                    ],
                    "end_line": info[
                        "end_line"
                    ],
                }
            )

        bug_results.append(
            {
                "project": project,
                "bug_id": bug_id,
                "candidate_count": len(
                    strengths
                ),
                "target_count": len(
                    targets
                ),
                "best_target_rank": (
                    best_target_rank
                ),
                "reciprocal_rank": (
                    reciprocal_rank
                ),
                "ranking": ranking,
            }
        )

    bug_count = len(
        reciprocal_ranks
    )

    metrics = {
        "bugs": bug_count,
        "mrr": (
            sum(reciprocal_ranks)
            / bug_count
            if bug_count
            else 0.0
        ),
        "top1": (
            top1 / bug_count
            if bug_count
            else 0.0
        ),
        "top3": (
            top3 / bug_count
            if bug_count
            else 0.0
        ),
        "top5": (
            top5 / bug_count
            if bug_count
            else 0.0
        ),
    }

    output = {
        "input": str(
            args.input
        ),
        "aggregation": (
            "Bradley-Terry"
        ),
        "configuration": {
            "max_iters": (
                args.max_iters
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "l2": args.l2,
        },
        "metrics": metrics,
        "bug_results": (
            bug_results
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print(
        "Bradley-Terry Pairwise Reranking"
    )
    print("=" * 100)

    print()
    print(
        f"Bugs: {metrics['bugs']}"
    )
    print(
        f"MRR: {metrics['mrr']:.4f}"
    )
    print(
        f"Top-1: {metrics['top1']:.4f}"
    )
    print(
        f"Top-3: {metrics['top3']:.4f}"
    )
    print(
        f"Top-5: {metrics['top5']:.4f}"
    )

    print()
    print("Saved:")
    print(args.output)


if __name__ == "__main__":
    main()