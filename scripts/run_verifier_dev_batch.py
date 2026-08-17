from __future__ import annotations

import argparse
import json
from pathlib import Path

from camd.llm.client import OpenAIClient

from camd.verification.critic import (
    ProgramWideCritic,
    ProgramWideCriticResult,
)

from camd.verification.detector import (
    ProgramWideDetectorResult,
)

from camd.verification.frozen_candidate_loader import (
    FrozenCandidateLoader,
)

from camd.verification.judge import (
    ProgramWideJudge,
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


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "verifier_dev"
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


def build_detector_result(
    item,
) -> ProgramWideDetectorResult:

    return ProgramWideDetectorResult(
        benchmark_id=(
            item["benchmark_id"]
        ),
        pool_position=int(
            item["pool_position"]
        ),
        class_name=(
            item["class_name"]
        ),
        method_name=(
            item["method_name"]
        ),
        source_file=(
            item["source_file"]
        ),
        start_line=int(
            item["start_line"]
        ),
        end_line=int(
            item["end_line"]
        ),
        hypothesis=(
            item["hypothesis"]
        ),
        supporting_evidence=tuple(
            item.get(
                "supporting_evidence",
                [],
            )
        ),
        target_defect_probability=float(
            item[
                "target_defect_probability"
            ]
        ),
        base_rank=(
            int(item["base_rank"])
            if item.get("base_rank")
            is not None
            else None
        ),
        base_score=(
            float(item["base_score"])
            if item.get("base_score")
            is not None
            else None
        ),
        from_base=bool(
            item.get(
                "from_base",
                False,
            )
        ),
        from_stack=bool(
            item.get(
                "from_stack",
                False,
            )
        ),
        from_call=bool(
            item.get(
                "from_call",
                False,
            )
        ),
        stack_depth=(
            int(item["stack_depth"])
            if item.get("stack_depth")
            is not None
            else None
        ),
        call_depth=(
            int(item["call_depth"])
            if item.get("call_depth")
            is not None
            else None
        ),
    )


def build_critic_result(
    item,
) -> ProgramWideCriticResult:

    return ProgramWideCriticResult(
        benchmark_id=(
            item["benchmark_id"]
        ),
        pool_position=int(
            item["pool_position"]
        ),
        class_name=(
            item["class_name"]
        ),
        method_name=(
            item["method_name"]
        ),
        source_file=(
            item["source_file"]
        ),
        start_line=int(
            item["start_line"]
        ),
        end_line=int(
            item["end_line"]
        ),
        detector_probability=float(
            item[
                "detector_probability"
            ]
        ),
        detector_hypothesis=(
            item[
                "detector_hypothesis"
            ]
        ),
        agrees_with_detector=bool(
            item[
                "agrees_with_detector"
            ]
        ),
        weaknesses=tuple(
            item.get(
                "weaknesses",
                [],
            )
        ),
        alternative_explanation=(
            item.get(
                "alternative_explanation",
                "",
            )
        ),
        critic_probability=float(
            item[
                "critic_probability"
            ]
        ),
    )


def find_candidate(
    case,
    pool_position: int,
):

    for candidate in case.candidates:

        if (
            candidate.pool_position
            == pool_position
        ):
            return candidate

    raise KeyError(
        "Candidate pool position "
        f"{pool_position} not found "
        f"for {case.benchmark_id}."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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

    critic = ProgramWideCritic(
        llm_client=client,
        include_retrieval_evidence=True,
    )

    judge = ProgramWideJudge(
        llm_client=client,
        include_retrieval_evidence=True,
    )

    total_critic_calls = 0
    total_judge_calls = 0

    print("=" * 100)
    print("CAMD Critic -> Judge Dev Batch")
    print("=" * 100)

    print(
        "Cases:",
        len(benchmark_ids),
    )

    print(
        "Retrieval budget:",
        args.budget,
    )

    print(
        "Verifier Top-K:",
        args.verify_top_k,
    )

    for case_index, benchmark_id in enumerate(
        benchmark_ids,
        start=1,
    ):

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

        if not detector_path.exists():
            raise FileNotFoundError(
                "Missing Detector result: "
                f"{detector_path}"
            )

        detector_data = (
            load_json(
                detector_path
            )
        )

        detector_ranking = (
            detector_data[
                "ranking"
            ][
                :args.verify_top_k
            ]
        )

        output_path = (
            args.output_dir
            / (
                f"{benchmark_id}"
                f"_budget{args.budget}"
                f"_top{args.verify_top_k}.json"
            )
        )

        if output_path.exists():
            output_data = (
                load_json(
                    output_path
                )
            )
        else:
            output_data = {
                "benchmark_id": (
                    benchmark_id
                ),
                "project": (
                    case.project
                ),
                "bug_id": (
                    case.bug_id
                ),
                "budget": (
                    args.budget
                ),
                "verify_top_k": (
                    args.verify_top_k
                ),
                "candidate_recall": (
                    case.candidate_recall
                ),
                "detector_candidate_count": (
                    len(
                        detector_data[
                            "ranking"
                        ]
                    )
                ),
                "shortlist_count": (
                    len(
                        detector_ranking
                    )
                ),
                "model": (
                    getattr(
                        client,
                        "model",
                        None,
                    )
                ),
                "results": [],
            }

        existing_by_pool = {
            int(
                item[
                    "pool_position"
                ]
            ): item
            for item
            in output_data.get(
                "results",
                [],
            )
        }

        print()
        print("=" * 100)
        print(
            f"[{case_index}/"
            f"{len(benchmark_ids)}] "
            f"{benchmark_id}"
        )
        print("=" * 100)

        for detector_item in (
            detector_ranking
        ):

            pool_position = int(
                detector_item[
                    "pool_position"
                ]
            )

            candidate = find_candidate(
                case,
                pool_position,
            )

            detector_result = (
                build_detector_result(
                    detector_item
                )
            )

            existing = (
                existing_by_pool.get(
                    pool_position
                )
            )

            # =============================================
            # Critic
            # =============================================

            if (
                existing
                and existing.get(
                    "critic"
                )
            ):

                critic_result = (
                    build_critic_result(
                        existing[
                            "critic"
                        ]
                    )
                )

                print(
                    f"pool={pool_position} "
                    "Critic cached"
                )

            else:

                print(
                    f"pool={pool_position} "
                    f"Critic running: "
                    f"{candidate.class_name}."
                    f"{candidate.method_name}"
                )

                critic_result = (
                    critic.analyze_candidate(
                        case=case,
                        candidate=candidate,
                        detector_result=(
                            detector_result
                        ),
                    )
                )

                total_critic_calls += 1

                if existing is None:
                    existing = {
                        "pool_position": (
                            pool_position
                        ),
                        "detector_rank": (
                            detector_item[
                                "detector_rank"
                            ]
                        ),
                        "candidate": {
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
                        },
                        "detector": (
                            detector_item
                        ),
                        "critic": None,
                        "judge": None,
                    }

                    output_data[
                        "results"
                    ].append(
                        existing
                    )

                    existing_by_pool[
                        pool_position
                    ] = existing

                existing[
                    "critic"
                ] = (
                    critic_result.to_dict()
                )

                save_json(
                    output_path,
                    output_data,
                )

            # =============================================
            # Judge
            # =============================================

            if (
                existing
                and existing.get(
                    "judge"
                )
            ):

                print(
                    f"pool={pool_position} "
                    "Judge cached"
                )

                continue

            print(
                f"pool={pool_position} "
                "Judge running"
            )

            judge_result = (
                judge.analyze_candidate(
                    case=case,
                    candidate=candidate,
                    detector_result=(
                        detector_result
                    ),
                    critic_result=(
                        critic_result
                    ),
                )
            )

            total_judge_calls += 1

            existing[
                "judge"
            ] = (
                judge_result.to_dict()
            )

            save_json(
                output_path,
                output_data,
            )

            print(
                "  D=",
                f"{detector_result.target_defect_probability:.4f}",
                "C=",
                f"{critic_result.critic_probability:.4f}",
                "J=",
                f"{judge_result.judge_probability:.4f}",
            )

        # =============================================
        # Final Judge ranking
        # =============================================

        complete_rows = [
            item
            for item
            in output_data[
                "results"
            ]
            if (
                item.get("critic")
                and item.get("judge")
            )
        ]

        complete_rows.sort(
            key=lambda item: (
                -float(
                    item[
                        "judge"
                    ][
                        "judge_probability"
                    ]
                ),
                int(
                    item[
                        "detector_rank"
                    ]
                ),
                int(
                    item[
                        "pool_position"
                    ]
                ),
            )
        )

        judge_ranking = []

        for rank, item in enumerate(
            complete_rows,
            start=1,
        ):

            row = {
                "judge_rank": (
                    rank
                ),
                "detector_rank": (
                    item[
                        "detector_rank"
                    ]
                ),
                "pool_position": (
                    item[
                        "pool_position"
                    ]
                ),
                **item[
                    "candidate"
                ],
                "detector_probability": (
                    item[
                        "detector"
                    ][
                        "target_defect_probability"
                    ]
                ),
                "critic_probability": (
                    item[
                        "critic"
                    ][
                        "critic_probability"
                    ]
                ),
                "judge_probability": (
                    item[
                        "judge"
                    ][
                        "judge_probability"
                    ]
                ),
                "is_target_defect": (
                    item[
                        "judge"
                    ][
                        "is_target_defect"
                    ]
                ),
                "defect_type": (
                    item[
                        "judge"
                    ][
                        "defect_type"
                    ]
                ),
            }

            judge_ranking.append(
                row
            )

        output_data[
            "judge_ranking"
        ] = judge_ranking

        output_data[
            "completed"
        ] = (
            len(complete_rows)
            == len(
                detector_ranking
            )
        )

        save_json(
            output_path,
            output_data,
        )

        print(
            "Completed:",
            output_data[
                "completed"
            ],
        )

        print(
            "Saved:",
            output_path,
        )

    print()
    print("=" * 100)
    print("Verifier Dev Batch Complete")
    print("=" * 100)

    print(
        "New Critic calls:",
        total_critic_calls,
    )

    print(
        "New Judge calls:",
        total_judge_calls,
    )


if __name__ == "__main__":
    main()