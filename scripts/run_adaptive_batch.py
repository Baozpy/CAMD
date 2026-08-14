import json
from pathlib import Path

from camd.evaluation.adaptive_candidate_runner import (
    AdaptiveCandidateRunner,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "defects4j"
)

PROJECT = "Lang"

THRESHOLD = 0.5

EXPANSION_STAGES = [
    (5, 10),
    (10, 20),
]


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    path: Path,
    data: dict,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_jsonl(
    path: Path,
) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def judge_score(
    record: dict,
) -> float:

    judge = record.get(
        "judge"
    )

    if not isinstance(
        judge,
        dict,
    ):
        return 0.0

    return float(
        judge.get(
            "target_defect_probability",
            0.0,
        )
    )


def get_best_saved_record(
    bug_id: int,
) -> tuple[
    dict | None,
    str | None,
]:

    bug_dir = (
        RESULTS_ROOT
        / f"{PROJECT}_{bug_id}"
    )

    candidates = [
        bug_dir
        / "multi_agent_adaptive.jsonl",

        bug_dir
        / "multi_agent_expanded_test.jsonl",

        bug_dir
        / "multi_agent.jsonl",
    ]

    for path in candidates:

        if not path.exists():
            continue

        records = (
            load_jsonl(
                path
            )
        )

        if not records:
            continue

        ranked = sorted(
            records,
            key=judge_score,
            reverse=True,
        )

        return (
            ranked[0],
            path.name,
        )

    return None, None


def method_matches_ground_truth(
    record: dict,
    ground_truth_methods: list[dict],
) -> bool:

    method_name = (
        record.get(
            "method"
        )
    )

    start_line = (
        record.get(
            "start_line"
        )
    )

    end_line = (
        record.get(
            "end_line"
        )
    )

    for gt in ground_truth_methods:

        if (
            gt.get("name")
            != method_name
        ):
            continue

        gt_start = (
            gt.get(
                "start_line"
            )
        )

        gt_end = (
            gt.get(
                "end_line"
            )
        )

        if (
            gt_start is None
            or gt_end is None
            or start_line is None
            or end_line is None
        ):
            return True

        overlap = not (
            end_line < gt_start
            or start_line > gt_end
        )

        if overlap:
            return True

    return False


def build_metrics(
    rank: int | None,
) -> dict:

    if rank is None:

        return {
            "rank": None,
            "rr": 0.0,
            "top_1": False,
            "top_3": False,
            "top_5": False,
            "top_10": False,
            "top_20": False,
        }

    return {
        "rank": rank,

        "rr": (
            1.0 / rank
        ),

        "top_1": (
            rank <= 1
        ),

        "top_3": (
            rank <= 3
        ),

        "top_5": (
            rank <= 5
        ),

        "top_10": (
            rank <= 10
        ),

        "top_20": (
            rank <= 20
        ),
    }


def find_gt_rank(
    records: list[dict],
    ground_truth_methods: list[dict],
) -> int | None:

    ranked = sorted(
        records,
        key=judge_score,
        reverse=True,
    )

    for rank, record in enumerate(
        ranked,
        start=1,
    ):

        if method_matches_ground_truth(
            record,
            ground_truth_methods,
        ):
            return rank

    return None


def aggregate_metrics(
    per_bug: list[dict],
) -> dict:

    total = len(per_bug)

    if total == 0:

        return {
            "evaluated_bugs": 0,
            "mrr": 0.0,
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "top_10_accuracy": 0.0,
            "top_20_accuracy": 0.0,
        }

    metrics = [
        item[
            "adaptive_metrics"
        ]
        for item
        in per_bug
    ]

    return {
        "evaluated_bugs": total,

        "mrr": (
            sum(
                item["rr"]
                for item in metrics
            )
            / total
        ),

        "top_1_accuracy": (
            sum(
                item["top_1"]
                for item in metrics
            )
            / total
        ),

        "top_3_accuracy": (
            sum(
                item["top_3"]
                for item in metrics
            )
            / total
        ),

        "top_5_accuracy": (
            sum(
                item["top_5"]
                for item in metrics
            )
            / total
        ),

        "top_10_accuracy": (
            sum(
                item["top_10"]
                for item in metrics
            )
            / total
        ),

        "top_20_accuracy": (
            sum(
                item["top_20"]
                for item in metrics
            )
            / total
        ),
    }


def main():

    method_summary_file = (
        RESULTS_ROOT
        / "Lang_1_20_final_summary.json"
    )

    if not method_summary_file.exists():

        raise RuntimeError(
            "Missing final method-level summary:\n"
            f"{method_summary_file}"
        )

    method_summary = (
        load_json(
            method_summary_file
        )
    )

    bug_records = (
        method_summary[
            "per_bug"
        ]
    )

    per_bug = []

    triggered_bugs = []

    total_new_candidates = 0

    failed = []

    for bug_record in bug_records:

        bug_id = (
            bug_record[
                "bug_id"
            ]
        )

        print()
        print("=" * 100)
        print(
            f"Adaptive CAMD: "
            f"{PROJECT}-{bug_id}"
        )
        print("=" * 100)

        initial_record, initial_source = (
            get_best_saved_record(
                bug_id
            )
        )

        if initial_record is None:

            failed.append(
                {
                    "bug_id": bug_id,
                    "error": (
                        "No saved Multi-Agent "
                        "results found."
                    ),
                }
            )

            print(
                "FAILED: no saved "
                "Multi-Agent result."
            )

            continue

        initial_score = (
            judge_score(
                initial_record
            )
        )

        initial_method = (
            initial_record.get(
                "method"
            )
        )

        print(
            f"Initial best: "
            f"{initial_method} "
            f"({initial_score:.4f})"
        )

        triggered = (
            initial_score
            < THRESHOLD
        )

        expansion_history = []

        if triggered:

            triggered_bugs.append(
                bug_id
            )

            for (
                initial_k,
                expanded_k,
            ) in EXPANSION_STAGES:

                current_record, _ = (
                    get_best_saved_record(
                        bug_id
                    )
                )

                current_score = (
                    judge_score(
                        current_record
                    )
                )

                if (
                    current_score
                    >= THRESHOLD
                ):
                    break

                runner = (
                    AdaptiveCandidateRunner(
                        project_root=(
                            PROJECT_ROOT
                        ),
                        threshold=(
                            THRESHOLD
                        ),
                        initial_top_k=(
                            initial_k
                        ),
                        expanded_top_k=(
                            expanded_k
                        ),
                    )
                )

                try:

                    result = (
                        runner.run_bug(
                            bug_record
                        )
                    )

                except Exception as exc:

                    failed.append(
                        {
                            "bug_id": bug_id,
                            "stage": (
                                f"{initial_k}"
                                f"->{expanded_k}"
                            ),
                            "error": (
                                str(exc)
                            ),
                        }
                    )

                    print(
                        f"FAILED during "
                        f"{initial_k}"
                        f"->{expanded_k}: "
                        f"{exc}"
                    )

                    break

                new_count = int(
                    result.get(
                        "new_candidates_evaluated",
                        0,
                    )
                )

                total_new_candidates += (
                    new_count
                )

                expansion_history.append(
                    {
                        "from_top_k": (
                            initial_k
                        ),
                        "to_top_k": (
                            expanded_k
                        ),
                        "new_candidates_evaluated": (
                            new_count
                        ),
                        "final_best_method": (
                            result.get(
                                "final_best_method"
                            )
                        ),
                        "final_best_score": (
                            result.get(
                                "final_best_score"
                            )
                        ),
                    }
                )

                print(
                    f"Expanded "
                    f"{initial_k}"
                    f"->{expanded_k}: "
                    f"best="
                    f"{result.get('final_best_method')} "
                    f"("
                    f"{result.get('final_best_score')}"
                    f")"
                )

        bug_dir = (
            RESULTS_ROOT
            / f"{PROJECT}_{bug_id}"
        )

        final_candidates = [
            bug_dir
            / "multi_agent_adaptive.jsonl",

            bug_dir
            / "multi_agent_expanded_test.jsonl",

            bug_dir
            / "multi_agent.jsonl",
        ]

        final_records = None
        final_source = None

        for path in final_candidates:

            if path.exists():

                records = (
                    load_jsonl(
                        path
                    )
                )

                if records:

                    final_records = (
                        records
                    )

                    final_source = (
                        path.name
                    )

                    break

        if not final_records:

            failed.append(
                {
                    "bug_id": bug_id,
                    "error": (
                        "Final candidate "
                        "ranking unavailable."
                    ),
                }
            )

            continue

        final_ranked = sorted(
            final_records,
            key=judge_score,
            reverse=True,
        )

        final_best = (
            final_ranked[0]
        )

        gt_rank = (
            find_gt_rank(
                final_ranked,
                bug_record.get(
                    "ground_truth_methods",
                    [],
                ),
            )
        )

        adaptive_metrics = (
            build_metrics(
                gt_rank
            )
        )

        per_bug.append(
            {
                "project": PROJECT,
                "bug_id": bug_id,

                "triggered": (
                    triggered
                ),

                "threshold": (
                    THRESHOLD
                ),

                "initial_best_method": (
                    initial_method
                ),

                "initial_best_score": (
                    initial_score
                ),

                "initial_source": (
                    initial_source
                ),

                "expansion_history": (
                    expansion_history
                ),

                "final_best_method": (
                    final_best.get(
                        "method"
                    )
                ),

                "final_best_score": (
                    judge_score(
                        final_best
                    )
                ),

                "final_source": (
                    final_source
                ),

                "adaptive_metrics": (
                    adaptive_metrics
                ),
            }
        )

    aggregate = (
        aggregate_metrics(
            per_bug
        )
    )

    base_metrics = (
        method_summary[
            "aggregate"
        ][
            "camd_multi_agent"
        ]
    )

    output = {
        "project": PROJECT,

        "bug_range": {
            "start": 1,
            "end": 20,
        },

        "threshold": (
            THRESHOLD
        ),

        "expansion_policy": [
            {
                "from_top_k": 5,
                "to_top_k": 10,
            },
            {
                "from_top_k": 10,
                "to_top_k": 20,
            },
        ],

        "triggered_bugs": (
            triggered_bugs
        ),

        "total_new_candidates_evaluated": (
            total_new_candidates
        ),

        "failed": failed,

        "base_camd": {
            "mrr": (
                base_metrics[
                    "mrr"
                ]
            ),

            "top_1_accuracy": (
                base_metrics[
                    "top_1_accuracy"
                ]
            ),

            "top_3_accuracy": (
                base_metrics[
                    "top_3_accuracy"
                ]
            ),

            "top_5_accuracy": (
                base_metrics[
                    "top_5_accuracy"
                ]
            ),
        },

        "adaptive_camd": (
            aggregate
        ),

        "per_bug": (
            per_bug
        ),
    }

    output_file = (
        RESULTS_ROOT
        / "Lang_1_20_adaptive_summary.json"
    )

    save_json(
        output_file,
        output,
    )

    print()
    print("=" * 100)
    print(
        "Adaptive CAMD Aggregate"
    )
    print("=" * 100)

    print(
        f"Evaluated bugs: "
        f"{aggregate['evaluated_bugs']}"
    )

    print(
        f"Triggered bugs: "
        f"{triggered_bugs}"
    )

    print(
        f"Extra candidates evaluated: "
        f"{total_new_candidates}"
    )

    print()

    print(
        "Base CAMD"
    )

    print(
        f"  MRR: "
        f"{base_metrics['mrr']:.4f}"
    )

    print(
        f"  Top-1: "
        f"{base_metrics['top_1_accuracy']:.4f}"
    )

    print()

    print(
        "Adaptive CAMD"
    )

    print(
        f"  MRR: "
        f"{aggregate['mrr']:.4f}"
    )

    print(
        f"  Top-1: "
        f"{aggregate['top_1_accuracy']:.4f}"
    )

    print(
        f"  Top-3: "
        f"{aggregate['top_3_accuracy']:.4f}"
    )

    print(
        f"  Top-5: "
        f"{aggregate['top_5_accuracy']:.4f}"
    )

    print(
        f"  Top-10: "
        f"{aggregate['top_10_accuracy']:.4f}"
    )

    print(
        f"  Top-20: "
        f"{aggregate['top_20_accuracy']:.4f}"
    )

    print()

    print(
        f"Failed bugs: "
        f"{[item['bug_id'] for item in failed]}"
    )

    print()

    print(
        "Summary saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()