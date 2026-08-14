import json
from pathlib import Path


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

BASE_TOP_K = 5

EXPANSION_STAGES = [
    10,
    20,
]


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
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
                json.loads(
                    line
                )
            )

    return records


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


def get_method_name(
    record: dict,
) -> str | None:

    return (
        record.get(
            "method"
        )
        or record.get(
            "method_name"
        )
    )


def sort_by_judge(
    records: list[dict],
) -> list[dict]:

    return sorted(
        records,
        key=judge_score,
        reverse=True,
    )


def load_base_records(
    bug_id: int,
) -> tuple[
    list[dict],
    str,
]:

    bug_dir = (
        RESULTS_ROOT
        / f"{PROJECT}_{bug_id}"
    )

    if bug_id <= 10:

        candidates = [
            bug_dir
            / "multi_agent_expanded_test.jsonl",

            bug_dir
            / "multi_agent.jsonl",
        ]

    else:

        candidates = [
            bug_dir
            / "multi_agent.jsonl",

            bug_dir
            / "multi_agent_expanded_test.jsonl",
        ]

    for path in candidates:

        if not path.exists():
            continue

        records = (
            load_jsonl(
                path
            )
        )

        if records:

            return (
                records,
                path.name,
            )

    raise RuntimeError(
        "No base Multi-Agent "
        f"result for Lang-{bug_id}"
    )


def load_adaptive_records(
    bug_id: int,
) -> list[dict] | None:

    path = (
        RESULTS_ROOT
        / f"{PROJECT}_{bug_id}"
        / "multi_agent_adaptive.jsonl"
    )

    if not path.exists():
        return None

    records = (
        load_jsonl(
            path
        )
    )

    if not records:
        return None

    return records


def record_matches_ground_truth(
    record: dict,
    ground_truth_methods: list[dict],
) -> bool:

    method_name = (
        get_method_name(
            record
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
            gt.get(
                "name"
            )
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

        overlaps = not (
            end_line < gt_start
            or start_line > gt_end
        )

        if overlaps:
            return True

    return False


def find_ground_truth_rank(
    records: list[dict],
    ground_truth_methods: list[dict],
) -> int | None:

    ranked = (
        sort_by_judge(
            records
        )
    )

    for rank, record in enumerate(
        ranked,
        start=1,
    ):

        if record_matches_ground_truth(
            record,
            ground_truth_methods,
        ):

            return rank

    return None


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


def select_records_up_to_b4_rank(
    records: list[dict],
    max_rank: int,
) -> list[dict]:

    selected = []

    for record in records:

        b4_rank = (
            record.get(
                "b4_rank"
            )
        )

        if b4_rank is None:
            continue

        try:
            rank_value = float(
                b4_rank
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if rank_value <= max_rank:

            selected.append(
                record
            )

    return selected


def best_record(
    records: list[dict],
) -> dict | None:

    if not records:
        return None

    return (
        sort_by_judge(
            records
        )[0]
    )


def aggregate_metrics(
    per_bug: list[dict],
    field: str,
) -> dict:

    metrics = [
        item[field]
        for item in per_bug
    ]

    total = len(
        metrics
    )

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

    return {
        "evaluated_bugs": total,

        "mrr": (
            sum(
                metric["rr"]
                for metric in metrics
            )
            / total
        ),

        "top_1_accuracy": (
            sum(
                metric["top_1"]
                for metric in metrics
            )
            / total
        ),

        "top_3_accuracy": (
            sum(
                metric["top_3"]
                for metric in metrics
            )
            / total
        ),

        "top_5_accuracy": (
            sum(
                metric["top_5"]
                for metric in metrics
            )
            / total
        ),

        "top_10_accuracy": (
            sum(
                metric["top_10"]
                for metric in metrics
            )
            / total
        ),

        "top_20_accuracy": (
            sum(
                metric["top_20"]
                for metric in metrics
            )
            / total
        ),
    }


def main():

    final_summary_file = (
        RESULTS_ROOT
        / "Lang_1_20_final_summary.json"
    )

    if not final_summary_file.exists():

        raise RuntimeError(
            "Missing method-level "
            "final summary:\n"
            f"{final_summary_file}"
        )

    final_summary = (
        load_json(
            final_summary_file
        )
    )

    per_bug = []

    triggered_bugs = []

    exhausted_bugs = []

    recovered_bugs = []

    failed = []

    total_extra_candidates = 0

    for bug_record in (
        final_summary[
            "per_bug"
        ]
    ):

        bug_id = (
            bug_record[
                "bug_id"
            ]
        )

        candidate_count = (
            bug_record.get(
                "candidate_method_count",
                0,
            )
            or 0
        )

        ground_truth_methods = (
            bug_record.get(
                "ground_truth_methods",
                [],
            )
        )

        try:

            (
                base_records,
                base_source,
            ) = (
                load_base_records(
                    bug_id
                )
            )

        except Exception as exc:

            failed.append(
                {
                    "bug_id": (
                        bug_id
                    ),
                    "error": (
                        str(exc)
                    ),
                }
            )

            continue

        base_ranked = (
            sort_by_judge(
                base_records
            )
        )

        if not base_ranked:

            failed.append(
                {
                    "bug_id": (
                        bug_id
                    ),
                    "error": (
                        "Base ranking is empty."
                    ),
                }
            )

            continue

        base_best = (
            base_ranked[0]
        )

        base_best_score = (
            judge_score(
                base_best
            )
        )

        base_gt_rank = (
            find_ground_truth_rank(
                base_ranked,
                ground_truth_methods,
            )
        )

        base_metrics = (
            build_metrics(
                base_gt_rank
            )
        )

        triggered = (
            base_best_score
            < THRESHOLD
        )

        expansion_history = []

        final_records = (
            base_records
        )

        adaptive_records = (
            load_adaptive_records(
                bug_id
            )
        )

        if triggered:

            triggered_bugs.append(
                bug_id
            )

            # -------------------------------------------------
            # Case 1:
            # Candidate pool already exhausted.
            #
            # Example:
            # Lang-6 has only 5 candidate methods.
            #
            # Confidence may be below threshold, but there is
            # literally nothing else to expand to.
            #
            # This is NOT an experiment failure.
            # -------------------------------------------------

            if (
                candidate_count
                <= BASE_TOP_K
            ):

                exhausted_bugs.append(
                    bug_id
                )

                expansion_history.append(
                    {
                        "from_top_k": (
                            BASE_TOP_K
                        ),

                        "to_top_k": (
                            BASE_TOP_K
                        ),

                        "best_method": (
                            get_method_name(
                                base_best
                            )
                        ),

                        "best_score": (
                            base_best_score
                        ),

                        "new_candidates_evaluated": (
                            0
                        ),

                        "expansion_available": (
                            False
                        ),

                        "reason": (
                            "Candidate pool exhausted; "
                            "no additional methods are "
                            "available for expansion."
                        ),
                    }
                )

                final_records = (
                    base_records
                )

            # -------------------------------------------------
            # Case 2:
            # Expansion is possible.
            # -------------------------------------------------

            else:

                if adaptive_records is None:

                    failed.append(
                        {
                            "bug_id": (
                                bug_id
                            ),

                            "error": (
                                "Adaptive expansion "
                                "was required but "
                                "multi_agent_adaptive.jsonl "
                                "is missing."
                            ),
                        }
                    )

                    continue

                current_k = (
                    BASE_TOP_K
                )

                current_best_score = (
                    base_best_score
                )

                for expanded_k in (
                    EXPANSION_STAGES
                ):

                    if (
                        current_best_score
                        >= THRESHOLD
                    ):
                        break

                    actual_expanded_k = (
                        min(
                            expanded_k,
                            candidate_count,
                        )
                    )

                    if (
                        actual_expanded_k
                        <= current_k
                    ):
                        break

                    stage_records = (
                        select_records_up_to_b4_rank(
                            adaptive_records,
                            actual_expanded_k,
                        )
                    )

                    stage_best = (
                        best_record(
                            stage_records
                        )
                    )

                    if stage_best is None:

                        failed.append(
                            {
                                "bug_id": (
                                    bug_id
                                ),

                                "error": (
                                    "Adaptive records "
                                    "could not reconstruct "
                                    f"Top-{actual_expanded_k}."
                                ),
                            }
                        )

                        break

                    stage_best_score = (
                        judge_score(
                            stage_best
                        )
                    )

                    new_candidates = (
                        actual_expanded_k
                        - current_k
                    )

                    total_extra_candidates += (
                        new_candidates
                    )

                    expansion_history.append(
                        {
                            "from_top_k": (
                                current_k
                            ),

                            "to_top_k": (
                                actual_expanded_k
                            ),

                            "best_method": (
                                get_method_name(
                                    stage_best
                                )
                            ),

                            "best_score": (
                                stage_best_score
                            ),

                            "new_candidates_evaluated": (
                                new_candidates
                            ),

                            "expansion_available": (
                                True
                            ),
                        }
                    )

                    current_k = (
                        actual_expanded_k
                    )

                    current_best_score = (
                        stage_best_score
                    )

                    final_records = (
                        stage_records
                    )

        final_ranked = (
            sort_by_judge(
                final_records
            )
        )

        if not final_ranked:

            failed.append(
                {
                    "bug_id": (
                        bug_id
                    ),
                    "error": (
                        "Final ranking is empty."
                    ),
                }
            )

            continue

        final_best = (
            final_ranked[0]
        )

        adaptive_gt_rank = (
            find_ground_truth_rank(
                final_ranked,
                ground_truth_methods,
            )
        )

        adaptive_metrics = (
            build_metrics(
                adaptive_gt_rank
            )
        )

        recovered = (
            not base_metrics[
                "top_1"
            ]
            and adaptive_metrics[
                "top_1"
            ]
        )

        if recovered:

            recovered_bugs.append(
                bug_id
            )

        per_bug.append(
            {
                "project": (
                    PROJECT
                ),

                "bug_id": (
                    bug_id
                ),

                "candidate_method_count": (
                    candidate_count
                ),

                "triggered": (
                    triggered
                ),

                "threshold": (
                    THRESHOLD
                ),

                "base_source": (
                    base_source
                ),

                "base_best_method": (
                    get_method_name(
                        base_best
                    )
                ),

                "base_best_score": (
                    base_best_score
                ),

                "base_metrics": (
                    base_metrics
                ),

                "expansion_history": (
                    expansion_history
                ),

                "adaptive_best_method": (
                    get_method_name(
                        final_best
                    )
                ),

                "adaptive_best_score": (
                    judge_score(
                        final_best
                    )
                ),

                "adaptive_metrics": (
                    adaptive_metrics
                ),

                "recovered_top1": (
                    recovered
                ),
            }
        )

    base_aggregate = (
        aggregate_metrics(
            per_bug,
            "base_metrics",
        )
    )

    adaptive_aggregate = (
        aggregate_metrics(
            per_bug,
            "adaptive_metrics",
        )
    )

    output = {
        "project": (
            PROJECT
        ),

        "bug_range": {
            "start": 1,
            "end": 20,
        },

        "protocol": {
            "threshold": (
                THRESHOLD
            ),

            "initial_top_k": (
                BASE_TOP_K
            ),

            "expansion_stages": (
                EXPANSION_STAGES
            ),

            "policy": (
                "Expand only when the highest "
                "Judge target-defect probability "
                "remains below the threshold. "
                "If the candidate pool is already "
                "exhausted, retain the base ranking."
            ),
        },

        "triggered_bugs": (
            triggered_bugs
        ),

        "candidate_pool_exhausted_bugs": (
            exhausted_bugs
        ),

        "recovered_top1_bugs": (
            recovered_bugs
        ),

        "total_extra_candidates_evaluated": (
            total_extra_candidates
        ),

        "failed": (
            failed
        ),

        "base_camd": (
            base_aggregate
        ),

        "adaptive_camd": (
            adaptive_aggregate
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
        "CAMD Adaptive Expansion "
        "Offline Aggregate"
    )
    print("=" * 100)

    print(
        f"Evaluated bugs: "
        f"{adaptive_aggregate['evaluated_bugs']}"
    )

    print(
        f"Triggered bugs: "
        f"{triggered_bugs}"
    )

    print(
        f"Candidate pool exhausted: "
        f"{exhausted_bugs}"
    )

    print(
        f"Recovered Top-1 bugs: "
        f"{recovered_bugs}"
    )

    print(
        f"Extra candidates evaluated: "
        f"{total_extra_candidates}"
    )

    print()

    print(
        "Base CAMD"
    )

    print(
        f"  MRR: "
        f"{base_aggregate['mrr']:.4f}"
    )

    print(
        f"  Top-1: "
        f"{base_aggregate['top_1_accuracy']:.4f}"
    )

    print(
        f"  Top-3: "
        f"{base_aggregate['top_3_accuracy']:.4f}"
    )

    print(
        f"  Top-5: "
        f"{base_aggregate['top_5_accuracy']:.4f}"
    )

    print(
        f"  Top-10: "
        f"{base_aggregate['top_10_accuracy']:.4f}"
    )

    print(
        f"  Top-20: "
        f"{base_aggregate['top_20_accuracy']:.4f}"
    )

    print()

    print(
        "Adaptive CAMD"
    )

    print(
        f"  MRR: "
        f"{adaptive_aggregate['mrr']:.4f}"
    )

    print(
        f"  Top-1: "
        f"{adaptive_aggregate['top_1_accuracy']:.4f}"
    )

    print(
        f"  Top-3: "
        f"{adaptive_aggregate['top_3_accuracy']:.4f}"
    )

    print(
        f"  Top-5: "
        f"{adaptive_aggregate['top_5_accuracy']:.4f}"
    )

    print(
        f"  Top-10: "
        f"{adaptive_aggregate['top_10_accuracy']:.4f}"
    )

    print(
        f"  Top-20: "
        f"{adaptive_aggregate['top_20_accuracy']:.4f}"
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