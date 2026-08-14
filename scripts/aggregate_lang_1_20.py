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

BUG_START = 1
BUG_END = 20

DEPRECATED_BUGS = {
    2,
    18,
}


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


def build_stage_metrics(
    per_bug: list[dict],
    stage_name: str,
) -> dict:

    valid = [
        item
        for item in per_bug
        if (
            item.get(stage_name)
            is not None
        )
    ]

    total = len(valid)

    if total == 0:

        return {
            "evaluated_bugs": 0,
            "mrr": 0.0,
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "top_10_accuracy": 0.0,
        }

    rr_values = [
        item[stage_name]["rr"]
        for item in valid
    ]

    top_1 = [
        item[stage_name]["top_1"]
        for item in valid
    ]

    top_3 = [
        item[stage_name]["top_3"]
        for item in valid
    ]

    top_5 = [
        item[stage_name]["top_5"]
        for item in valid
    ]

    top_10 = [
        item[stage_name]["top_10"]
        for item in valid
    ]

    return {
        "evaluated_bugs": total,

        "mrr": (
            sum(rr_values)
            / total
        ),

        "top_1_accuracy": (
            sum(top_1)
            / total
        ),

        "top_3_accuracy": (
            sum(top_3)
            / total
        ),

        "top_5_accuracy": (
            sum(top_5)
            / total
        ),

        "top_10_accuracy": (
            sum(top_10)
            / total
        ),
    }


def main():

    per_bug = []

    missing = []

    for bug_id in range(
        BUG_START,
        BUG_END + 1,
    ):

        if bug_id in DEPRECATED_BUGS:
            continue

        bug_dir = (
            RESULTS_ROOT
            / f"Lang_{bug_id}"
        )

        summary_file = (
            bug_dir
            / "summary.json"
        )

        if not summary_file.exists():

            missing.append(
                bug_id
            )

            continue

        summary = (
            load_json(
                summary_file
            )
        )

        # -------------------------------------------------
        # B1 and B4 always come from the normal summary.
        # -------------------------------------------------

        b1_metrics = (
            summary[
                "b1_method_only"
            ]
        )

        b4_metrics = (
            summary[
                "b4_static_aware"
            ]
        )

        # -------------------------------------------------
        # CAMD selection rule:
        #
        # Lang 1-10:
        # use expanded-test result if available.
        #
        # Lang 11-20:
        # current normal summary already uses expanded
        # failing-test context.
        # -------------------------------------------------

        expanded_file = (
            bug_dir
            / "summary_expanded_test.json"
        )

        if (
            bug_id <= 10
            and expanded_file.exists()
        ):

            expanded_summary = (
                load_json(
                    expanded_file
                )
            )

            camd_metrics = (
                expanded_summary[
                    "expanded_test_context"
                ]
            )

            camd_source = (
                "summary_expanded_test.json"
            )

        else:

            camd_metrics = (
                summary[
                    "camd_multi_agent"
                ]
            )

            camd_source = (
                "summary.json"
            )

        record = {
            "project": "Lang",
            "bug_id": bug_id,

            "modified_classes": (
                summary.get(
                    "modified_classes",
                    [],
                )
            ),

            "failing_tests": (
                summary.get(
                    "failing_tests",
                    [],
                )
            ),

            "ground_truth_methods": (
                summary.get(
                    "ground_truth_methods",
                    [],
                )
            ),

            "candidate_method_count": (
                summary.get(
                    "candidate_method_count"
                )
            ),

            "b1_method_only": (
                b1_metrics
            ),

            "b4_static_aware": (
                b4_metrics
            ),

            "camd_multi_agent": (
                camd_metrics
            ),

            "camd_result_source": (
                camd_source
            ),
        }

        per_bug.append(
            record
        )

    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    aggregate = {
        "total_valid_bugs": (
            len(per_bug)
        ),

        "deprecated_bugs": (
            sorted(
                DEPRECATED_BUGS
            )
        ),

        "included_bugs": [
            item["bug_id"]
            for item in per_bug
        ],

        "b1_method_only": (
            build_stage_metrics(
                per_bug,
                "b1_method_only",
            )
        ),

        "b4_static_aware": (
            build_stage_metrics(
                per_bug,
                "b4_static_aware",
            )
        ),

        "camd_multi_agent": (
            build_stage_metrics(
                per_bug,
                "camd_multi_agent",
            )
        ),
    }

    # -----------------------------------------------------
    # Candidate-generation failures
    # -----------------------------------------------------

    candidate_generation_failures = []

    for item in per_bug:

        b4 = (
            item[
                "b4_static_aware"
            ]
        )

        if (
            not b4["top_5"]
        ):

            candidate_generation_failures.append(
                {
                    "bug_id": (
                        item["bug_id"]
                    ),
                    "b4_rank": (
                        b4["rank"]
                    ),
                    "camd_rank": (
                        item[
                            "camd_multi_agent"
                        ]["rank"]
                    ),
                }
            )

    final_summary = {
        "project": "Lang",

        "bug_range": {
            "start": BUG_START,
            "end": BUG_END,
        },

        "evaluation_protocol": {
            "deprecated_bugs_excluded": (
                sorted(
                    DEPRECATED_BUGS
                )
            ),

            "camd_test_context": (
                "Failing test body with "
                "1-hop direct test helper "
                "expansion when available."
            ),

            "candidate_pruning": (
                "Top-5 candidates from "
                "B4 static-aware ranking."
            ),

            "tie_handling": (
                "Average rank for equal "
                "suspicion scores."
            ),
        },

        "missing_non_deprecated_bugs": (
            missing
        ),

        "per_bug": (
            per_bug
        ),

        "aggregate": (
            aggregate
        ),

        "candidate_generation_failures": (
            candidate_generation_failures
        ),
    }

    output_file = (
        RESULTS_ROOT
        / "Lang_1_20_final_summary.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Print report
    # -----------------------------------------------------

    print()
    print("=" * 100)
    print(
        "CAMD Final Lang 1-20 Results"
    )
    print("=" * 100)

    print(
        f"Valid bugs: "
        f"{aggregate['total_valid_bugs']}"
    )

    print(
        f"Deprecated: "
        f"{aggregate['deprecated_bugs']}"
    )

    print(
        f"Missing non-deprecated bugs: "
        f"{missing}"
    )

    print()

    for stage in [
        "b1_method_only",
        "b4_static_aware",
        "camd_multi_agent",
    ]:

        metrics = (
            aggregate[stage]
        )

        print(stage)

        print(
            f"  MRR: "
            f"{metrics['mrr']:.4f}"
        )

        print(
            f"  Top-1: "
            f"{metrics['top_1_accuracy']:.4f}"
        )

        print(
            f"  Top-3: "
            f"{metrics['top_3_accuracy']:.4f}"
        )

        print(
            f"  Top-5: "
            f"{metrics['top_5_accuracy']:.4f}"
        )

        print(
            f"  Top-10: "
            f"{metrics['top_10_accuracy']:.4f}"
        )

        print()

    print(
        "Candidate-generation failures:"
    )

    if not candidate_generation_failures:

        print(
            "  None"
        )

    else:

        for failure in (
            candidate_generation_failures
        ):

            print(
                f"  Lang-{failure['bug_id']}: "
                f"B4 rank="
                f"{failure['b4_rank']}, "
                f"CAMD rank="
                f"{failure['camd_rank']}"
            )

    print()

    print(
        f"Final summary saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()