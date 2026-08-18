from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DEFAULT_FINAL_VERIFIER = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "final_verifier_summary.json"
)

DEFAULT_FAILURE_ANALYSIS = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "final_failure_analysis.json"
)

DEFAULT_RECOVERY_ANALYSIS = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "candidate_recovery_analysis.json"
)

DEFAULT_POSITION_ANALYSIS = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "recovered_candidate_positions.json"
)

DEFAULT_EXPANSION_ANALYSIS = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "expansion_signal_analysis.json"
)

DEFAULT_DEV_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "verifier_dev_summary.json"
)

DEFAULT_JSON_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "camd_final_experiment_summary.json"
)

DEFAULT_MD_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "verification"
    / "final"
    / "CAMD_FINAL_RESULTS.md"
)


# Historical first official retrieval evaluation.
#
# Important:
# The later frozen candidate-pool export reproduced
# @10 = 71, @20 = 71, @100 = 86,
# but produced @50 = 80 because Mockito-25 was recovered.
#
# The original unbiased first-run @50 result remains 79/98.
OFFICIAL_FIRST_RUN_RETRIEVAL = {
    "10": {
        "count": 71,
        "total": 98,
    },
    "20": {
        "count": 71,
        "total": 98,
    },
    "50": {
        "count": 79,
        "total": 98,
    },
    "100": {
        "count": 86,
        "total": 98,
    },
}


def load_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    path: Path,
    data: dict,
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


def save_text(
    path: Path,
    text: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def rate(
    count: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return count / total


def pct(
    value: float,
) -> str:

    return (
        f"{value * 100:.2f}%"
    )


def metric_value(
    data: dict,
    section: str,
    key: str,
    default=None,
):

    return (
        data
        .get(
            section,
            {},
        )
        .get(
            key,
            default,
        )
    )


def build_report(
    final_verifier: dict,
    failure_analysis: dict,
    recovery_analysis: dict,
    position_analysis: dict,
    expansion_analysis: dict,
    dev_summary: dict,
) -> dict:

    total = int(
        final_verifier[
            "configuration"
        ][
            "total_cases"
        ]
    )

    # =========================================================
    # Current frozen candidate-pool coverage
    # =========================================================

    frozen_coverage_raw = (
        recovery_analysis[
            "coverage"
        ]
    )

    frozen_retrieval = {}

    for budget in [
        "10",
        "20",
        "50",
        "100",
    ]:

        count = int(
            frozen_coverage_raw[
                budget
            ]
        )

        frozen_retrieval[
            budget
        ] = {
            "count": count,
            "total": total,
            "recall": rate(
                count,
                total,
            ),
        }

    official_retrieval = {}

    for budget, item in (
        OFFICIAL_FIRST_RUN_RETRIEVAL.items()
    ):

        official_retrieval[
            budget
        ] = {
            **item,
            "recall": rate(
                item[
                    "count"
                ],
                item[
                    "total"
                ],
            ),
        }

    # =========================================================
    # Detector / Judge
    # =========================================================

    detector = {
        "top1": metric_value(
            final_verifier,
            "detector",
            "top1",
        ),
        "top3": metric_value(
            final_verifier,
            "detector",
            "top3",
        ),
        "top5": metric_value(
            final_verifier,
            "detector",
            "top5",
        ),
        "top10": metric_value(
            final_verifier,
            "detector",
            "top10",
        ),
        "mrr": metric_value(
            final_verifier,
            "detector",
            "mrr",
        ),
    }

    judge = {
        "top1": metric_value(
            final_verifier,
            "judge",
            "top1",
        ),
        "top3": metric_value(
            final_verifier,
            "judge",
            "top3",
        ),
        "top5": metric_value(
            final_verifier,
            "judge",
            "top5",
        ),
        "top10": metric_value(
            final_verifier,
            "judge",
            "top10",
        ),
        "mrr": metric_value(
            final_verifier,
            "judge",
            "mrr",
        ),
    }

    conditional = (
        final_verifier[
            "conditional"
        ]
    )

    # =========================================================
    # Failure decomposition
    # =========================================================

    failure_overall = (
        failure_analysis[
            "overall"
        ]
    )

    retrieval_failures = int(
        failure_overall[
            "retrieval_failures"
        ]
    )

    detector_ranking_failures = int(
        failure_overall[
            "detector_ranking_failures_given_recall"
        ]
    )

    detector_top1_correct = int(
        failure_overall[
            "detector_top1_correct"
        ]
    )

    total_detector_failures = (
        total
        - detector_top1_correct
    )

    # =========================================================
    # Recovery analysis
    # =========================================================

    recovery = (
        recovery_analysis[
            "recovery"
        ]
    )

    first_at_20 = len(
        recovery[
            "first_at_20"
        ]
    )

    first_at_50 = len(
        recovery[
            "first_at_50"
        ]
    )

    first_at_100 = len(
        recovery[
            "first_at_100"
        ]
    )

    never_recovered = len(
        recovery[
            "never_recovered"
        ]
    )

    recoverable_depth_failures = (
        first_at_20
        + first_at_50
        + first_at_100
    )

    # =========================================================
    # Position analysis
    # =========================================================

    positions = (
        position_analysis.get(
            "recovered_pool_positions",
            [],
        )
    )

    sorted_positions = sorted(
        positions
    )

    if sorted_positions:

        position_mean = (
            sum(
                sorted_positions
            )
            / len(
                sorted_positions
            )
        )

        n_positions = len(
            sorted_positions
        )

        if n_positions % 2 == 1:

            position_median = (
                sorted_positions[
                    n_positions // 2
                ]
            )

        else:

            position_median = (
                sorted_positions[
                    n_positions // 2 - 1
                ]
                + sorted_positions[
                    n_positions // 2
                ]
            ) / 2

        position_stats = {
            "n_gt_methods": (
                n_positions
            ),
            "min": min(
                sorted_positions
            ),
            "max": max(
                sorted_positions
            ),
            "mean": (
                position_mean
            ),
            "median": (
                position_median
            ),
        }

    else:

        position_stats = {
            "n_gt_methods": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    admission_sources = (
        position_analysis
        .get(
            "summary",
            {},
        )
        .get(
            "admission_sources",
            {},
        )
    )

    # =========================================================
    # Dev vs final
    # =========================================================

    dev_detector_top1 = (
        dev_summary[
            "detector"
        ][
            "top1"
        ]
    )

    dev_judge_top1 = (
        dev_summary[
            "judge"
        ][
            "top1"
        ]
    )

    dev_detector_mrr = (
        dev_summary[
            "detector"
        ][
            "mrr"
        ]
    )

    dev_judge_mrr = (
        dev_summary[
            "judge"
        ][
            "mrr"
        ]
    )

    # =========================================================
    # Expansion-signal post-hoc analysis
    # =========================================================

    expansion_summary = (
        expansion_analysis[
            "summary"
        ]
    )

    # =========================================================
    # Final report
    # =========================================================

    report = {
        "artifact": (
            "CAMD final experiment summary"
        ),

        "status": (
            "frozen held-out evaluation"
        ),

        "benchmark": {
            "method_applicable_bugs": (
                total
            ),
            "projects": [
                "Lang",
                "Math",
                "Chart",
                "Time",
                "Mockito",
            ],
            "retrieval_budget_main": 10,
            "verifier_top_k": 10,
        },

        "retrieval": {
            "official_first_run": (
                official_retrieval
            ),

            "current_frozen_candidate_pools": (
                frozen_retrieval
            ),

            "audit_note": (
                "The original unbiased first final retrieval run "
                "reported Recall@50 = 79/98. The later frozen "
                "candidate-pool export reports 80/98 because "
                "Mockito-25 entered the Top-50 pool. The historical "
                "run contained 7 failing tests for Mockito-25, while "
                "subsequent stable exports contained 6. Because "
                "failure evidence contributes to retrieval scoring, "
                "the original 79/98 result is retained as the official "
                "first-run retrieval metric. Downstream verifier "
                "analyses use the internally consistent frozen "
                "candidate pools."
            ),
        },

        "detector_final": (
            detector
        ),

        "judge_final": (
            judge
        ),

        "conditional_final": {
            "n": (
                conditional[
                    "n"
                ]
            ),

            "detector_top1": (
                conditional[
                    "detector_top1"
                ]
            ),

            "judge_top1": (
                conditional[
                    "judge_top1"
                ]
            ),

            "detector_top3": (
                conditional[
                    "detector_top3"
                ]
            ),

            "judge_top3": (
                conditional[
                    "judge_top3"
                ]
            ),

            "detector_mrr": (
                conditional[
                    "detector_mrr"
                ]
            ),

            "judge_mrr": (
                conditional[
                    "judge_mrr"
                ]
            ),
        },

        "multi_agent_transition_final": (
            final_verifier[
                "transitions"
            ]
        ),

        "failure_decomposition": {
            "total_detector_top1_failures": (
                total_detector_failures
            ),

            "retrieval_failures": (
                retrieval_failures
            ),

            "retrieval_failure_share": rate(
                retrieval_failures,
                total_detector_failures,
            ),

            "detector_ranking_failures": (
                detector_ranking_failures
            ),

            "detector_ranking_failure_share": rate(
                detector_ranking_failures,
                total_detector_failures,
            ),

            "recoverable_depth_failures": (
                recoverable_depth_failures
            ),

            "still_missing_at_100": (
                never_recovered
            ),

            "recovered_first_at_20": (
                first_at_20
            ),

            "recovered_first_at_50": (
                first_at_50
            ),

            "recovered_first_at_100": (
                first_at_100
            ),
        },

        "recovered_candidate_positions": {
            **position_stats,
            "admission_sources": (
                admission_sources
            ),
        },

        "per_project_failure_decomposition": (
            failure_analysis[
                "per_project"
            ]
        ),

        "dev_vs_final_multi_agent": {
            "development": {
                "detector_top1": (
                    dev_detector_top1
                ),

                "judge_top1": (
                    dev_judge_top1
                ),

                "delta_top1": (
                    dev_judge_top1
                    - dev_detector_top1
                ),

                "detector_mrr": (
                    dev_detector_mrr
                ),

                "judge_mrr": (
                    dev_judge_mrr
                ),

                "delta_mrr": (
                    dev_judge_mrr
                    - dev_detector_mrr
                ),
            },

            "held_out_final": {
                "detector_top1": (
                    detector[
                        "top1"
                    ]
                ),

                "judge_top1": (
                    judge[
                        "top1"
                    ]
                ),

                "delta_top1": (
                    judge[
                        "top1"
                    ]
                    - detector[
                        "top1"
                    ]
                ),

                "detector_mrr": (
                    detector[
                        "mrr"
                    ]
                ),

                "judge_mrr": (
                    judge[
                        "mrr"
                    ]
                ),

                "delta_mrr": (
                    judge[
                        "mrr"
                    ]
                    - detector[
                        "mrr"
                    ]
                ),
            },

            "interpretation": (
                "Critic/Judge improved the development set but did "
                "not generalize to the frozen held-out benchmark. "
                "The multi-agent verifier should therefore be treated "
                "as an ablation rather than the primary CAMD claim."
            ),
        },

        "post_hoc_expansion_signal_analysis": {
            "groups": (
                expansion_summary
            ),

            "warning": (
                "These confidence distributions were inspected after "
                "the held-out final results were known. They support "
                "hypothesis generation only and must not be used to "
                "claim a tuned adaptive-expansion improvement on the "
                "same final benchmark."
            ),
        },

        "main_findings": [
            (
                "Program-wide candidate retrieval is the dominant "
                "end-to-end bottleneck."
            ),
            (
                "At K=10, the Detector achieves very high conditional "
                "Top-1 accuracy once the ground-truth method is present."
            ),
            (
                "Additional Critic/Judge deliberation does not improve "
                "held-out localization accuracy."
            ),
            (
                "Fifteen of the twenty-seven K=10 retrieval misses are "
                "recoverable by expanding the frozen base candidate "
                "pool to K<=100."
            ),
            (
                "All observed recovered ground-truth methods entered "
                "through base retrieval rather than stack/call "
                "augmentation."
            ),
            (
                "Future work should prioritize candidate ranking, "
                "efficient reranking, and separately investigate the "
                "twelve cases still absent at K=100."
            ),
        ],
    }

    return report


def build_markdown(
    report: dict,
) -> str:

    total = (
        report[
            "benchmark"
        ][
            "method_applicable_bugs"
        ]
    )

    official = (
        report[
            "retrieval"
        ][
            "official_first_run"
        ]
    )

    frozen = (
        report[
            "retrieval"
        ][
            "current_frozen_candidate_pools"
        ]
    )

    detector = (
        report[
            "detector_final"
        ]
    )

    judge = (
        report[
            "judge_final"
        ]
    )

    conditional = (
        report[
            "conditional_final"
        ]
    )

    failures = (
        report[
            "failure_decomposition"
        ]
    )

    positions = (
        report[
            "recovered_candidate_positions"
        ]
    )

    dev_final = (
        report[
            "dev_vs_final_multi_agent"
        ]
    )

    per_project = (
        report[
            "per_project_failure_decomposition"
        ]
    )

    lines = []

    lines.append(
        "# CAMD Final Experimental Results"
    )

    lines.append("")

    lines.append(
        "> Status: frozen held-out evaluation. "
        "Do not tune prompts, thresholds, retrieval budgets, "
        "or scoring rules against these final results."
    )

    lines.append("")

    lines.append(
        "## 1. Benchmark"
    )

    lines.append("")

    lines.append(
        f"- Method-applicable bugs: **{total}**"
    )

    lines.append(
        "- Projects: **Lang, Math, Chart, Time, Mockito**"
    )

    lines.append(
        "- Main retrieval budget: **K = 10**"
    )

    lines.append(
        "- Multi-agent verification shortlist: **Detector Top-10**"
    )

    lines.append("")

    # =========================================================
    # Retrieval
    # =========================================================

    lines.append(
        "## 2. Program-Wide Retrieval"
    )

    lines.append("")

    lines.append(
        "### Original first-run final retrieval"
    )

    lines.append("")

    lines.append(
        "| Budget | Recall |"
    )

    lines.append(
        "|---:|---:|"
    )

    for budget in [
        "10",
        "20",
        "50",
        "100",
    ]:

        item = (
            official[
                budget
            ]
        )

        lines.append(
            f"| {budget} | "
            f"{item['count']}/{item['total']} "
            f"({pct(item['recall'])}) |"
        )

    lines.append("")

    lines.append(
        "### Current frozen candidate-pool artifact"
    )

    lines.append("")

    lines.append(
        "| Budget | Recall |"
    )

    lines.append(
        "|---:|---:|"
    )

    for budget in [
        "10",
        "20",
        "50",
        "100",
    ]:

        item = (
            frozen[
                budget
            ]
        )

        lines.append(
            f"| {budget} | "
            f"{item['count']}/{item['total']} "
            f"({pct(item['recall'])}) |"
        )

    lines.append("")

    lines.append(
        "**Audit note.** "
        + report[
            "retrieval"
        ][
            "audit_note"
        ]
    )

    lines.append("")

    # =========================================================
    # Detector / Judge
    # =========================================================

    lines.append(
        "## 3. End-to-End Localization"
    )

    lines.append("")

    lines.append(
        "| Method | Top-1 | Top-3 | Top-5 | Top-10 | MRR |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    lines.append(
        "| Retriever + Detector | "
        f"{pct(detector['top1'])} | "
        f"{pct(detector['top3'])} | "
        f"{pct(detector['top5'])} | "
        f"{pct(detector['top10'])} | "
        f"{detector['mrr']:.4f} |"
    )

    lines.append(
        "| Retriever + Detector + Critic + Judge | "
        f"{pct(judge['top1'])} | "
        f"{pct(judge['top3'])} | "
        f"{pct(judge['top5'])} | "
        f"{pct(judge['top10'])} | "
        f"{judge['mrr']:.4f} |"
    )

    lines.append("")

    detector_top1_count = round(
        detector[
            "top1"
        ]
        * total
    )

    judge_top1_count = round(
        judge[
            "top1"
        ]
        * total
    )

    lines.append(
        f"Detector Top-1: **{detector_top1_count}/{total} "
        f"= {pct(detector['top1'])}**."
    )

    lines.append(
        f"Judge Top-1: **{judge_top1_count}/{total} "
        f"= {pct(judge['top1'])}**."
    )

    lines.append("")

    # =========================================================
    # Conditional
    # =========================================================

    lines.append(
        "## 4. Conditional Localization Quality"
    )

    lines.append("")

    n_conditional = (
        conditional[
            "n"
        ]
    )

    detector_cond_count = round(
        conditional[
            "detector_top1"
        ]
        * n_conditional
    )

    judge_cond_count = round(
        conditional[
            "judge_top1"
        ]
        * n_conditional
    )

    lines.append(
        f"Among the **{n_conditional}** bugs whose ground-truth "
        "method is present in the Detector Top-10 shortlist:"
    )

    lines.append("")

    lines.append(
        f"- Detector Top-1: **{detector_cond_count}/{n_conditional} "
        f"= {pct(conditional['detector_top1'])}**"
    )

    lines.append(
        f"- Judge Top-1: **{judge_cond_count}/{n_conditional} "
        f"= {pct(conditional['judge_top1'])}**"
    )

    lines.append(
        f"- Detector conditional MRR: "
        f"**{conditional['detector_mrr']:.4f}**"
    )

    lines.append(
        f"- Judge conditional MRR: "
        f"**{conditional['judge_mrr']:.4f}**"
    )

    lines.append("")

    lines.append(
        "This shows that once the correct method enters the "
        "shortlist, the single Detector is already highly effective."
    )

    lines.append("")

    # =========================================================
    # Failure decomposition
    # =========================================================

    lines.append(
        "## 5. Final Failure Decomposition"
    )

    lines.append("")

    lines.append(
        f"Detector Top-1 failures: "
        f"**{failures['total_detector_top1_failures']}**"
    )

    lines.append("")

    lines.append(
        f"- Retrieval failures: "
        f"**{failures['retrieval_failures']} "
        f"({pct(failures['retrieval_failure_share'])})**"
    )

    lines.append(
        f"- Detector ranking failures after successful retrieval: "
        f"**{failures['detector_ranking_failures']} "
        f"({pct(failures['detector_ranking_failure_share'])})**"
    )

    lines.append("")

    lines.append(
        "The failure tree is therefore:"
    )

    lines.append("")

    lines.append("```text")

    lines.append(
        f"{total} method-applicable bugs"
    )

    lines.append(
        "├── 71 GT retrieved at K=10"
    )

    lines.append(
        "│   ├── 67 Detector Top-1 correct"
    )

    lines.append(
        "│   └── 4 Detector ranking failures"
    )

    lines.append(
        "└── 27 GT missing at K=10"
    )

    lines.append(
        f"    ├── {failures['recovered_first_at_50']} "
        "first recovered at K=50"
    )

    lines.append(
        f"    ├── {failures['recovered_first_at_100']} "
        "first recovered at K=100"
    )

    lines.append(
        f"    └── {failures['still_missing_at_100']} "
        "still absent at K=100"
    )

    lines.append("```")

    lines.append("")

    # =========================================================
    # Recovery
    # =========================================================

    lines.append(
        "## 6. Candidate-Depth Analysis"
    )

    lines.append("")

    lines.append(
        f"Of the 27 K=10 retrieval misses, "
        f"**{failures['recoverable_depth_failures']}** "
        "are recoverable by expanding the frozen candidate pool "
        "to K<=100."
    )

    lines.append("")

    lines.append(
        f"Observed recovered GT methods: "
        f"**{positions['n_gt_methods']}**"
    )

    lines.append(
        f"- Minimum pool/base rank: **{positions['min']}**"
    )

    lines.append(
        f"- Maximum pool/base rank: **{positions['max']}**"
    )

    lines.append(
        f"- Mean rank: **{positions['mean']:.2f}**"
    )

    lines.append(
        f"- Median rank: **{positions['median']}**"
    )

    sources = (
        positions[
            "admission_sources"
        ]
    )

    lines.append("")

    lines.append(
        "Admission sources for recovered GT methods:"
    )

    lines.append("")

    for source, count in sorted(
        sources.items()
    ):

        lines.append(
            f"- {source}: **{count}**"
        )

    lines.append("")

    lines.append(
        "All observed recovered GT methods entered through "
        "**base retrieval**, indicating that these cases are "
        "primarily ranking-depth failures rather than failures "
        "of stack/call augmentation."
    )

    lines.append("")

    # =========================================================
    # Per project
    # =========================================================

    lines.append(
        "## 7. Per-Project Failure Decomposition"
    )

    lines.append("")

    lines.append(
        "| Project | Total | Retrieval success | Retrieval miss | "
        "Detector Top-1 correct | Detector ranking failure |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for project in [
        "Chart",
        "Lang",
        "Math",
        "Mockito",
        "Time",
    ]:

        stats = (
            per_project[
                project
            ]
        )

        lines.append(
            f"| {project} | "
            f"{stats['total']} | "
            f"{stats['retrieval_success']} | "
            f"{stats['retrieval_failure']} | "
            f"{stats['detector_top1_correct']} | "
            f"{stats['detector_top1_wrong_given_recall']} |"
        )

    lines.append("")

    # =========================================================
    # Multi-agent
    # =========================================================

    lines.append(
        "## 8. Multi-Agent Verification Ablation"
    )

    lines.append("")

    dev = (
        dev_final[
            "development"
        ]
    )

    final = (
        dev_final[
            "held_out_final"
        ]
    )

    lines.append(
        "| Split | Detector Top-1 | Judge Top-1 | Delta |"
    )

    lines.append(
        "|---|---:|---:|---:|"
    )

    lines.append(
        "| Development | "
        f"{pct(dev['detector_top1'])} | "
        f"{pct(dev['judge_top1'])} | "
        f"{dev['delta_top1'] * 100:+.2f} pp |"
    )

    lines.append(
        "| Held-out final | "
        f"{pct(final['detector_top1'])} | "
        f"{pct(final['judge_top1'])} | "
        f"{final['delta_top1'] * 100:+.2f} pp |"
    )

    lines.append("")

    lines.append(
        "The Critic/Judge stage improved the development set "
        "but did not generalize to the frozen held-out benchmark. "
        "It should therefore be reported as an **ablation**, not "
        "as the primary CAMD improvement."
    )

    lines.append("")

    # =========================================================
    # Confidence
    # =========================================================

    lines.append(
        "## 9. Post-Hoc Expansion-Signal Analysis"
    )

    lines.append("")

    lines.append(
        "| Group | N | Mean Detector p1 | Median p1 | "
        "Mean Top1-Top2 margin |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    expansion_groups = (
        report[
            "post_hoc_expansion_signal_analysis"
        ][
            "groups"
        ]
    )

    for group in [
        "retrieved_at_10",
        "recovered_at_50",
        "recovered_at_100",
        "never_recovered",
    ]:

        values = (
            expansion_groups[
                group
            ]
        )

        p1_stats = (
            values[
                "p1"
            ]
        )

        margin_stats = (
            values[
                "margin"
            ]
        )

        lines.append(
            f"| {group} | "
            f"{p1_stats['n']} | "
            f"{p1_stats['mean']:.4f} | "
            f"{p1_stats['median']:.4f} | "
            f"{margin_stats['mean']:.4f} |"
        )

    lines.append("")

    lines.append(
        "**Important methodological warning:** "
        + report[
            "post_hoc_expansion_signal_analysis"
        ][
            "warning"
        ]
    )

    lines.append("")

    # =========================================================
    # Main conclusions
    # =========================================================

    lines.append(
        "## 10. Frozen Conclusions"
    )

    lines.append("")

    lines.append(
        "1. **Candidate retrieval is the dominant bottleneck.** "
        "27 of 31 Detector Top-1 failures originate before "
        "LLM verification."
    )

    lines.append(
        "2. **The Detector is strong once GT is retrieved.** "
        "Conditional Top-1 reaches 94.37%."
    )

    lines.append(
        "3. **Additional multi-agent deliberation does not "
        "improve held-out performance.**"
    )

    lines.append(
        "4. **Candidate depth matters.** "
        "15 of 27 K=10 misses become reachable by K<=100."
    )

    lines.append(
        "5. **The recoverable failures are base-ranking failures.** "
        "All observed recovered GT methods were admitted by the "
        "base retriever."
    )

    lines.append(
        "6. **Future work should prioritize efficient reranking "
        "and candidate coverage**, while separately investigating "
        "the 12 cases still absent at K=100."
    )

    lines.append("")

    lines.append(
        "### Recommended CAMD v1 framing"
    )

    lines.append("")

    lines.append("```text")

    lines.append(
        "Whole-program methods"
    )

    lines.append(
        "        ↓"
    )

    lines.append(
        "Program-wide evidence-aware retrieval"
    )

    lines.append(
        "        ↓"
    )

    lines.append(
        "Top-K candidate shortlist"
    )

    lines.append(
        "        ↓"
    )

    lines.append(
        "LLM evidence-grounded Detector"
    )

    lines.append("```")

    lines.append("")

    lines.append(
        "Critic/Judge is retained as an experimental "
        "verification ablation rather than the main inference path."
    )

    lines.append("")

    return "\n".join(
        lines
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--final-verifier",
        type=Path,
        default=DEFAULT_FINAL_VERIFIER,
    )

    parser.add_argument(
        "--failure-analysis",
        type=Path,
        default=DEFAULT_FAILURE_ANALYSIS,
    )

    parser.add_argument(
        "--recovery-analysis",
        type=Path,
        default=DEFAULT_RECOVERY_ANALYSIS,
    )

    parser.add_argument(
        "--position-analysis",
        type=Path,
        default=DEFAULT_POSITION_ANALYSIS,
    )

    parser.add_argument(
        "--expansion-analysis",
        type=Path,
        default=DEFAULT_EXPANSION_ANALYSIS,
    )

    parser.add_argument(
        "--dev-summary",
        type=Path,
        default=DEFAULT_DEV_SUMMARY,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )

    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD_OUTPUT,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    final_verifier = load_json(
        args.final_verifier
    )

    failure_analysis = load_json(
        args.failure_analysis
    )

    recovery_analysis = load_json(
        args.recovery_analysis
    )

    position_analysis = load_json(
        args.position_analysis
    )

    expansion_analysis = load_json(
        args.expansion_analysis
    )

    dev_summary = load_json(
        args.dev_summary
    )

    report = build_report(
        final_verifier=(
            final_verifier
        ),
        failure_analysis=(
            failure_analysis
        ),
        recovery_analysis=(
            recovery_analysis
        ),
        position_analysis=(
            position_analysis
        ),
        expansion_analysis=(
            expansion_analysis
        ),
        dev_summary=(
            dev_summary
        ),
    )

    markdown = build_markdown(
        report
    )

    save_json(
        args.json_output,
        report,
    )

    save_text(
        args.md_output,
        markdown,
    )

    print("=" * 100)
    print(
        "CAMD Final Experiment Report"
    )
    print("=" * 100)

    print(
        "JSON:"
    )

    print(
        args.json_output
    )

    print()

    print(
        "Markdown:"
    )

    print(
        args.md_output
    )

    print()

    print(
        "No LLM/API calls were used."
    )


if __name__ == "__main__":
    main()