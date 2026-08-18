#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_method(signature):
    """
    Compare at fully-qualified class + method-name level because
    CAMD frozen GT does not preserve parameter signatures.
    """
    signature = str(signature).strip()

    if "(" in signature:
        signature = signature.split("(", 1)[0]

    return signature.strip()


def reciprocal_rank_at_5(rank):
    if rank is None or rank <= 0 or rank > 5:
        return 0.0

    return 1.0 / rank


def metrics_at_5(ranks):
    n = len(ranks)

    def topk(k):
        return sum(
            rank is not None and rank <= k
            for rank in ranks
        )

    return {
        "n": n,

        "top1_count": topk(1),
        "top1": topk(1) / n if n else 0.0,

        "top3_count": topk(3),
        "top3": topk(3) / n if n else 0.0,

        "top5_count": topk(5),
        "top5": topk(5) / n if n else 0.0,

        "mrr_at_5": (
            sum(
                reciprocal_rank_at_5(rank)
                for rank in ranks
            ) / n
            if n else 0.0
        ),
    }


def exact_mcnemar(b, c):
    """
    Two-sided exact McNemar test.

    b: CAMD Top-1 correct, FlexFL wrong
    c: CAMD wrong, FlexFL Top-1 correct
    """
    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)

    tail = (
        sum(
            math.comb(n, i)
            for i in range(k + 1)
        )
        / (2 ** n)
    )

    return min(1.0, 2.0 * tail)


def flexfl_rank(result, gt_methods):
    """
    FlexFL publishes the final postprocessed Top-5 list in fl_results.

    Return the first rank at which one of FlexFL's own GT methods
    occurs, matching the protocol in Results/eval.py.
    """
    suspicious = result.get("fl_results", [])

    if not isinstance(suspicious, list):
        raise ValueError(
            "Expected FlexFL 'fl_results' to be a list."
        )

    for index, method in enumerate(suspicious[:5], start=1):
        if method in gt_methods:
            return index

    return None


def compare(ids, camd_ranks, flexfl_ranks):
    ids = sorted(ids)

    camd_values = [
        camd_ranks[bug_id]
        for bug_id in ids
    ]

    flexfl_values = [
        flexfl_ranks[bug_id]
        for bug_id in ids
    ]

    camd_metrics = metrics_at_5(camd_values)
    flexfl_metrics = metrics_at_5(flexfl_values)

    both = []
    camd_only = []
    flexfl_only = []
    neither = []

    for bug_id in ids:
        c = camd_ranks[bug_id] == 1
        f = flexfl_ranks[bug_id] == 1

        if c and f:
            both.append(bug_id)

        elif c:
            camd_only.append(bug_id)

        elif f:
            flexfl_only.append(bug_id)

        else:
            neither.append(bug_id)

    return {
        "camd": camd_metrics,
        "flexfl": flexfl_metrics,

        "pairwise_top1": {
            "both_correct": len(both),
            "camd_only_correct": len(camd_only),
            "flexfl_only_correct": len(flexfl_only),
            "neither_correct": len(neither),

            "camd_only_cases": camd_only,
            "flexfl_only_cases": flexfl_only,

            "mcnemar_exact_two_sided_p":
                exact_mcnemar(
                    len(camd_only),
                    len(flexfl_only),
                ),
        },
    }


def pct(value):
    return f"{100 * value:.2f}%"


def print_table(title, result):
    print()
    print(title)
    print("-" * 72)

    print(
        f"{'Method':<16}"
        f"{'N':>6}"
        f"{'Top-1':>15}"
        f"{'Top-3':>15}"
        f"{'Top-5':>15}"
        f"{'MRR@5':>11}"
    )

    for name, m in [
        ("CAMD", result["camd"]),
        ("FlexFL", result["flexfl"]),
    ]:
        print(
            f"{name:<16}"
            f"{m['n']:>6}"
            f"{m['top1_count']:>5}/{m['n']:<4}"
            f"{pct(m['top1']):>6}"
            f"{m['top3_count']:>5}/{m['n']:<4}"
            f"{pct(m['top3']):>6}"
            f"{m['top5_count']:>5}/{m['n']:<4}"
            f"{pct(m['top5']):>6}"
            f"{m['mrr_at_5']:>11.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CAMD against released FlexFL "
            "Defects4J results."
        )
    )

    parser.add_argument(
        "--camd",
        default=(
            "results/verification/final/"
            "final_verifier_summary.json"
        ),
    )

    parser.add_argument(
        "--audit",
        default=(
            "results/baselines/"
            "flexfl_ground_truth_audit.json"
        ),
    )

    parser.add_argument(
        "--flexfl-results",
        default=(
            "external/baselines/flexfl/"
            "Results/Llama3_Defects4J_All"
        ),
    )

    parser.add_argument(
        "--flexfl-gt",
        default=(
            "external/baselines/flexfl/"
            "FlexFL/data/input/ground_truth/"
            "Defects4J/gt.json"
        ),
    )

    parser.add_argument(
        "--flexfl-table4",
        default=(
            "external/baselines/flexfl/"
            "Results/res_Table_4.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/baselines/"
            "flexfl_strict_comparison.json"
        ),
    )

    args = parser.parse_args()

    camd = load_json(Path(args.camd))
    audit = load_json(Path(args.audit))

    flexfl_results_dir = Path(args.flexfl_results)
    flexfl_gt = load_json(Path(args.flexfl_gt))
    official_table4 = load_json(
        Path(args.flexfl_table4)
    )

    # ========================================================
    # CAMD ranks
    # ========================================================

    camd_ranks = {
        record["benchmark_id"]:
            record.get("detector_best_gt_rank")
        for record in camd["records"]
    }

    if len(camd_ranks) != 98:
        raise AssertionError(
            f"Expected 98 CAMD records, "
            f"found {len(camd_ranks)}"
        )

    # ========================================================
    # FlexFL ranks for all published bugs
    # ========================================================

    flexfl_ranks = {}

    raw_files = sorted(
        flexfl_results_dir.glob("*.json")
    )

    for path in raw_files:
        bug_id = path.stem

        if bug_id not in flexfl_gt:
            continue

        result = load_json(path)

        flexfl_ranks[bug_id] = flexfl_rank(
            result,
            flexfl_gt[bug_id],
        )

    # ========================================================
    # Validate our parser against FlexFL's official Table 4.
    # ========================================================

    official_ids = set(flexfl_gt) & set(flexfl_ranks)

    # Official bug universe comes from the available raw results.
    all_flexfl_rank_values = [
        flexfl_ranks[bug_id]
        for bug_id in sorted(official_ids)
    ]

    recomputed_table4 = metrics_at_5(
        all_flexfl_rank_values
    )

    if len(official_ids) != official_table4["Total"]:
        raise AssertionError(
            "FlexFL raw-result count does not reproduce "
            f"official Table 4 total: "
            f"{len(official_ids)} vs "
            f"{official_table4['Total']}"
        )

    if (
        recomputed_table4["top1_count"]
        != official_table4["Top-1"]
    ):
        raise AssertionError(
            "FlexFL Top-1 parser check failed."
        )

    if (
        recomputed_table4["top3_count"]
        != official_table4["Top-3"]
    ):
        raise AssertionError(
            "FlexFL Top-3 parser check failed."
        )

    if (
        recomputed_table4["top5_count"]
        != official_table4["Top-5"]
    ):
        raise AssertionError(
            "FlexFL Top-5 parser check failed."
        )

    official_mrr = float(
        official_table4["MRR"]
    )

    if not math.isclose(
        recomputed_table4["mrr_at_5"],
        official_mrr,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise AssertionError(
            "FlexFL MRR parser check failed: "
            f"{recomputed_table4['mrr_at_5']} "
            f"vs {official_mrr}"
        )

    # ========================================================
    # Primary strict aligned subset = 90
    # ========================================================

    strict_ids = set(
        audit["cases"][
            "exact_identity_set_match"
        ]
    )

    broad_ids = {
        record["benchmark_id"]
        for record in audit["records"]
    }

    if len(strict_ids) != 90:
        raise AssertionError(
            f"Expected 90 strict aligned bugs, "
            f"found {len(strict_ids)}"
        )

    if len(broad_ids) != 98:
        raise AssertionError(
            f"Expected 98 common bugs, "
            f"found {len(broad_ids)}"
        )

    for bug_id in broad_ids:
        if bug_id not in camd_ranks:
            raise RuntimeError(
                f"Missing CAMD rank: {bug_id}"
            )

        if bug_id not in flexfl_ranks:
            raise RuntimeError(
                f"Missing FlexFL rank: {bug_id}"
            )

    strict_result = compare(
        strict_ids,
        camd_ranks,
        flexfl_ranks,
    )

    broad_result = compare(
        broad_ids,
        camd_ranks,
        flexfl_ranks,
    )

    # ========================================================
    # Per-project strict comparison
    # ========================================================

    per_project = {}

    for project in [
        "Chart",
        "Lang",
        "Math",
        "Mockito",
        "Time",
    ]:
        project_ids = {
            bug_id
            for bug_id in strict_ids
            if bug_id.startswith(project + "-")
        }

        per_project[project] = compare(
            project_ids,
            camd_ranks,
            flexfl_ranks,
        )

    # ========================================================
    # Write results
    # ========================================================

    output = {
        "baseline": (
            "FlexFL Llama3-8B-Instruct "
            "released Defects4J v2.0 results"
        ),

        "metric_note": (
            "FlexFL publishes final Top-5 results. "
            "MRR is therefore evaluated as MRR@5 for "
            "both systems in this comparison."
        ),

        "official_parser_validation": {
            "total": len(official_ids),
            "top1": (
                recomputed_table4["top1_count"]
            ),
            "top3": (
                recomputed_table4["top3_count"]
            ),
            "top5": (
                recomputed_table4["top5_count"]
            ),
            "mrr_at_5": (
                recomputed_table4["mrr_at_5"]
            ),
            "status": "PASS",
        },

        "strict_exact_gt_subset": {
            "n": len(strict_ids),
            "ids": sorted(strict_ids),
            **strict_result,
        },

        "broad_common_bug_sensitivity": {
            "n": len(broad_ids),
            "ids": sorted(broad_ids),
            **broad_result,
        },

        "per_project_strict": per_project,
    }

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Human-readable output
    # ========================================================

    print()
    print("CAMD vs FlexFL Final Baseline Evaluation")
    print("=" * 72)

    print()
    print("FlexFL Official-Artifact Parser Validation")
    print("-" * 72)

    print(
        f"Total:                         "
        f"{len(official_ids)}"
    )

    print(
        f"Top-1:                         "
        f"{recomputed_table4['top1_count']}"
    )

    print(
        f"Top-3:                         "
        f"{recomputed_table4['top3_count']}"
    )

    print(
        f"Top-5:                         "
        f"{recomputed_table4['top5_count']}"
    )

    print(
        f"MRR@5:                         "
        f"{recomputed_table4['mrr_at_5']:.4f}"
    )

    print(
        "Official Table 4 validation:   PASS"
    )

    print_table(
        "PRIMARY: Strict Exact-GT Subset",
        strict_result,
    )

    pair = strict_result["pairwise_top1"]

    print()
    print("Strict Subset Top-1 Pairwise")
    print("-" * 72)

    print(
        f"Both correct:                  "
        f"{pair['both_correct']}"
    )

    print(
        f"CAMD only correct:             "
        f"{pair['camd_only_correct']}"
    )

    print(
        f"FlexFL only correct:           "
        f"{pair['flexfl_only_correct']}"
    )

    print(
        f"Neither correct:               "
        f"{pair['neither_correct']}"
    )

    print(
        f"Exact McNemar p-value:         "
        f"{pair['mcnemar_exact_two_sided_p']:.4f}"
    )

    print_table(
        "SENSITIVITY: All 98 Common Bugs",
        broad_result,
    )

    print()
    print("Per Project: Strict Exact-GT Subset")
    print("-" * 72)

    print(
        f"{'Project':<10}"
        f"{'N':>5}"
        f"{'CAMD@1':>12}"
        f"{'FlexFL@1':>12}"
    )

    for project, result in per_project.items():
        n = result["camd"]["n"]

        print(
            f"{project:<10}"
            f"{n:>5}"
            f"{result['camd']['top1_count']:>6}/{n:<5}"
            f"{result['flexfl']['top1_count']:>6}/{n:<5}"
        )

    print()
    print(f"Saved: {output_path}")
    print()


if __name__ == "__main__":
    main()
