#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_autofl_bug_id(autofl_id: str) -> str:
    """
    AutoFL uses IDs such as:
        Chart_17
        Lang_22

    CAMD uses:
        Chart-17
        Lang-22
    """
    return autofl_id.replace("_", "-", 1)


def reciprocal_rank(rank):
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / rank


def compute_metrics(records, rank_key):
    n = len(records)

    ranks = [record[rank_key] for record in records]

    def top_k(k):
        return sum(
            rank is not None and rank <= k
            for rank in ranks
        )

    return {
        "n": n,
        "top1_count": top_k(1),
        "top1": top_k(1) / n if n else 0.0,
        "top3_count": top_k(3),
        "top3": top_k(3) / n if n else 0.0,
        "top5_count": top_k(5),
        "top5": top_k(5) / n if n else 0.0,
        "top10_count": top_k(10),
        "top10": top_k(10) / n if n else 0.0,
        "mrr": (
            sum(reciprocal_rank(rank) for rank in ranks) / n
            if n else 0.0
        ),
    }


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare CAMD Detector with released AutoFL results "
            "on the exact common Defects4J bug subset."
        )
    )

    parser.add_argument(
        "--camd",
        default=(
            "results/verification/final/"
            "final_verifier_summary.json"
        ),
        help="CAMD frozen verifier summary",
    )

    parser.add_argument(
        "--autofl",
        default=(
            "external/baselines/autofl/"
            "combined_fl_results/"
            "d4j_gpt4_results_R2.json"
        ),
        help="AutoFL released result JSON",
    )

    parser.add_argument(
        "--output",
        default=(
            "results/baselines/"
            "autofl_gpt4_r2_common.json"
        ),
        help="Output JSON path",
    )

    args = parser.parse_args()

    camd_path = Path(args.camd)
    autofl_path = Path(args.autofl)
    output_path = Path(args.output)

    if not camd_path.exists():
        raise FileNotFoundError(
            f"CAMD file not found: {camd_path}"
        )

    if not autofl_path.exists():
        raise FileNotFoundError(
            f"AutoFL file not found: {autofl_path}"
        )

    camd = load_json(camd_path)
    autofl = load_json(autofl_path)

    # --------------------------------------------------------
    # CAMD method-applicable held-out records
    # --------------------------------------------------------

    camd_records = camd["records"]

    camd_by_bug = {
        record["benchmark_id"]: record
        for record in camd_records
    }

    # --------------------------------------------------------
    # AutoFL result map
    #
    # For bugs with multiple faulty methods, localization
    # success is determined by the best-ranked faulty method.
    # --------------------------------------------------------

    autofl_buggy_methods = autofl["buggy_methods"]

    autofl_by_bug = {}

    for autofl_bug_id, methods in autofl_buggy_methods.items():
        bug_id = normalize_autofl_bug_id(autofl_bug_id)

        ranks = []

        for method_name, method_info in methods.items():
            rank = method_info.get("autofl_rank")

            if isinstance(rank, int) and rank > 0:
                ranks.append(rank)

        best_rank = min(ranks) if ranks else None

        autofl_by_bug[bug_id] = {
            "best_gt_rank": best_rank,
            "ground_truth_method_count": len(methods),
            "methods": methods,
        }

    # --------------------------------------------------------
    # Exact common bug subset
    # --------------------------------------------------------

    camd_ids = set(camd_by_bug)
    autofl_ids = set(autofl_by_bug)

    common_ids = sorted(
        camd_ids & autofl_ids,
        key=lambda x: (
            x.split("-")[0],
            int(x.split("-")[1]),
        ),
    )

    camd_only = sorted(
        camd_ids - autofl_ids,
        key=lambda x: (
            x.split("-")[0],
            int(x.split("-")[1]),
        ),
    )

    autofl_only = sorted(
        autofl_ids - camd_ids,
        key=lambda x: (
            x.split("-")[0],
            int(x.split("-")[1]),
        ),
    )

    comparison_records = []

    for bug_id in common_ids:
        camd_record = camd_by_bug[bug_id]
        autofl_record = autofl_by_bug[bug_id]

        comparison_records.append(
            {
                "benchmark_id": bug_id,
                "project": camd_record["project"],
                "camd_rank": (
                    camd_record.get(
                        "detector_best_gt_rank"
                    )
                ),
                "autofl_rank": (
                    autofl_record["best_gt_rank"]
                ),
            }
        )

    # --------------------------------------------------------
    # Metrics on the identical subset
    # --------------------------------------------------------

    camd_metrics = compute_metrics(
        comparison_records,
        "camd_rank",
    )

    autofl_metrics = compute_metrics(
        comparison_records,
        "autofl_rank",
    )

    # --------------------------------------------------------
    # Per-project comparison
    # --------------------------------------------------------

    projects = sorted(
        set(record["project"] for record in comparison_records)
    )

    per_project = {}

    for project in projects:
        subset = [
            record
            for record in comparison_records
            if record["project"] == project
        ]

        per_project[project] = {
            "n": len(subset),
            "camd": compute_metrics(
                subset,
                "camd_rank",
            ),
            "autofl": compute_metrics(
                subset,
                "autofl_rank",
            ),
        }

    # --------------------------------------------------------
    # Pairwise Top-1 transitions
    # --------------------------------------------------------

    camd_only_top1 = []
    autofl_only_top1 = []
    both_top1 = []
    neither_top1 = []

    for record in comparison_records:
        camd_correct = record["camd_rank"] == 1
        autofl_correct = record["autofl_rank"] == 1

        bug_id = record["benchmark_id"]

        if camd_correct and autofl_correct:
            both_top1.append(bug_id)

        elif camd_correct and not autofl_correct:
            camd_only_top1.append(bug_id)

        elif autofl_correct and not camd_correct:
            autofl_only_top1.append(bug_id)

        else:
            neither_top1.append(bug_id)

    result = {
        "baseline": (
            "AutoFL GPT-4 released aggregate R2"
        ),
        "autofl_source_file": str(autofl_path),
        "camd_source_file": str(camd_path),

        "subset": {
            "camd_total": len(camd_ids),
            "autofl_total": len(autofl_ids),
            "common": len(common_ids),
            "camd_only": camd_only,
            "common_ids": common_ids,
        },

        "camd": camd_metrics,
        "autofl": autofl_metrics,

        "top1_pairwise": {
            "both_correct": len(both_top1),
            "camd_only_correct": len(camd_only_top1),
            "autofl_only_correct": len(autofl_only_top1),
            "neither_correct": len(neither_top1),

            "camd_only_cases": camd_only_top1,
            "autofl_only_cases": autofl_only_top1,
        },

        "per_project": per_project,

        "records": comparison_records,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    print()
    print("CAMD vs AutoFL: Common-Subset Evaluation")
    print("=" * 60)

    print()
    print("Subset")
    print("-" * 60)

    print(
        f"CAMD applicable bugs:       {len(camd_ids)}"
    )
    print(
        f"AutoFL released bugs:       {len(autofl_ids)}"
    )
    print(
        f"Common bugs:                {len(common_ids)}"
    )
    print(
        f"CAMD bugs missing AutoFL:   {len(camd_only)}"
    )

    if camd_only:
        print(
            "Missing from AutoFL:        "
            + ", ".join(camd_only)
        )

    print()
    print("Overall Metrics on Exact Common Subset")
    print("-" * 60)

    print(
        f"{'Method':<12}"
        f"{'N':>6}"
        f"{'Top-1':>14}"
        f"{'Top-3':>14}"
        f"{'Top-5':>14}"
        f"{'Top-10':>14}"
        f"{'MRR':>10}"
    )

    for name, metrics in [
        ("CAMD", camd_metrics),
        ("AutoFL", autofl_metrics),
    ]:
        print(
            f"{name:<12}"
            f"{metrics['n']:>6}"
            f"{metrics['top1_count']:>5}/"
            f"{metrics['n']:<4}"
            f"{pct(metrics['top1']):>5}"
            f"{metrics['top3_count']:>5}/"
            f"{metrics['n']:<4}"
            f"{pct(metrics['top3']):>5}"
            f"{metrics['top5_count']:>5}/"
            f"{metrics['n']:<4}"
            f"{pct(metrics['top5']):>5}"
            f"{metrics['top10_count']:>5}/"
            f"{metrics['n']:<4}"
            f"{pct(metrics['top10']):>5}"
            f"{metrics['mrr']:>10.4f}"
        )

    print()
    print("Top-1 Pairwise")
    print("-" * 60)

    print(
        f"Both correct:               {len(both_top1)}"
    )
    print(
        f"CAMD only correct:          {len(camd_only_top1)}"
    )
    print(
        f"AutoFL only correct:        {len(autofl_only_top1)}"
    )
    print(
        f"Neither correct:            {len(neither_top1)}"
    )

    print()
    print("Per Project")
    print("-" * 60)

    print(
        f"{'Project':<10}"
        f"{'N':>5}"
        f"{'CAMD@1':>10}"
        f"{'AutoFL@1':>12}"
    )

    for project, values in per_project.items():
        n = values["n"]

        camd_top1 = values["camd"]["top1_count"]
        autofl_top1 = values["autofl"]["top1_count"]

        print(
            f"{project:<10}"
            f"{n:>5}"
            f"{camd_top1:>5}/{n:<4}"
            f"{autofl_top1:>6}/{n:<4}"
        )

    print()
    print(
        "Saved: "
        f"{output_path}"
    )
    print()


if __name__ == "__main__":
    main()
