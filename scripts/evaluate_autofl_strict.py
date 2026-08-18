#!/usr/bin/env python3

import json
import math
from pathlib import Path


CAMD_PATH = Path(
    "results/verification/final/final_verifier_summary.json"
)

AUDIT_PATH = Path(
    "results/baselines/autofl_ground_truth_audit.json"
)

AUTOFL_R1_PATH = Path(
    "external/baselines/autofl/"
    "combined_fl_results/d4j_gpt4_results_R1.json"
)

AUTOFL_R2_PATH = Path(
    "external/baselines/autofl/"
    "combined_fl_results/d4j_gpt4_results_R2.json"
)

OUTPUT_PATH = Path(
    "results/baselines/autofl_strict_comparison.json"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_autofl_bug_id(raw):
    return raw.replace("_", "-", 1)


def reciprocal_rank(rank):
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / rank


def metrics(ranks):
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
        "top10_count": topk(10),
        "top10": topk(10) / n if n else 0.0,
        "mrr": (
            sum(reciprocal_rank(rank) for rank in ranks) / n
            if n else 0.0
        ),
    }


def load_autofl_ranks(data):
    result = {}

    for raw_bug_id, methods in data["buggy_methods"].items():
        bug_id = normalize_autofl_bug_id(raw_bug_id)

        ranks = []

        for method_info in methods.values():
            rank = method_info.get("autofl_rank")

            if isinstance(rank, int) and rank > 0:
                ranks.append(rank)

        result[bug_id] = min(ranks) if ranks else None

    return result


def exact_mcnemar(b, c):
    """
    Exact two-sided McNemar test.

    b = CAMD correct, AutoFL wrong
    c = CAMD wrong, AutoFL correct
    """
    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)

    tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return min(1.0, 2.0 * tail)


def compare(ids, camd_ranks, autofl_ranks):
    records = []

    for bug_id in ids:
        camd_rank = camd_ranks.get(bug_id)
        autofl_rank = autofl_ranks.get(bug_id)

        records.append(
            {
                "benchmark_id": bug_id,
                "camd_rank": camd_rank,
                "autofl_rank": autofl_rank,
            }
        )

    camd_values = [
        r["camd_rank"]
        for r in records
    ]

    autofl_values = [
        r["autofl_rank"]
        for r in records
    ]

    camd_metrics = metrics(camd_values)
    autofl_metrics = metrics(autofl_values)

    both_correct = []
    camd_only = []
    autofl_only = []
    neither = []

    for r in records:
        camd_correct = r["camd_rank"] == 1
        autofl_correct = r["autofl_rank"] == 1

        bug_id = r["benchmark_id"]

        if camd_correct and autofl_correct:
            both_correct.append(bug_id)

        elif camd_correct:
            camd_only.append(bug_id)

        elif autofl_correct:
            autofl_only.append(bug_id)

        else:
            neither.append(bug_id)

    p_value = exact_mcnemar(
        len(camd_only),
        len(autofl_only),
    )

    return {
        "camd": camd_metrics,
        "autofl": autofl_metrics,

        "pairwise_top1": {
            "both_correct": len(both_correct),
            "camd_only_correct": len(camd_only),
            "autofl_only_correct": len(autofl_only),
            "neither_correct": len(neither),

            "camd_only_cases": camd_only,
            "autofl_only_cases": autofl_only,

            "mcnemar_exact_two_sided_p": p_value,
        },

        "records": records,
    }


def pct(v):
    return f"{100 * v:.2f}%"


def print_table(title, result_r1, result_r2):
    print()
    print(title)
    print("-" * 82)

    print(
        f"{'Method':<18}"
        f"{'N':>5}"
        f"{'Top-1':>13}"
        f"{'Top-3':>13}"
        f"{'Top-5':>13}"
        f"{'Top-10':>13}"
        f"{'MRR':>9}"
    )

    methods = [
        ("CAMD", result_r1["camd"]),
        ("AutoFL R1", result_r1["autofl"]),
        ("AutoFL R2", result_r2["autofl"]),
    ]

    for name, m in methods:
        print(
            f"{name:<18}"
            f"{m['n']:>5}"
            f"{m['top1_count']:>4}/{m['n']:<3}"
            f"{pct(m['top1']):>6}"
            f"{m['top3_count']:>4}/{m['n']:<3}"
            f"{pct(m['top3']):>6}"
            f"{m['top5_count']:>4}/{m['n']:<3}"
            f"{pct(m['top5']):>6}"
            f"{m['top10_count']:>4}/{m['n']:<3}"
            f"{pct(m['top10']):>6}"
            f"{m['mrr']:>9.4f}"
        )


def print_pairwise(name, result):
    p = result["pairwise_top1"]

    print()
    print(name)
    print("-" * 60)

    print(
        f"Both Top-1 correct:          "
        f"{p['both_correct']}"
    )

    print(
        f"CAMD only Top-1 correct:     "
        f"{p['camd_only_correct']}"
    )

    print(
        f"AutoFL only Top-1 correct:   "
        f"{p['autofl_only_correct']}"
    )

    print(
        f"Neither Top-1 correct:       "
        f"{p['neither_correct']}"
    )

    print(
        f"Exact McNemar p-value:       "
        f"{p['mcnemar_exact_two_sided_p']:.4f}"
    )


def main():
    camd = load_json(CAMD_PATH)
    audit = load_json(AUDIT_PATH)
    autofl_r1 = load_json(AUTOFL_R1_PATH)
    autofl_r2 = load_json(AUTOFL_R2_PATH)

    camd_ranks = {
        record["benchmark_id"]:
            record.get("detector_best_gt_rank")
        for record in camd["records"]
    }

    r1_ranks = load_autofl_ranks(autofl_r1)
    r2_ranks = load_autofl_ranks(autofl_r2)

    # ========================================================
    # Strict subset:
    # exact CAMD / AutoFL faulty-method identity-set match
    # ========================================================

    strict_ids = audit["cases"][
        "exact_identity_set_match"
    ]

    strict_ids = sorted(strict_ids)

    # ========================================================
    # Broader sensitivity subset:
    # all common method-applicable bugs
    # ========================================================

    broad_ids = [
        record["benchmark_id"]
        for record in audit["records"]
    ]

    broad_ids = sorted(broad_ids)

    assert len(camd_ranks) == 98
    assert len(strict_ids) == 68
    assert len(broad_ids) == 78

    for bug_id in strict_ids:
        if bug_id not in camd_ranks:
            raise RuntimeError(
                f"Missing CAMD rank: {bug_id}"
            )

        if bug_id not in r1_ranks:
            raise RuntimeError(
                f"Missing AutoFL R1 rank: {bug_id}"
            )

        if bug_id not in r2_ranks:
            raise RuntimeError(
                f"Missing AutoFL R2 rank: {bug_id}"
            )

    strict_r1 = compare(
        strict_ids,
        camd_ranks,
        r1_ranks,
    )

    strict_r2 = compare(
        strict_ids,
        camd_ranks,
        r2_ranks,
    )

    broad_r1 = compare(
        broad_ids,
        camd_ranks,
        r1_ranks,
    )

    broad_r2 = compare(
        broad_ids,
        camd_ranks,
        r2_ranks,
    )

    result = {
        "protocol": {
            "primary_subset": (
                "Exact method-level ground-truth identity-set "
                "alignment between CAMD and AutoFL."
            ),

            "comparison_level": (
                "Fully-qualified class + method name."
            ),

            "signature_limitation": (
                "CAMD frozen annotations do not preserve "
                "parameter signatures, so overload-level "
                "equivalence cannot be independently audited."
            ),

            "strict_n": len(strict_ids),
            "broad_common_n": len(broad_ids),
        },

        "strict_exact_gt_subset": {
            "ids": strict_ids,

            "autofl_r1": strict_r1,
            "autofl_r2": strict_r2,
        },

        "broad_common_bug_sensitivity": {
            "ids": broad_ids,

            "autofl_r1": broad_r1,
            "autofl_r2": broad_r2,
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("CAMD vs AutoFL Final Baseline Evaluation")
    print("=" * 82)

    print()
    print(
        "Primary protocol: exact method-level "
        "ground-truth identity-set alignment."
    )

    print(
        f"Strict aligned bugs:          "
        f"{len(strict_ids)}"
    )

    print(
        f"Broader common bugs:          "
        f"{len(broad_ids)}"
    )

    print_table(
        "PRIMARY: Strict Exact-GT Subset",
        strict_r1,
        strict_r2,
    )

    print_pairwise(
        "Strict subset: CAMD vs AutoFL R1 Top-1",
        strict_r1,
    )

    print_pairwise(
        "Strict subset: CAMD vs AutoFL R2 Top-1",
        strict_r2,
    )

    print_table(
        "SENSITIVITY: All Common Bugs",
        broad_r1,
        broad_r2,
    )

    print()
    print(
        "Saved: "
        f"{OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()
