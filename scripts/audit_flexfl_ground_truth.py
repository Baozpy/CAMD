#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def camd_method_identity(gt):
    """
    CAMD frozen GT contains:
        class_name
        method_name

    We therefore audit at fully-qualified class + method-name level.
    """
    class_name = str(gt.get("class_name", "")).strip()
    method_name = str(gt.get("method_name", "")).strip()

    if not class_name or not method_name:
        return None

    return f"{class_name}.{method_name}"


def flexfl_method_identity(signature):
    """
    FlexFL GT examples:
        org.apache.commons.math3.distribution.DiscreteDistribution.sample(int)

    Normalize to:
        org.apache.commons.math3.distribution.DiscreteDistribution.sample

    because CAMD does not preserve parameter signatures.
    """
    signature = str(signature).strip()

    if "(" in signature:
        signature = signature.split("(", 1)[0]

    return signature.strip()


def sort_bug_ids(ids):
    def key(value):
        try:
            project, bug = value.rsplit("-", 1)
            return project, int(bug)
        except Exception:
            return value, 0

    return sorted(ids, key=key)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit method-level ground-truth alignment "
            "between CAMD and FlexFL."
        )
    )

    parser.add_argument(
        "--camd-pools",
        default=(
            "results/defects4j/"
            "fse_ase_frozen_candidate_pools.json"
        ),
    )

    parser.add_argument(
        "--camd-eval",
        default=(
            "results/verification/final/"
            "final_verifier_summary.json"
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
        "--flexfl-results",
        default=(
            "external/baselines/flexfl/"
            "Results/Llama3_Defects4J_All"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/baselines/"
            "flexfl_ground_truth_audit.json"
        ),
    )

    args = parser.parse_args()

    camd_pools = load_json(Path(args.camd_pools))
    camd_eval = load_json(Path(args.camd_eval))
    flexfl_gt_raw = load_json(Path(args.flexfl_gt))

    flexfl_results_dir = Path(args.flexfl_results)

    if not flexfl_results_dir.exists():
        raise FileNotFoundError(
            f"FlexFL results directory not found: "
            f"{flexfl_results_dir}"
        )

    # ========================================================
    # Formal CAMD evaluation universe: 98 bugs
    # ========================================================

    applicable_ids = {
        record["benchmark_id"]
        for record in camd_eval["records"]
    }

    if len(applicable_ids) != 98:
        raise AssertionError(
            f"Expected 98 CAMD method-applicable bugs, "
            f"found {len(applicable_ids)}"
        )

    # ========================================================
    # CAMD method-level GT
    # ========================================================

    camd_gt = {}

    malformed_camd = []

    for record in camd_pools["records"]:
        bug_id = record["benchmark_id"]

        if bug_id not in applicable_ids:
            continue

        identities = set()

        for gt in record.get("ground_truth", []):
            identity = camd_method_identity(gt)

            if identity:
                identities.add(identity)
            else:
                malformed_camd.append(
                    {
                        "benchmark_id": bug_id,
                        "ground_truth": gt,
                    }
                )

        camd_gt[bug_id] = identities

    missing_camd_gt = applicable_ids - set(camd_gt)

    if missing_camd_gt:
        raise RuntimeError(
            "Missing CAMD GT records: "
            + ", ".join(sort_bug_ids(missing_camd_gt))
        )

    # ========================================================
    # FlexFL available result universe
    # ========================================================

    flexfl_result_ids = {
        path.stem
        for path in flexfl_results_dir.glob("*.json")
    }

    common_result_ids = (
        applicable_ids & flexfl_result_ids
    )

    # ========================================================
    # FlexFL GT
    # ========================================================

    flexfl_gt = {}

    for bug_id, methods in flexfl_gt_raw.items():
        identities = {
            flexfl_method_identity(method)
            for method in methods
        }

        flexfl_gt[bug_id] = identities

    common_ids = (
        applicable_ids
        & flexfl_result_ids
        & set(flexfl_gt)
    )

    common_ids = sort_bug_ids(common_ids)

    # ========================================================
    # Compare GT sets
    # ========================================================

    categories = {
        "exact_identity_set_match": [],
        "camd_subset_of_flexfl": [],
        "flexfl_subset_of_camd": [],
        "partial_overlap": [],
        "disjoint": [],
    }

    records = []

    for bug_id in common_ids:
        camd_set = camd_gt[bug_id]
        flexfl_set = flexfl_gt[bug_id]

        overlap = camd_set & flexfl_set

        if camd_set == flexfl_set:
            category = "exact_identity_set_match"

        elif camd_set and camd_set < flexfl_set:
            category = "camd_subset_of_flexfl"

        elif flexfl_set and flexfl_set < camd_set:
            category = "flexfl_subset_of_camd"

        elif overlap:
            category = "partial_overlap"

        else:
            category = "disjoint"

        categories[category].append(bug_id)

        records.append(
            {
                "benchmark_id": bug_id,
                "category": category,

                "camd_ground_truth":
                    sorted(camd_set),

                "flexfl_ground_truth":
                    sorted(flexfl_set),

                "overlap":
                    sorted(overlap),

                "camd_only":
                    sorted(camd_set - flexfl_set),

                "flexfl_only":
                    sorted(flexfl_set - camd_set),
            }
        )

    compatible = [
        record["benchmark_id"]
        for record in records
        if record["overlap"]
    ]

    incompatible = [
        record["benchmark_id"]
        for record in records
        if not record["overlap"]
    ]

    exact = categories[
        "exact_identity_set_match"
    ]

    # ========================================================
    # Per-project coverage / alignment
    # ========================================================

    projects = [
        "Chart",
        "Lang",
        "Math",
        "Mockito",
        "Time",
    ]

    per_project = {}

    for project in projects:
        project_records = [
            record
            for record in records
            if record["benchmark_id"].startswith(
                project + "-"
            )
        ]

        per_project[project] = {
            "common": len(project_records),

            "exact": sum(
                r["category"]
                == "exact_identity_set_match"
                for r in project_records
            ),

            "compatible": sum(
                bool(r["overlap"])
                for r in project_records
            ),

            "disjoint": sum(
                r["category"] == "disjoint"
                for r in project_records
            ),
        }

    # ========================================================
    # Output
    # ========================================================

    result = {
        "evaluation_universe": {
            "camd_method_applicable":
                len(applicable_ids),

            "flexfl_raw_results":
                len(flexfl_result_ids),

            "camd_with_flexfl_result":
                len(common_result_ids),

            "common_with_ground_truth":
                len(common_ids),
        },

        "comparison_level": (
            "fully-qualified class + method identity"
        ),

        "limitation": (
            "CAMD frozen annotations do not preserve Java "
            "parameter signatures. FlexFL signatures are "
            "therefore normalized to fully-qualified class "
            "and method names. Overload-level equivalence "
            "cannot be independently audited."
        ),

        "summary": {
            "exact_identity_set_match":
                len(exact),

            "camd_subset_of_flexfl":
                len(
                    categories[
                        "camd_subset_of_flexfl"
                    ]
                ),

            "flexfl_subset_of_camd":
                len(
                    categories[
                        "flexfl_subset_of_camd"
                    ]
                ),

            "partial_overlap":
                len(
                    categories[
                        "partial_overlap"
                    ]
                ),

            "disjoint":
                len(
                    categories[
                        "disjoint"
                    ]
                ),

            "compatible_any_overlap":
                len(compatible),

            "incompatible_no_overlap":
                len(incompatible),

            "malformed_camd_gt_entries":
                len(malformed_camd),
        },

        "cases": categories,

        "per_project": per_project,

        "records": records,
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
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Human-readable report
    # ========================================================

    print()
    print("CAMD vs FlexFL Ground-Truth Audit")
    print("=" * 64)

    print()
    print("Evaluation Universe")
    print("-" * 64)

    print(
        f"CAMD method-applicable bugs:    "
        f"{len(applicable_ids)}"
    )

    print(
        f"FlexFL raw result files:        "
        f"{len(flexfl_result_ids)}"
    )

    print(
        f"CAMD bugs with FlexFL result:   "
        f"{len(common_result_ids)}"
    )

    print(
        f"Common bugs with GT:            "
        f"{len(common_ids)}"
    )

    print()
    print("Ground-Truth Alignment")
    print("-" * 64)

    print(
        f"Exact identity-set match:       "
        f"{len(exact)}"
    )

    print(
        f"CAMD subset of FlexFL:          "
        f"{len(categories['camd_subset_of_flexfl'])}"
    )

    print(
        f"FlexFL subset of CAMD:          "
        f"{len(categories['flexfl_subset_of_camd'])}"
    )

    print(
        f"Partial overlap:                "
        f"{len(categories['partial_overlap'])}"
    )

    print(
        f"Disjoint / no overlap:          "
        f"{len(categories['disjoint'])}"
    )

    print()
    print(
        f"Compatible (any overlap):       "
        f"{len(compatible)}/{len(common_ids)}"
    )

    print(
        f"Strict exact-match subset:      "
        f"{len(exact)}/{len(common_ids)}"
    )

    print()
    print("Per Project")
    print("-" * 64)

    print(
        f"{'Project':<10}"
        f"{'Common':>8}"
        f"{'Exact':>8}"
        f"{'Compat.':>10}"
        f"{'Disjoint':>10}"
    )

    for project, stats in per_project.items():
        print(
            f"{project:<10}"
            f"{stats['common']:>8}"
            f"{stats['exact']:>8}"
            f"{stats['compatible']:>10}"
            f"{stats['disjoint']:>10}"
        )

    print()
    print("Non-exact cases")
    print("-" * 64)

    any_nonexact = False

    for category in [
        "camd_subset_of_flexfl",
        "flexfl_subset_of_camd",
        "partial_overlap",
        "disjoint",
    ]:
        ids = categories[category]

        if ids:
            any_nonexact = True
            print(f"{category}:")
            print("  " + ", ".join(ids))

    if not any_nonexact:
        print("None.")

    print()
    print(
        "Important: comparison is at fully-qualified "
        "class + method-name level."
    )

    print(
        "CAMD does not preserve parameter signatures, "
        "so overload-level equivalence is not audited."
    )

    print()
    print(f"Saved: {output_path}")
    print()


if __name__ == "__main__":
    main()
