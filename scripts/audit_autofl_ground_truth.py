#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_bug_id(autofl_bug_id: str) -> str:
    return autofl_bug_id.replace("_", "-", 1)


def camd_method_identity(gt):
    class_name = str(gt.get("class_name", "")).strip()
    method_name = str(gt.get("method_name", "")).strip()

    if not class_name or not method_name:
        return None

    return f"{class_name}.{method_name}"


def parse_autofl_method(signature: str):
    signature = signature.strip()

    if "(" in signature:
        prefix = signature.split("(", 1)[0]
    else:
        prefix = signature

    prefix = prefix.strip()

    if "." not in prefix:
        return prefix

    class_name, method_name = prefix.rsplit(".", 1)

    return f"{class_name.strip()}.{method_name.strip()}"


def sort_bug_ids(ids):
    def key(value):
        project, bug = value.rsplit("-", 1)
        return project, int(bug)

    return sorted(ids, key=key)


def main():
    parser = argparse.ArgumentParser()

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
        "--autofl",
        default=(
            "external/baselines/autofl/"
            "combined_fl_results/"
            "d4j_gpt4_results_R2.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/baselines/"
            "autofl_ground_truth_audit.json"
        ),
    )

    args = parser.parse_args()

    pools = load_json(Path(args.camd_pools))
    camd_eval = load_json(Path(args.camd_eval))
    autofl = load_json(Path(args.autofl))

    # ========================================================
    # Formal CAMD evaluation universe: 98 applicable bugs
    # ========================================================

    applicable_ids = {
        record["benchmark_id"]
        for record in camd_eval["records"]
    }

    # ========================================================
    # CAMD method-level GT
    # ========================================================

    camd_by_bug = {}

    for record in pools["records"]:
        bug_id = record["benchmark_id"]

        if bug_id not in applicable_ids:
            continue

        identities = set()

        for gt in record.get("ground_truth", []):
            identity = camd_method_identity(gt)

            if identity:
                identities.add(identity)

        camd_by_bug[bug_id] = identities

    # Ensure all formal evaluation bugs are represented.
    missing_pool_records = (
        applicable_ids - set(camd_by_bug)
    )

    if missing_pool_records:
        raise RuntimeError(
            "Applicable bugs missing from candidate pools: "
            + ", ".join(sort_bug_ids(missing_pool_records))
        )

    # ========================================================
    # AutoFL method-level GT
    # ========================================================

    autofl_by_bug = {}

    for raw_bug_id, methods in autofl["buggy_methods"].items():
        bug_id = normalize_bug_id(raw_bug_id)

        identities = {
            parse_autofl_method(signature)
            for signature in methods
        }

        autofl_by_bug[bug_id] = identities

    # ========================================================
    # Common subset
    # ========================================================

    camd_ids = set(camd_by_bug)
    autofl_ids = set(autofl_by_bug)

    common_ids = sort_bug_ids(
        camd_ids & autofl_ids
    )

    camd_missing_autofl = sort_bug_ids(
        camd_ids - autofl_ids
    )

    # ========================================================
    # Compare GT sets
    # ========================================================

    categories = {
        "exact_identity_set_match": [],
        "camd_subset_of_autofl": [],
        "autofl_subset_of_camd": [],
        "partial_overlap": [],
        "disjoint": [],
    }

    records = []

    for bug_id in common_ids:
        camd_set = camd_by_bug[bug_id]
        autofl_set = autofl_by_bug[bug_id]

        overlap = camd_set & autofl_set

        if camd_set == autofl_set:
            category = "exact_identity_set_match"

        elif camd_set and camd_set < autofl_set:
            category = "camd_subset_of_autofl"

        elif autofl_set and autofl_set < camd_set:
            category = "autofl_subset_of_camd"

        elif overlap:
            category = "partial_overlap"

        else:
            category = "disjoint"

        categories[category].append(bug_id)

        records.append(
            {
                "benchmark_id": bug_id,
                "category": category,

                "camd_ground_truth": sorted(camd_set),
                "autofl_ground_truth": sorted(autofl_set),

                "overlap": sorted(overlap),

                "camd_only": sorted(
                    camd_set - autofl_set
                ),

                "autofl_only": sorted(
                    autofl_set - camd_set
                ),
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

    exact = categories["exact_identity_set_match"]

    # ========================================================
    # Output
    # ========================================================

    result = {
        "evaluation_universe": {
            "camd_method_applicable": len(applicable_ids),
            "autofl_released": len(autofl_ids),
            "common": len(common_ids),
            "camd_missing_autofl": camd_missing_autofl,
        },

        "comparison_level": (
            "fully-qualified class + method identity"
        ),

        "limitation": (
            "CAMD frozen annotations do not preserve "
            "parameter signatures. Overload-level equivalence "
            "therefore cannot be independently audited."
        ),

        "summary": {
            "exact_identity_set_match": len(exact),

            "camd_subset_of_autofl":
                len(categories["camd_subset_of_autofl"]),

            "autofl_subset_of_camd":
                len(categories["autofl_subset_of_camd"]),

            "partial_overlap":
                len(categories["partial_overlap"]),

            "disjoint":
                len(categories["disjoint"]),

            "compatible_any_overlap":
                len(compatible),

            "incompatible_no_overlap":
                len(incompatible),
        },

        "cases": categories,

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
    print("CAMD vs AutoFL Ground-Truth Audit")
    print("=" * 60)

    print()
    print("Evaluation Universe")
    print("-" * 60)

    print(
        f"CAMD method-applicable bugs:   "
        f"{len(applicable_ids)}"
    )

    print(
        f"AutoFL released bugs:          "
        f"{len(autofl_ids)}"
    )

    print(
        f"Common applicable bugs:        "
        f"{len(common_ids)}"
    )

    print(
        f"CAMD bugs missing AutoFL:      "
        f"{len(camd_missing_autofl)}"
    )

    print()
    print("Ground-Truth Alignment")
    print("-" * 60)

    print(
        f"Exact identity-set match:      "
        f"{len(exact)}"
    )

    print(
        f"CAMD subset of AutoFL:         "
        f"{len(categories['camd_subset_of_autofl'])}"
    )

    print(
        f"AutoFL subset of CAMD:         "
        f"{len(categories['autofl_subset_of_camd'])}"
    )

    print(
        f"Partial overlap:               "
        f"{len(categories['partial_overlap'])}"
    )

    print(
        f"Disjoint / no overlap:         "
        f"{len(categories['disjoint'])}"
    )

    print()
    print(
        f"Compatible (any overlap):      "
        f"{len(compatible)}/{len(common_ids)}"
    )

    print(
        f"Strict exact-match subset:     "
        f"{len(exact)}/{len(common_ids)}"
    )

    print()
    print("Non-exact cases")
    print("-" * 60)

    for category in [
        "camd_subset_of_autofl",
        "autofl_subset_of_camd",
        "partial_overlap",
        "disjoint",
    ]:
        ids = categories[category]

        if ids:
            print(
                f"{category}:"
            )
            print(
                "  " + ", ".join(ids)
            )

    print()
    print(
        "Important: comparison is at fully-qualified "
        "class + method-name level."
    )

    print(
        "Parameter signatures are unavailable in CAMD's "
        "frozen ground truth."
    )

    print()
    print(f"Saved: {output_path}")
    print()


if __name__ == "__main__":
    main()
