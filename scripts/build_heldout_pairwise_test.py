from __future__ import annotations

import argparse
import json
from pathlib import Path

from camd.finetuning.pairwise_dataset_builder import (
    build_pairwise_dataset,
    load_jsonl,
    save_jsonl,
    summarize_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
    / "heldout_lang_1_20"
    / "test.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
    / "heldout_lang_1_20"
    / "pairwise"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final held-out Lang 1-20 "
            "pairwise defect-ranking test set."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=43,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.negatives_per_positive != 4:
        raise ValueError(
            "Final held-out evaluation must use "
            "--negatives-per-positive 4."
        )

    rows = load_jsonl(
        args.input
    )

    bug_ids = sorted(
        {
            int(row["bug_id"])
            for row in rows
        }
    )

    expected_bug_ids = [
        1,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        19,
        20,
    ]

    if bug_ids != expected_bug_ids:
        raise RuntimeError(
            "Held-out input does not match the "
            "expected Lang 1-20 bug set.\n"
            f"Observed: {bug_ids}\n"
            f"Expected: {expected_bug_ids}"
        )

    pairs = build_pairwise_dataset(
        rows,
        negatives_per_positive=(
            args.negatives_per_positive
        ),
        seed=args.seed,
    )

    summary = summarize_pairs(
        pairs
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_output = (
        args.output_dir
        / "test.jsonl"
    )

    manifest_output = (
        args.output_dir
        / "test_manifest.json"
    )

    save_jsonl(
        pairs,
        test_output,
    )

    manifest = {
        "project": "Lang",
        "purpose": (
            "final_held_out_pairwise_evaluation_only"
        ),
        "source": str(
            args.input
        ),
        "negatives_per_positive": (
            args.negatives_per_positive
        ),
        "seed": args.seed,
        "held_out_policy": (
            "Lang 1-20 are reserved exclusively "
            "for final evaluation. Pairwise test "
            "results must not be used for training, "
            "checkpoint selection, prompt tuning, "
            "aggregation tuning, or hyperparameter tuning."
        ),
        "summary": summary,
    }

    with manifest_output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print(
        "Held-Out Pairwise Test Dataset Summary"
    )
    print("=" * 100)

    print()
    print(
        f"Bugs: "
        f"{summary['bugs']}"
    )

    print(
        f"Pairs: "
        f"{summary['pairs']}"
    )

    print(
        f"Preferred A: "
        f"{summary['preferred_a']}"
    )

    print(
        f"Preferred B: "
        f"{summary['preferred_b']}"
    )

    print(
        "Preferred A ratio: "
        f"{summary['preferred_a_ratio']:.4f}"
    )

    print(
        f"Bug IDs: "
        f"{summary['bug_ids']}"
    )

    print()
    print("Saved:")
    print(test_output)
    print(manifest_output)


if __name__ == "__main__":
    main()