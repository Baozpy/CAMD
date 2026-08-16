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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-input",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "finetuning"
        / "train.jsonl",
    )

    parser.add_argument(
        "--validation-input",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "finetuning"
        / "validation.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "finetuning"
        / "pairwise",
    )

    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def print_summary(
    name: str,
    summary: dict,
) -> None:
    print()
    print(name)
    print(f"  Bugs: {summary['bugs']}")
    print(f"  Pairs: {summary['pairs']}")
    print(
        "  Preferred A: "
        f"{summary['preferred_a']}"
    )
    print(
        "  Preferred B: "
        f"{summary['preferred_b']}"
    )
    print(
        "  Preferred A ratio: "
        f"{summary['preferred_a_ratio']:.4f}"
    )
    print(
        f"  Bug IDs: {summary['bug_ids']}"
    )


def main() -> None:
    args = parse_args()

    train_samples = load_jsonl(
        args.train_input
    )
    validation_samples = load_jsonl(
        args.validation_input
    )

    train_pairs = build_pairwise_dataset(
        train_samples,
        negatives_per_positive=(
            args.negatives_per_positive
        ),
        seed=args.seed,
    )

    validation_pairs = build_pairwise_dataset(
        validation_samples,
        negatives_per_positive=(
            args.negatives_per_positive
        ),
        seed=args.seed + 1,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_output = (
        args.output_dir / "train.jsonl"
    )
    validation_output = (
        args.output_dir / "validation.jsonl"
    )

    save_jsonl(
        train_pairs,
        train_output,
    )
    save_jsonl(
        validation_pairs,
        validation_output,
    )

    train_summary = summarize_pairs(
        train_pairs
    )
    validation_summary = summarize_pairs(
        validation_pairs
    )

    manifest = {
        "task": "pairwise_defect_ranking",
        "negatives_per_positive": (
            args.negatives_per_positive
        ),
        "seed": args.seed,
        "train": train_summary,
        "validation": validation_summary,
    }

    manifest_path = (
        args.output_dir
        / "dataset_manifest.json"
    )

    with manifest_path.open(
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
        "Pairwise Ranking Dataset Summary"
    )
    print("=" * 100)

    print_summary(
        "Train",
        train_summary,
    )

    print_summary(
        "Validation",
        validation_summary,
    )

    print()
    print("Saved:")
    print(train_output)
    print(validation_output)
    print(manifest_path)


if __name__ == "__main__":
    main()