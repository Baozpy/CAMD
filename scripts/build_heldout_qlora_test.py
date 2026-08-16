from __future__ import annotations

import argparse
import json
from pathlib import Path

from camd.finetuning.dataset_builder import (
    QLoRADatasetBuilder,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
    / "heldout_lang_1_20"
)

HELD_OUT_LANG_BUGS = list(
    range(1, 21)
)

KNOWN_DEPRECATED_LANG_BUGS = {
    2,
    18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final held-out Lang 1-20 "
            "QLoRA evaluation dataset."
        )
    )

    parser.add_argument(
        "--project",
        default="Lang",
    )

    parser.add_argument(
        "--bug-start",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--bug-end",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--all-negatives",
        action="store_true",
        help=(
            "Keep every negative candidate instead "
            "of deterministic hard-negative sampling."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def sample_to_record(sample) -> dict:
    return {
        "project": sample.project,
        "bug_id": sample.bug_id,
        "class_name": sample.class_name,
        "method_name": sample.method_name,
        "start_line": sample.start_line,
        "end_line": sample.end_line,
        "method_length": sample.method_length,
        "label": sample.label,
        "is_target_defect": sample.is_target_defect,
        "input": sample.input,
        "output": sample.output,
    }


def write_jsonl(
    records: list[dict],
    output_file: Path,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def print_statistics(
    records: list[dict],
) -> None:
    bug_ids = sorted(
        {
            int(record["bug_id"])
            for record in records
        }
    )

    positive = sum(
        int(record["label"])
        for record in records
    )

    negative = (
        len(records)
        - positive
    )

    print()
    print(
        "=" * 100
    )
    print(
        "Held-Out QLoRA Test Dataset Summary"
    )
    print(
        "=" * 100
    )

    print()
    print(
        f"Bugs: {len(bug_ids)}"
    )

    print(
        f"Bug IDs: {bug_ids}"
    )

    print(
        f"Samples: {len(records)}"
    )

    print(
        f"Positive: {positive}"
    )

    print(
        f"Negative: {negative}"
    )

    if records:
        print(
            "Positive ratio: "
            f"{positive / len(records):.4f}"
        )


def main() -> None:
    args = parse_args()

    if args.project != "Lang":
        raise ValueError(
            "This held-out builder is currently "
            "restricted to the Lang project."
        )

    if (
        args.bug_start < 1
        or args.bug_end > 20
    ):
        raise ValueError(
            "Held-out evaluation must remain "
            "within Lang 1-20."
        )

    if args.bug_start > args.bug_end:
        raise ValueError(
            "--bug-start must be <= --bug-end."
        )

    if args.all_negatives:
        max_negatives = None
    else:
        if (
            args.max_negatives_per_positive
            < 1
        ):
            raise ValueError(
                "--max-negatives-per-positive "
                "must be >= 1."
            )

        max_negatives = (
            args.max_negatives_per_positive
        )

    builder = QLoRADatasetBuilder(
        project_root=PROJECT_ROOT,
        max_negatives_per_positive=(
            max_negatives
        ),
    )

    all_records: list[dict] = []

    successful_bug_ids: list[int] = []
    skipped_bug_ids: list[int] = []
    failed: list[dict] = []

    for bug_id in range(
        args.bug_start,
        args.bug_end + 1,
    ):
        if (
            bug_id
            in KNOWN_DEPRECATED_LANG_BUGS
        ):
            print()
            print(
                f"Skipping Lang-{bug_id}: "
                "deprecated."
            )

            skipped_bug_ids.append(
                bug_id
            )

            continue

        if (
            bug_id
            not in HELD_OUT_LANG_BUGS
        ):
            raise RuntimeError(
                f"Lang-{bug_id} is not part "
                "of the held-out Lang 1-20 set."
            )

        print()
        print(
            "=" * 100
        )

        print(
            f"Building held-out samples: "
            f"Lang-{bug_id}"
        )

        print(
            "=" * 100
        )

        try:
            samples = (
                builder.build_bug_samples(
                    project="Lang",
                    bug_id=bug_id,
                )
            )

            records = [
                sample_to_record(
                    sample
                )
                for sample in samples
            ]

            positive = sum(
                int(
                    record["label"]
                )
                for record in records
            )

            negative = (
                len(records)
                - positive
            )

            print(
                f"Samples: "
                f"{len(records)}"
            )

            print(
                f"Positive: "
                f"{positive}"
            )

            print(
                f"Negative: "
                f"{negative}"
            )

            if records:
                print(
                    "Positive ratio: "
                    f"{positive / len(records):.4f}"
                )

            all_records.extend(
                records
            )

            successful_bug_ids.append(
                bug_id
            )

        except Exception as exc:
            print()
            print(
                f"FAILED: Lang-{bug_id}"
            )

            print(exc)

            failed.append(
                {
                    "bug_id": bug_id,
                    "error": str(exc),
                }
            )

    if not all_records:
        print()
        print(
            "No held-out samples were built."
        )
        return

    all_records.sort(
        key=lambda record: (
            int(
                record["bug_id"]
            ),
            record["class_name"],
            int(
                record["start_line"]
            ),
            int(
                record["end_line"]
            ),
            record["method_name"],
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_file = (
        args.output_dir
        / "test.jsonl"
    )

    manifest_file = (
        args.output_dir
        / "test_manifest.json"
    )

    write_jsonl(
        all_records,
        test_file,
    )

    positive = sum(
        int(record["label"])
        for record in all_records
    )

    negative = (
        len(all_records)
        - positive
    )

    manifest = {
        "project": "Lang",
        "purpose": (
            "final_held_out_evaluation_only"
        ),
        "requested_range": {
            "start": args.bug_start,
            "end": args.bug_end,
        },
        "held_out_policy": (
            "Lang 1-20 are reserved exclusively "
            "for final evaluation. These samples "
            "must not be used for training, "
            "checkpoint selection, threshold "
            "selection, prompt tuning, or "
            "hyperparameter tuning."
        ),
        "deprecated_bug_ids": sorted(
            KNOWN_DEPRECATED_LANG_BUGS
        ),
        "successful_bug_ids": (
            successful_bug_ids
        ),
        "skipped_bug_ids": (
            skipped_bug_ids
        ),
        "failed": failed,
        "negative_sampling": {
            "strategy": (
                "all"
                if max_negatives is None
                else (
                    "deterministic_hard_negative_"
                    "by_class_name_and_method_length"
                )
            ),
            "max_negatives_per_positive": (
                max_negatives
            ),
        },
        "counts": {
            "bugs": len(
                successful_bug_ids
            ),
            "samples": len(
                all_records
            ),
            "positive": positive,
            "negative": negative,
        },
    }

    with manifest_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_statistics(
        all_records
    )

    print()
    print(
        f"Skipped bugs: "
        f"{skipped_bug_ids}"
    )

    print(
        f"Failed bugs: "
        f"{[item['bug_id'] for item in failed]}"
    )

    print()
    print("Saved:")
    print(test_file)
    print(manifest_file)


if __name__ == "__main__":
    main()