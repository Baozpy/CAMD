import argparse
import json
import random
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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
)

HELD_OUT_LANG_BUGS = set(
    range(
        1,
        21,
    )
)

KNOWN_DEPRECATED_LANG_BUGS = {
    2,
    18,
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

            file.write(
                "\n"
            )


def split_bug_ids(
    bug_ids: list[int],
    validation_ratio: float,
    seed: int,
) -> tuple[
    list[int],
    list[int],
]:

    unique_bug_ids = sorted(
        set(
            bug_ids
        )
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        unique_bug_ids
    )

    if len(
        unique_bug_ids
    ) <= 1:

        return (
            unique_bug_ids,
            [],
        )

    validation_count = max(
        1,
        round(
            len(
                unique_bug_ids
            )
            * validation_ratio
        ),
    )

    validation_count = min(
        validation_count,
        len(
            unique_bug_ids
        )
        - 1,
    )

    validation_bug_ids = sorted(
        unique_bug_ids[
            :validation_count
        ]
    )

    train_bug_ids = sorted(
        unique_bug_ids[
            validation_count:
        ]
    )

    return (
        train_bug_ids,
        validation_bug_ids,
    )


def print_statistics(
    name: str,
    records: list[dict],
) -> None:

    positive = sum(
        int(
            item["label"]
        )
        for item
        in records
    )

    negative = (
        len(records)
        - positive
    )

    bug_ids = sorted(
        {
            item["bug_id"]
            for item
            in records
        }
    )

    print()
    print(name)

    print(
        f"  Bugs: "
        f"{len(bug_ids)}"
    )

    print(
        f"  Samples: "
        f"{len(records)}"
    )

    print(
        f"  Positive: "
        f"{positive}"
    )

    print(
        f"  Negative: "
        f"{negative}"
    )

    if records:

        ratio = (
            positive
            / len(records)
        )

        print(
            f"  Positive ratio: "
            f"{ratio:.4f}"
        )

    print(
        f"  Bug IDs: "
        f"{bug_ids}"
    )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--project",
        default="Lang",
    )

    parser.add_argument(
        "--bug-start",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--bug-end",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
            "Keep every negative candidate "
            "instead of hard-negative sampling."
        ),
    )

    parser.add_argument(
        "--allow-held-out",
        action="store_true",
        help=(
            "Allow Lang 1-20 to be included. "
            "Do not use this for QLoRA training."
        ),
    )

    args = (
        parser.parse_args()
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

    builder = (
        QLoRADatasetBuilder(
            project_root=(
                PROJECT_ROOT
            ),
            max_negatives_per_positive=(
                max_negatives
            ),
        )
    )

    all_records = []

    successful_bug_ids = []

    skipped_bug_ids = []

    failed = []

    for bug_id in range(
        args.bug_start,
        args.bug_end + 1,
    ):

        if (
            args.project == "Lang"
            and bug_id
            in KNOWN_DEPRECATED_LANG_BUGS
        ):

            print()
            print(
                f"Skipping "
                f"Lang-{bug_id}: "
                f"deprecated."
            )

            skipped_bug_ids.append(
                bug_id
            )

            continue

        if (
            args.project == "Lang"
            and bug_id
            in HELD_OUT_LANG_BUGS
            and not args.allow_held_out
        ):

            print()
            print(
                f"Skipping "
                f"Lang-{bug_id}: "
                f"held-out test bug."
            )

            skipped_bug_ids.append(
                bug_id
            )

            continue

        print()
        print(
            "=" * 100
        )

        print(
            f"Building QLoRA samples: "
            f"{args.project}-{bug_id}"
        )

        print(
            "=" * 100
        )

        try:

            samples = (
                builder
                .build_bug_samples(
                    project=(
                        args.project
                    ),
                    bug_id=(
                        bug_id
                    ),
                )
            )

            records = [
                {
                    "project": (
                        sample.project
                    ),

                    "bug_id": (
                        sample.bug_id
                    ),

                    "class_name": (
                        sample.class_name
                    ),

                    "method_name": (
                        sample.method_name
                    ),

                    "start_line": (
                        sample.start_line
                    ),

                    "end_line": (
                        sample.end_line
                    ),

                    "method_length": (
                        sample.method_length
                    ),

                    "label": (
                        sample.label
                    ),

                    "is_target_defect": (
                        sample.is_target_defect
                    ),

                    "input": (
                        sample.input
                    ),

                    "output": (
                        sample.output
                    ),
                }
                for sample
                in samples
            ]

            positive = sum(
                record["label"]
                for record
                in records
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

            per_bug_file = (
                OUTPUT_ROOT
                / "by_bug"
                / (
                    f"{args.project}_"
                    f"{bug_id}.jsonl"
                )
            )

            write_jsonl(
                records,
                per_bug_file,
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
                f"FAILED: "
                f"{args.project}-{bug_id}"
            )

            print(
                exc
            )

            failed.append(
                {
                    "bug_id": (
                        bug_id
                    ),
                    "error": (
                        str(exc)
                    ),
                }
            )

    if not all_records:

        print()
        print(
            "No samples were built."
        )

        return

    (
        train_bug_ids,
        validation_bug_ids,
    ) = split_bug_ids(
        bug_ids=(
            successful_bug_ids
        ),
        validation_ratio=(
            args.validation_ratio
        ),
        seed=(
            args.seed
        ),
    )

    train_bug_set = set(
        train_bug_ids
    )

    validation_bug_set = set(
        validation_bug_ids
    )

    train_records = [
        record
        for record
        in all_records
        if (
            record["bug_id"]
            in train_bug_set
        )
    ]

    validation_records = [
        record
        for record
        in all_records
        if (
            record["bug_id"]
            in validation_bug_set
        )
    ]

    train_file = (
        OUTPUT_ROOT
        / "train.jsonl"
    )

    validation_file = (
        OUTPUT_ROOT
        / "validation.jsonl"
    )

    manifest_file = (
        OUTPUT_ROOT
        / "dataset_manifest.json"
    )

    write_jsonl(
        train_records,
        train_file,
    )

    write_jsonl(
        validation_records,
        validation_file,
    )

    manifest = {
        "project": (
            args.project
        ),

        "requested_range": {
            "start": (
                args.bug_start
            ),
            "end": (
                args.bug_end
            ),
        },

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

        "held_out_test_policy": (
            "Lang 1-20 are reserved "
            "for final evaluation and "
            "are excluded from training "
            "unless --allow-held-out is "
            "explicitly supplied."
        ),

        "successful_bug_ids": (
            successful_bug_ids
        ),

        "skipped_bug_ids": (
            skipped_bug_ids
        ),

        "failed": (
            failed
        ),

        "split": {
            "seed": (
                args.seed
            ),

            "validation_ratio": (
                args.validation_ratio
            ),

            "train_bug_ids": (
                train_bug_ids
            ),

            "validation_bug_ids": (
                validation_bug_ids
            ),
        },

        "counts": {
            "all_samples": (
                len(
                    all_records
                )
            ),

            "train_samples": (
                len(
                    train_records
                )
            ),

            "validation_samples": (
                len(
                    validation_records
                )
            ),

            "train_positive": (
                sum(
                    record["label"]
                    for record
                    in train_records
                )
            ),

            "train_negative": (
                sum(
                    1 - record["label"]
                    for record
                    in train_records
                )
            ),

            "validation_positive": (
                sum(
                    record["label"]
                    for record
                    in validation_records
                )
            ),

            "validation_negative": (
                sum(
                    1 - record["label"]
                    for record
                    in validation_records
                )
            ),
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

    print()
    print(
        "=" * 100
    )

    print(
        "QLoRA Dataset Summary"
    )

    print(
        "=" * 100
    )

    print_statistics(
        "Train",
        train_records,
    )

    print_statistics(
        "Validation",
        validation_records,
    )

    print()
    print(
        f"Failed bugs: "
        f"{[item['bug_id'] for item in failed]}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        train_file
    )

    print(
        validation_file
    )

    print(
        manifest_file
    )


if __name__ == "__main__":
    main()