from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "hf_home"
    / "hub"
    / "models--Qwen--Qwen3.5-9B"
    / "snapshots"
    / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)

SYSTEM_PROMPT = (
    "You are a software defect localization assistant. "
    "Compare two candidate Java methods and determine which candidate "
    "is more likely to contain the target defect responsible for the "
    "given failing test. Return only the required JSON object."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "finetuning"
            / "pairwise"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "finetuning"
            / "pairwise_mlx"
        ),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def save_jsonl(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def extract_between(
    text: str,
    start_marker: str,
    end_marker: str | None,
) -> str:
    start = text.find(start_marker)

    if start == -1:
        return ""

    start += len(start_marker)

    if end_marker is None:
        return text[start:].strip()

    end = text.find(
        end_marker,
        start,
    )

    if end == -1:
        return text[start:].strip()

    return text[start:end].strip()


def extract_project(prompt: str) -> str:
    return extract_between(
        prompt,
        "PROJECT\n=======\n",
        "FAILING TEST CONTEXT",
    ).strip()


def extract_failing_test(prompt: str) -> str:
    return extract_between(
        prompt,
        "FAILING TEST CONTEXT\n====================\n",
        "CANDIDATE A\n===========",
    ).strip()


def extract_candidate_a(prompt: str) -> str:
    return extract_between(
        prompt,
        "CANDIDATE A\n===========\n",
        "CANDIDATE A STATIC EVIDENCE",
    ).strip()


def extract_candidate_a_static(prompt: str) -> str:
    return extract_between(
        prompt,
        "CANDIDATE A STATIC EVIDENCE\n===========================\n",
        "CANDIDATE B\n===========",
    ).strip()


def extract_candidate_b(prompt: str) -> str:
    return extract_between(
        prompt,
        "CANDIDATE B\n===========\n",
        "CANDIDATE B STATIC EVIDENCE",
    ).strip()


def extract_candidate_b_static(prompt: str) -> str:
    section = extract_between(
        prompt,
        "CANDIDATE B STATIC EVIDENCE\n===========================\n",
        "Return which candidate",
    )

    return section.strip()


def token_ids(
    tokenizer,
    text: str,
) -> list[int]:
    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def token_len(
    tokenizer,
    text: str,
) -> int:
    return len(
        token_ids(
            tokenizer,
            text,
        )
    )


def truncate_tokens_head_tail(
    tokenizer,
    text: str,
    max_tokens: int,
    *,
    head_ratio: float = 0.75,
) -> tuple[str, bool]:
    ids = token_ids(
        tokenizer,
        text,
    )

    if len(ids) <= max_tokens:
        return text, False

    if max_tokens <= 16:
        trimmed_ids = ids[:max_tokens]

        return (
            tokenizer.decode(
                trimmed_ids,
                skip_special_tokens=True,
            ),
            True,
        )

    marker = "\n[...TRUNCATED...]\n"

    marker_ids = token_ids(
        tokenizer,
        marker,
    )

    available = max(
        1,
        max_tokens - len(marker_ids),
    )

    head_count = int(
        available * head_ratio
    )

    tail_count = (
        available - head_count
    )

    kept_ids = (
        ids[:head_count]
        + marker_ids
        + (
            ids[-tail_count:]
            if tail_count > 0
            else []
        )
    )

    return (
        tokenizer.decode(
            kept_ids,
            skip_special_tokens=True,
        ),
        True,
    )


def truncate_tokens_head(
    tokenizer,
    text: str,
    max_tokens: int,
) -> tuple[str, bool]:
    ids = token_ids(
        tokenizer,
        text,
    )

    if len(ids) <= max_tokens:
        return text, False

    trimmed_ids = ids[:max_tokens]

    return (
        tokenizer.decode(
            trimmed_ids,
            skip_special_tokens=True,
        )
        + "\n[...TRUNCATED...]",
        True,
    )


def build_prompt(
    project: str,
    failing_test: str,
    candidate_a: str,
    candidate_a_static: str,
    candidate_b: str,
    candidate_b_static: str,
) -> str:
    return f"""You are given two candidate Java methods from the same buggy program.

Your task is to determine which candidate is MORE LIKELY to contain the target defect responsible for the CURRENT failing test.

Compare Candidate A and Candidate B directly.

Do not search for unrelated defects.
Do not propose a patch.
Choose exactly one candidate.

PROJECT
=======

{project}


FAILING TEST CONTEXT
====================

{failing_test}


CANDIDATE A
===========

{candidate_a}


CANDIDATE A STATIC EVIDENCE
===========================

{candidate_a_static}


CANDIDATE B
===========

{candidate_b}


CANDIDATE B STATIC EVIDENCE
===========================

{candidate_b_static}


Return which candidate is more likely to contain the target defect for the current failing test."""


def make_messages(
    prompt: str,
    preferred_candidate: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "preferred_candidate":
                    preferred_candidate
                },
                ensure_ascii=False,
            ),
        },
    ]


def rendered_length(
    tokenizer,
    prompt: str,
    preferred_candidate: str,
) -> int:
    rendered = tokenizer.apply_chat_template(
        make_messages(
            prompt,
            preferred_candidate,
        ),
        tokenize=False,
        add_generation_prompt=False,
    )

    return token_len(
        tokenizer,
        rendered,
    )


def pack_pairwise_prompt(
    tokenizer,
    original_prompt: str,
    preferred_candidate: str,
    max_seq_length: int,
) -> tuple[str, int, dict[str, bool]]:
    project = extract_project(
        original_prompt
    )

    failing_test = extract_failing_test(
        original_prompt
    )

    candidate_a = extract_candidate_a(
        original_prompt
    )

    candidate_a_static = (
        extract_candidate_a_static(
            original_prompt
        )
    )

    candidate_b = extract_candidate_b(
        original_prompt
    )

    candidate_b_static = (
        extract_candidate_b_static(
            original_prompt
        )
    )

    truncation = {
        "failing_test": False,
        "candidate_a": False,
        "candidate_b": False,
        "candidate_a_static": False,
        "candidate_b_static": False,
        "safety_reduction": False,
    }

    # Initial section budgets.
    #
    # Important sections receive larger budgets.
    failing_test, truncated = (
        truncate_tokens_head_tail(
            tokenizer,
            failing_test,
            520,
            head_ratio=0.75,
        )
    )
    truncation["failing_test"] = truncated

    candidate_a, truncated = (
        truncate_tokens_head_tail(
            tokenizer,
            candidate_a,
            480,
            head_ratio=0.70,
        )
    )
    truncation["candidate_a"] = truncated

    candidate_b, truncated = (
        truncate_tokens_head_tail(
            tokenizer,
            candidate_b,
            480,
            head_ratio=0.70,
        )
    )
    truncation["candidate_b"] = truncated

    candidate_a_static, truncated = (
        truncate_tokens_head(
            tokenizer,
            candidate_a_static,
            180,
        )
    )
    truncation[
        "candidate_a_static"
    ] = truncated

    candidate_b_static, truncated = (
        truncate_tokens_head(
            tokenizer,
            candidate_b_static,
            180,
        )
    )
    truncation[
        "candidate_b_static"
    ] = truncated

    prompt = build_prompt(
        project,
        failing_test,
        candidate_a,
        candidate_a_static,
        candidate_b,
        candidate_b_static,
    )

    length = rendered_length(
        tokenizer,
        prompt,
        preferred_candidate,
    )

    if length <= max_seq_length:
        return (
            prompt,
            length,
            truncation,
        )

    # Safety reduction:
    # reduce static evidence first.
    candidate_a_static, extra_a = (
        truncate_tokens_head(
            tokenizer,
            candidate_a_static,
            100,
        )
    )

    candidate_b_static, extra_b = (
        truncate_tokens_head(
            tokenizer,
            candidate_b_static,
            100,
        )
    )

    truncation[
        "candidate_a_static"
    ] = (
        truncation[
            "candidate_a_static"
        ]
        or extra_a
    )

    truncation[
        "candidate_b_static"
    ] = (
        truncation[
            "candidate_b_static"
        ]
        or extra_b
    )

    truncation["safety_reduction"] = True

    prompt = build_prompt(
        project,
        failing_test,
        candidate_a,
        candidate_a_static,
        candidate_b,
        candidate_b_static,
    )

    length = rendered_length(
        tokenizer,
        prompt,
        preferred_candidate,
    )

    if length <= max_seq_length:
        return (
            prompt,
            length,
            truncation,
        )

    # Reduce failing-test context slightly.
    failing_test, extra = (
        truncate_tokens_head_tail(
            tokenizer,
            failing_test,
            420,
            head_ratio=0.75,
        )
    )

    truncation["failing_test"] = (
        truncation["failing_test"]
        or extra
    )

    prompt = build_prompt(
        project,
        failing_test,
        candidate_a,
        candidate_a_static,
        candidate_b,
        candidate_b_static,
    )

    length = rendered_length(
        tokenizer,
        prompt,
        preferred_candidate,
    )

    if length <= max_seq_length:
        return (
            prompt,
            length,
            truncation,
        )

    # Reduce both candidate methods symmetrically.
    candidate_a, extra_a = (
        truncate_tokens_head_tail(
            tokenizer,
            candidate_a,
            400,
            head_ratio=0.70,
        )
    )

    candidate_b, extra_b = (
        truncate_tokens_head_tail(
            tokenizer,
            candidate_b,
            400,
            head_ratio=0.70,
        )
    )

    truncation["candidate_a"] = (
        truncation["candidate_a"]
        or extra_a
    )

    truncation["candidate_b"] = (
        truncation["candidate_b"]
        or extra_b
    )

    prompt = build_prompt(
        project,
        failing_test,
        candidate_a,
        candidate_a_static,
        candidate_b,
        candidate_b_static,
    )

    length = rendered_length(
        tokenizer,
        prompt,
        preferred_candidate,
    )

    if length <= max_seq_length:
        return (
            prompt,
            length,
            truncation,
        )

    # Final fallback:
    # keep both candidates, preserve symmetry, sacrifice static evidence.
    candidate_a_static = (
        "[STATIC EVIDENCE TRUNCATED]"
    )

    candidate_b_static = (
        "[STATIC EVIDENCE TRUNCATED]"
    )

    truncation[
        "candidate_a_static"
    ] = True

    truncation[
        "candidate_b_static"
    ] = True

    candidate_a, _ = (
        truncate_tokens_head_tail(
            tokenizer,
            candidate_a,
            350,
            head_ratio=0.70,
        )
    )

    candidate_b, _ = (
        truncate_tokens_head_tail(
            tokenizer,
            candidate_b,
            350,
            head_ratio=0.70,
        )
    )

    failing_test, _ = (
        truncate_tokens_head_tail(
            tokenizer,
            failing_test,
            350,
            head_ratio=0.75,
        )
    )

    prompt = build_prompt(
        project,
        failing_test,
        candidate_a,
        candidate_a_static,
        candidate_b,
        candidate_b_static,
    )

    length = rendered_length(
        tokenizer,
        prompt,
        preferred_candidate,
    )

    if length > max_seq_length:
        raise ValueError(
            "Unable to fit pairwise sample "
            f"within {max_seq_length} tokens "
            f"after section-aware packing. "
            f"Final length: {length}"
        )

    return (
        prompt,
        length,
        truncation,
    )


def percentile(
    values: list[int],
    q: float,
) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    position = (
        len(values) - 1
    ) * q

    lower = int(position)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = position - lower

    return (
        values[lower]
        * (1.0 - weight)
        + values[upper]
        * weight
    )


def convert_split(
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    max_seq_length: int,
    split_name: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    converted = []

    lengths = []

    truncation_counts = {
        "failing_test": 0,
        "candidate_a": 0,
        "candidate_b": 0,
        "candidate_a_static": 0,
        "candidate_b_static": 0,
        "safety_reduction": 0,
    }

    for index, row in enumerate(
        rows,
        start=1,
    ):
        (
            prompt,
            length,
            truncation,
        ) = pack_pairwise_prompt(
            tokenizer,
            row["input"],
            row["preferred_candidate"],
            max_seq_length,
        )

        messages = make_messages(
            prompt,
            row["preferred_candidate"],
        )

        converted.append(
            {
                "messages": messages,
            }
        )

        lengths.append(length)

        for key in truncation_counts:
            if truncation.get(
                key,
                False,
            ):
                truncation_counts[key] += 1

        if index % 50 == 0:
            print(
                f"  Packed {index}/{len(rows)}"
            )

    stats = {
        "split": split_name,
        "samples": len(rows),
        "min_tokens": (
            min(lengths)
            if lengths
            else 0
        ),
        "median_tokens": (
            statistics.median(
                lengths
            )
            if lengths
            else 0
        ),
        "p90_tokens": percentile(
            lengths,
            0.90,
        ),
        "p95_tokens": percentile(
            lengths,
            0.95,
        ),
        "max_tokens": (
            max(lengths)
            if lengths
            else 0
        ),
        "over_limit": sum(
            length > max_seq_length
            for length in lengths
        ),
        "truncation_counts":
            truncation_counts,
    }

    return (
        converted,
        stats,
    )


def print_stats(
    name: str,
    stats: dict[str, Any],
) -> None:
    print()
    print(name)

    print(
        f"  Samples: "
        f"{stats['samples']}"
    )

    print("  Token lengths:")

    print(
        f"    Min: "
        f"{stats['min_tokens']}"
    )

    print(
        f"    Median: "
        f"{stats['median_tokens']}"
    )

    print(
        f"    P90: "
        f"{stats['p90_tokens']:.1f}"
    )

    print(
        f"    P95: "
        f"{stats['p95_tokens']:.1f}"
    )

    print(
        f"    Max: "
        f"{stats['max_tokens']}"
    )

    print(
        f"    > 2048: "
        f"{stats['over_limit']}"
    )

    print("  Truncated sections:")

    counts = stats[
        "truncation_counts"
    ]

    print(
        "    Failing test: "
        f"{counts['failing_test']}"
    )

    print(
        "    Candidate A: "
        f"{counts['candidate_a']}"
    )

    print(
        "    Candidate B: "
        f"{counts['candidate_b']}"
    )

    print(
        "    Candidate A static: "
        f"{counts['candidate_a_static']}"
    )

    print(
        "    Candidate B static: "
        f"{counts['candidate_b_static']}"
    )

    print(
        "    Safety reduction: "
        f"{counts['safety_reduction']}"
    )


def main() -> None:
    args = parse_args()

    train_path = (
        args.input_dir
        / "train.jsonl"
    )

    validation_path = (
        args.input_dir
        / "validation.jsonl"
    )

    print()
    print("Loading tokenizer from:")
    print(args.model_path)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(args.model_path),
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    train_rows = load_jsonl(
        train_path
    )

    validation_rows = load_jsonl(
        validation_path
    )

    print()
    print(
        "Packing pairwise training "
        "dataset..."
    )

    (
        train_converted,
        train_stats,
    ) = convert_split(
        tokenizer,
        train_rows,
        max_seq_length=(
            args.max_seq_length
        ),
        split_name="train",
    )

    print()
    print(
        "Packing pairwise validation "
        "dataset..."
    )

    (
        validation_converted,
        validation_stats,
    ) = convert_split(
        tokenizer,
        validation_rows,
        max_seq_length=(
            args.max_seq_length
        ),
        split_name="validation",
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_output = (
        args.output_dir
        / "train.jsonl"
    )

    valid_output = (
        args.output_dir
        / "valid.jsonl"
    )

    stats_output = (
        args.output_dir
        / "dataset_stats.json"
    )

    save_jsonl(
        train_converted,
        train_output,
    )

    save_jsonl(
        validation_converted,
        valid_output,
    )

    stats = {
        "max_seq_length":
            args.max_seq_length,
        "train":
            train_stats,
        "validation":
            validation_stats,
    }

    with stats_output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print(
        "Pairwise MLX Dataset Conversion"
    )
    print("=" * 100)

    print_stats(
        "Train",
        train_stats,
    )

    print_stats(
        "Validation",
        validation_stats,
    )

    print()
    print("Saved:")
    print(train_output)
    print(valid_output)
    print(stats_output)


if __name__ == "__main__":
    main()