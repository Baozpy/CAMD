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
    / "mlx"
)

SYSTEM_PROMPT = (
    "You are a software defect localization assistant. "
    "Given one candidate Java method, determine whether it is "
    "a target defect responsible for the current failing test. "
    "Return only the required JSON object."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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


def token_ids(
    tokenizer,
    text: str,
) -> list[int]:
    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def truncate_head_tail(
    tokenizer,
    text: str,
    max_tokens: int,
    *,
    head_ratio: float,
) -> tuple[str, bool]:
    ids = token_ids(
        tokenizer,
        text,
    )

    if len(ids) <= max_tokens:
        return text, False

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

    kept = (
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
            kept,
            skip_special_tokens=True,
        ),
        True,
    )


def truncate_head(
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

    kept = ids[:max_tokens]

    return (
        tokenizer.decode(
            kept,
            skip_special_tokens=True,
        )
        + "\n[...TRUNCATED...]",
        True,
    )


def render_messages(
    tokenizer,
    prompt: str,
    label: bool,
) -> str:
    messages = [
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
                    "is_target_defect": label
                },
                ensure_ascii=False,
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def rendered_length(
    tokenizer,
    prompt: str,
    label: bool,
) -> int:
    rendered = render_messages(
        tokenizer,
        prompt,
        label,
    )

    return len(
        token_ids(
            tokenizer,
            rendered,
        )
    )


def pack_prompt(
    tokenizer,
    original_prompt: str,
    label: bool,
    max_seq_length: int,
) -> tuple[
    str,
    int,
    dict[str, bool],
]:
    project = extract_between(
        original_prompt,
        "PROJECT\n=======",
        "CANDIDATE CLASS",
    )

    candidate_class = extract_between(
        original_prompt,
        "CANDIDATE CLASS\n===============",
        "CANDIDATE METHOD",
    )

    candidate_method = extract_between(
        original_prompt,
        "CANDIDATE METHOD\n================",
        "STATIC EVIDENCE",
    )

    static_evidence = extract_between(
        original_prompt,
        "STATIC EVIDENCE\n===============",
        "FAILING TEST CONTEXT",
    )

    failing_test = extract_between(
        original_prompt,
        "FAILING TEST CONTEXT\n====================",
        "Return whether this candidate method",
    )

    truncated = {
        "candidate_method": False,
        "failing_test": False,
        "static_evidence": False,
    }

    candidate_method, flag = (
        truncate_head_tail(
            tokenizer,
            candidate_method,
            900,
            head_ratio=0.70,
        )
    )
    truncated["candidate_method"] = flag

    failing_test, flag = (
        truncate_head_tail(
            tokenizer,
            failing_test,
            650,
            head_ratio=0.75,
        )
    )
    truncated["failing_test"] = flag

    static_evidence, flag = (
        truncate_head(
            tokenizer,
            static_evidence,
            250,
        )
    )
    truncated["static_evidence"] = flag

    prompt = f"""You are given one candidate Java method from a buggy program.

Your task is to determine whether this method is a target defect
responsible for the CURRENT failing test.

Do not search for unrelated defects.
Do not propose a patch.

PROJECT
=======

{project}


CANDIDATE CLASS
===============

{candidate_class}


CANDIDATE METHOD
================

{candidate_method}


FAILING TEST CONTEXT
====================

{failing_test}


STATIC EVIDENCE
===============

{static_evidence}


Return whether this candidate method is a target defect for the
current failing test."""

    length = rendered_length(
        tokenizer,
        prompt,
        label,
    )

    if length <= max_seq_length:
        return (
            prompt,
            length,
            truncated,
        )

    static_evidence, _ = (
        truncate_head(
            tokenizer,
            static_evidence,
            120,
        )
    )
    truncated["static_evidence"] = True

    prompt = f"""You are given one candidate Java method from a buggy program.

Your task is to determine whether this method is a target defect
responsible for the CURRENT failing test.

Do not search for unrelated defects.
Do not propose a patch.

PROJECT
=======

{project}


CANDIDATE CLASS
===============

{candidate_class}


CANDIDATE METHOD
================

{candidate_method}


FAILING TEST CONTEXT
====================

{failing_test}


STATIC EVIDENCE
===============

{static_evidence}


Return whether this candidate method is a target defect for the
current failing test."""

    length = rendered_length(
        tokenizer,
        prompt,
        label,
    )

    if length > max_seq_length:
        raise ValueError(
            "Unable to fit held-out sample "
            f"within {max_seq_length} tokens. "
            f"Final length: {length}"
        )

    return (
        prompt,
        length,
        truncated,
    )


def percentile(
    values: list[int],
    q: float,
) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    pos = (
        len(values) - 1
    ) * q

    lower = int(pos)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = pos - lower

    return (
        values[lower]
        * (1.0 - weight)
        + values[upper]
        * weight
    )


def main() -> None:
    args = parse_args()

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

    rows = load_jsonl(
        args.input
    )

    converted = []
    lengths = []

    truncation_counts = {
        "candidate_method": 0,
        "failing_test": 0,
        "static_evidence": 0,
    }

    print()
    print(
        "Packing held-out test dataset..."
    )

    for index, row in enumerate(
        rows,
        start=1,
    ):
        label = bool(
            row["is_target_defect"]
        )

        (
            prompt,
            length,
            truncated,
        ) = pack_prompt(
            tokenizer,
            row["input"],
            label,
            args.max_seq_length,
        )

        messages = [
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
                        "is_target_defect":
                        label
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        converted.append(
            {
                "messages": messages,
            }
        )

        lengths.append(
            length
        )

        for key in truncation_counts:
            if truncated[key]:
                truncation_counts[
                    key
                ] += 1

        if index % 50 == 0:
            print(
                f"  Packed "
                f"{index}/{len(rows)}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        args.output_dir
        / "test.jsonl"
    )

    stats_file = (
        args.output_dir
        / "dataset_stats.json"
    )

    save_jsonl(
        converted,
        output_file,
    )

    stats = {
        "samples": len(rows),
        "min_tokens": min(lengths),
        "median_tokens": (
            statistics.median(
                lengths
            )
        ),
        "p90_tokens": percentile(
            lengths,
            0.90,
        ),
        "p95_tokens": percentile(
            lengths,
            0.95,
        ),
        "max_tokens": max(lengths),
        "over_2048": sum(
            value
            > args.max_seq_length
            for value in lengths
        ),
        "truncated_sections": (
            truncation_counts
        ),
    }

    with stats_file.open(
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
    print(
        "=" * 100
    )
    print(
        "Held-Out MLX Test Conversion"
    )
    print(
        "=" * 100
    )

    print()
    print(
        f"Samples: "
        f"{stats['samples']}"
    )

    print("Token lengths:")

    print(
        f"  Min: "
        f"{stats['min_tokens']}"
    )

    print(
        f"  Median: "
        f"{stats['median_tokens']}"
    )

    print(
        f"  P90: "
        f"{stats['p90_tokens']:.1f}"
    )

    print(
        f"  P95: "
        f"{stats['p95_tokens']:.1f}"
    )

    print(
        f"  Max: "
        f"{stats['max_tokens']}"
    )

    print(
        f"  > 2048: "
        f"{stats['over_2048']}"
    )

    print()
    print(
        "Truncated sections:"
    )

    print(
        "  Candidate method: "
        f"{truncation_counts['candidate_method']}"
    )

    print(
        "  Failing test: "
        f"{truncation_counts['failing_test']}"
    )

    print(
        "  Static evidence: "
        f"{truncation_counts['static_evidence']}"
    )

    print()
    print("Saved:")
    print(output_file)
    print(stats_file)


if __name__ == "__main__":
    main()