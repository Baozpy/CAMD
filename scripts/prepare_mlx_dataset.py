import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
)

OUTPUT_ROOT = (
    SOURCE_ROOT
    / "mlx"
)

HF_MODEL_ROOT = (
    PROJECT_ROOT
    / "hf_home"
    / "hub"
    / "models--Qwen--Qwen3.5-9B"
    / "snapshots"
)


SYSTEM_PROMPT = (
    "You are a software defect localization assistant. "
    "Determine whether the given Java method is the target defect "
    "responsible for the current failing test. "
    "Use the candidate method, static evidence, and failing-test "
    "context. Return only JSON."
)


# ---------------------------------------------------------------------
# Default token budgets
#
# These are section budgets, not the final sequence length.
#
# The final rendered chat is checked again against max_seq_length.
# ---------------------------------------------------------------------

DEFAULT_METHOD_BUDGET = 900
DEFAULT_TEST_BUDGET = 650
DEFAULT_STATIC_BUDGET = 250

DEFAULT_MAX_SEQ_LENGTH = 2048


def find_local_model_snapshot() -> Path:

    if not HF_MODEL_ROOT.exists():

        raise RuntimeError(
            "Local Qwen3.5-9B Hugging Face cache was not found:\n"
            f"{HF_MODEL_ROOT}"
        )

    snapshots = sorted(
        path
        for path in HF_MODEL_ROOT.iterdir()
        if path.is_dir()
    )

    if not snapshots:

        raise RuntimeError(
            "No local Qwen3.5-9B snapshot was found under:\n"
            f"{HF_MODEL_ROOT}"
        )

    # There should normally only be one cached revision.
    # If several exist, use the newest directory by mtime.
    snapshots.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return snapshots[0]


def load_jsonl(
    path: Path,
) -> list[dict]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(
                    line
                )
            )

    return records


def save_jsonl(
    records: list[dict],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


def save_json(
    data: dict,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def tokenize(
    tokenizer,
    text: str,
) -> list[int]:

    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def decode(
    tokenizer,
    token_ids: list[int],
) -> str:

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )


def token_count(
    tokenizer,
    text: str,
) -> int:

    return len(
        tokenize(
            tokenizer,
            text,
        )
    )


def truncate_head_tail(
    tokenizer,
    text: str,
    max_tokens: int,
    head_ratio: float = 0.70,
) -> str:
    """
    Preserve both the beginning and end of a long section.

    For source methods this is preferable to retaining only the
    beginning because return statements, boundary checks, exception
    handling, and closing branches can appear near the end.
    """

    if max_tokens <= 0:
        return ""

    tokens = tokenize(
        tokenizer,
        text,
    )

    if len(tokens) <= max_tokens:
        return text

    marker = (
        "\n\n"
        "[... content truncated for token budget ...]"
        "\n\n"
    )

    marker_tokens = tokenize(
        tokenizer,
        marker,
    )

    available = (
        max_tokens
        - len(marker_tokens)
    )

    if available <= 4:

        return decode(
            tokenizer,
            tokens[:max_tokens],
        )

    head_count = int(
        available
        * head_ratio
    )

    tail_count = (
        available
        - head_count
    )

    head = tokens[
        :head_count
    ]

    tail = tokens[
        -tail_count:
    ]

    return (
        decode(
            tokenizer,
            head,
        )
        + marker
        + decode(
            tokenizer,
            tail,
        )
    )


def truncate_head(
    tokenizer,
    text: str,
    max_tokens: int,
) -> str:

    if max_tokens <= 0:
        return ""

    tokens = tokenize(
        tokenizer,
        text,
    )

    if len(tokens) <= max_tokens:
        return text

    marker = (
        "\n"
        "[... truncated ...]"
    )

    marker_tokens = tokenize(
        tokenizer,
        marker,
    )

    available = (
        max_tokens
        - len(marker_tokens)
    )

    if available <= 0:

        return decode(
            tokenizer,
            tokens[:max_tokens],
        )

    return (
        decode(
            tokenizer,
            tokens[:available],
        )
        + marker
    )


def extract_between(
    text: str,
    start_marker: str,
    end_marker: str | None,
) -> str:

    start_index = (
        text.find(
            start_marker
        )
    )

    if start_index == -1:
        return ""

    start_index += len(
        start_marker
    )

    if end_marker is None:

        result = text[
            start_index:
        ]

    else:

        end_index = (
            text.find(
                end_marker,
                start_index,
            )
        )

        if end_index == -1:

            result = text[
                start_index:
            ]

        else:

            result = text[
                start_index:
                end_index
            ]

    return result.strip()


def parse_original_input(
    text: str,
) -> dict:

    project = extract_between(
        text,
        "PROJECT\n=======\n",
        "\n\nCANDIDATE CLASS",
    )

    class_name = extract_between(
        text,
        "CANDIDATE CLASS\n===============\n",
        "\n\nCANDIDATE METHOD",
    )

    candidate = extract_between(
        text,
        "CANDIDATE METHOD\n================\n",
        "\n\nSTATIC EVIDENCE",
    )

    static_evidence = extract_between(
        text,
        "STATIC EVIDENCE\n===============\n",
        "\n\nFAILING TEST CONTEXT",
    )

    failing_test = extract_between(
        text,
        "FAILING TEST CONTEXT\n====================\n",
        "\n\nReturn whether",
    )

    return {
        "project": project,
        "class_name": class_name,
        "candidate": candidate,
        "static_evidence": static_evidence,
        "failing_test": failing_test,
    }


def build_compact_user_prompt(
    parsed: dict,
    candidate_text: str,
    static_text: str,
    failing_test_text: str,
) -> str:

    return f"""PROJECT
{parsed["project"]}

CLASS
{parsed["class_name"]}

CANDIDATE METHOD
{candidate_text}

FAILING TEST CONTEXT
{failing_test_text}

STATIC EVIDENCE
{static_text}

Determine whether the candidate method is the target defect for the current failing test.
Return only:
{{"is_target_defect": true}}
or
{{"is_target_defect": false}}""".strip()


def build_assistant_answer(
    record: dict,
) -> str:

    value = bool(
        record[
            "is_target_defect"
        ]
    )

    return json.dumps(
        {
            "is_target_defect": value
        },
        ensure_ascii=False,
    )


def build_messages(
    user_content: str,
    answer: str,
) -> list[dict]:

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]


def rendered_chat_token_count(
    tokenizer,
    messages: list[dict],
) -> int:
    """
    Render the full conversation using the model's chat template,
    then tokenize the rendered text explicitly.

    This avoids version-dependent return types from
    apply_chat_template(tokenize=True).
    """

    rendered_text = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    )

    input_ids = (
        tokenizer.encode(
            rendered_text,
            add_special_tokens=False,
        )
    )

    return len(
        input_ids
    )


def pack_record(
    tokenizer,
    record: dict,
    max_seq_length: int,
    method_budget: int,
    test_budget: int,
    static_budget: int,
) -> tuple[
    dict,
    dict,
]:

    parsed = (
        parse_original_input(
            record[
                "input"
            ]
        )
    )

    original_method_tokens = (
        token_count(
            tokenizer,
            parsed[
                "candidate"
            ],
        )
    )

    original_test_tokens = (
        token_count(
            tokenizer,
            parsed[
                "failing_test"
            ],
        )
    )

    original_static_tokens = (
        token_count(
            tokenizer,
            parsed[
                "static_evidence"
            ],
        )
    )

    candidate_text = (
        truncate_head_tail(
            tokenizer,
            parsed[
                "candidate"
            ],
            method_budget,
            head_ratio=0.70,
        )
    )

    # Failing tests are important enough that we preserve both
    # beginning and ending context.
    failing_test_text = (
        truncate_head_tail(
            tokenizer,
            parsed[
                "failing_test"
            ],
            test_budget,
            head_ratio=0.75,
        )
    )

    # Static evidence generally contains summaries/features and can
    # be compressed more aggressively.
    static_text = (
        truncate_head(
            tokenizer,
            parsed[
                "static_evidence"
            ],
            static_budget,
        )
    )

    answer = (
        build_assistant_answer(
            record
        )
    )

    user_content = (
        build_compact_user_prompt(
            parsed=parsed,
            candidate_text=(
                candidate_text
            ),
            static_text=(
                static_text
            ),
            failing_test_text=(
                failing_test_text
            ),
        )
    )

    messages = (
        build_messages(
            user_content=(
                user_content
            ),
            answer=answer,
        )
    )

    packed_tokens = (
        rendered_chat_token_count(
            tokenizer,
            messages,
        )
    )

    # -----------------------------------------------------------------
    # Safety pass
    #
    # Section budgets should normally keep the sample below 2048.
    # If chat-template overhead or unusual metadata still pushes it over,
    # progressively shrink optional sections.
    # -----------------------------------------------------------------

    current_method_budget = (
        method_budget
    )

    current_test_budget = (
        test_budget
    )

    current_static_budget = (
        static_budget
    )

    safety_round = 0

    while (
        packed_tokens
        > max_seq_length
        and safety_round < 20
    ):

        safety_round += 1

        overflow = (
            packed_tokens
            - max_seq_length
        )

        # Static evidence is the first section to sacrifice.
        if current_static_budget > 80:

            reduction = max(
                32,
                min(
                    overflow + 16,
                    current_static_budget - 80,
                ),
            )

            current_static_budget -= (
                reduction
            )

        # Next reduce the candidate method, but keep a meaningful body.
        elif current_method_budget > 500:

            reduction = max(
                32,
                min(
                    overflow + 16,
                    current_method_budget - 500,
                ),
            )

            current_method_budget -= (
                reduction
            )

        # Failing-test context is reduced last.
        elif current_test_budget > 350:

            reduction = max(
                32,
                min(
                    overflow + 16,
                    current_test_budget - 350,
                ),
            )

            current_test_budget -= (
                reduction
            )

        else:

            break

        candidate_text = (
            truncate_head_tail(
                tokenizer,
                parsed[
                    "candidate"
                ],
                current_method_budget,
                head_ratio=0.70,
            )
        )

        failing_test_text = (
            truncate_head_tail(
                tokenizer,
                parsed[
                    "failing_test"
                ],
                current_test_budget,
                head_ratio=0.75,
            )
        )

        static_text = (
            truncate_head(
                tokenizer,
                parsed[
                    "static_evidence"
                ],
                current_static_budget,
            )
        )

        user_content = (
            build_compact_user_prompt(
                parsed=parsed,
                candidate_text=(
                    candidate_text
                ),
                static_text=(
                    static_text
                ),
                failing_test_text=(
                    failing_test_text
                ),
            )
        )

        messages = (
            build_messages(
                user_content=(
                    user_content
                ),
                answer=answer,
            )
        )

        packed_tokens = (
            rendered_chat_token_count(
                tokenizer,
                messages,
            )
        )

    output = {
        "messages": messages,

        "metadata": {
            "project": (
                record[
                    "project"
                ]
            ),

            "bug_id": (
                record[
                    "bug_id"
                ]
            ),

            "class_name": (
                record[
                    "class_name"
                ]
            ),

            "method_name": (
                record[
                    "method_name"
                ]
            ),

            "start_line": (
                record[
                    "start_line"
                ]
            ),

            "end_line": (
                record[
                    "end_line"
                ]
            ),

            "label": (
                record[
                    "label"
                ]
            ),

            "packed_token_count": (
                packed_tokens
            ),
        },
    }

    stats = {
        "project": (
            record[
                "project"
            ]
        ),

        "bug_id": (
            record[
                "bug_id"
            ]
        ),

        "method_name": (
            record[
                "method_name"
            ]
        ),

        "label": (
            record[
                "label"
            ]
        ),

        "original_method_tokens": (
            original_method_tokens
        ),

        "original_test_tokens": (
            original_test_tokens
        ),

        "original_static_tokens": (
            original_static_tokens
        ),

        "final_method_budget": (
            current_method_budget
        ),

        "final_test_budget": (
            current_test_budget
        ),

        "final_static_budget": (
            current_static_budget
        ),

        "packed_tokens": (
            packed_tokens
        ),

        "fits_max_length": (
            packed_tokens
            <= max_seq_length
        ),

        "method_truncated": (
            original_method_tokens
            > current_method_budget
        ),

        "test_truncated": (
            original_test_tokens
            > current_test_budget
        ),

        "static_truncated": (
            original_static_tokens
            > current_static_budget
        ),
    }

    return (
        output,
        stats,
    )


def percentile(
    values: list[int],
    percentile_value: float,
) -> float:

    if not values:
        return 0.0

    sorted_values = sorted(
        values
    )

    position = (
        percentile_value
        / 100.0
        * (
            len(sorted_values)
            - 1
        )
    )

    lower = math.floor(
        position
    )

    upper = math.ceil(
        position
    )

    if lower == upper:

        return float(
            sorted_values[
                lower
            ]
        )

    lower_value = (
        sorted_values[
            lower
        ]
    )

    upper_value = (
        sorted_values[
            upper
        ]
    )

    fraction = (
        position
        - lower
    )

    return (
        lower_value
        + (
            upper_value
            - lower_value
        )
        * fraction
    )


def summarize_token_lengths(
    stats: list[dict],
    max_seq_length: int,
) -> dict:

    values = [
        item[
            "packed_tokens"
        ]
        for item
        in stats
    ]

    if not values:

        return {
            "count": 0,
        }

    values_sorted = sorted(
        values
    )

    count_over_limit = sum(
        value
        > max_seq_length
        for value
        in values
    )

    method_truncated = sum(
        item[
            "method_truncated"
        ]
        for item
        in stats
    )

    test_truncated = sum(
        item[
            "test_truncated"
        ]
        for item
        in stats
    )

    static_truncated = sum(
        item[
            "static_truncated"
        ]
        for item
        in stats
    )

    return {
        "count": (
            len(values)
        ),

        "min_tokens": (
            min(values)
        ),

        "median_tokens": (
            percentile(
                values,
                50,
            )
        ),

        "p90_tokens": (
            percentile(
                values,
                90,
            )
        ),

        "p95_tokens": (
            percentile(
                values,
                95,
            )
        ),

        "max_tokens": (
            max(values)
        ),

        "max_seq_length": (
            max_seq_length
        ),

        "over_max_count": (
            count_over_limit
        ),

        "over_max_ratio": (
            count_over_limit
            / len(values)
        ),

        "method_truncated_count": (
            method_truncated
        ),

        "test_truncated_count": (
            test_truncated
        ),

        "static_truncated_count": (
            static_truncated
        ),
    }


def print_split_stats(
    name: str,
    records: list[dict],
    token_stats: list[dict],
    max_seq_length: int,
) -> None:

    positives = sum(
        record[
            "metadata"
        ][
            "label"
        ]
        for record
        in records
    )

    negatives = (
        len(records)
        - positives
    )

    bugs = {
        record[
            "metadata"
        ][
            "bug_id"
        ]
        for record
        in records
    }

    summary = (
        summarize_token_lengths(
            stats=token_stats,
            max_seq_length=(
                max_seq_length
            ),
        )
    )

    print()
    print(name)

    print(
        f"  Bugs: "
        f"{len(bugs)}"
    )

    print(
        f"  Samples: "
        f"{len(records)}"
    )

    print(
        f"  Positive: "
        f"{positives}"
    )

    print(
        f"  Negative: "
        f"{negatives}"
    )

    print(
        "  Token lengths:"
    )

    print(
        f"    Min: "
        f"{summary['min_tokens']}"
    )

    print(
        f"    Median: "
        f"{summary['median_tokens']:.1f}"
    )

    print(
        f"    P90: "
        f"{summary['p90_tokens']:.1f}"
    )

    print(
        f"    P95: "
        f"{summary['p95_tokens']:.1f}"
    )

    print(
        f"    Max: "
        f"{summary['max_tokens']}"
    )

    print(
        f"    > {max_seq_length}: "
        f"{summary['over_max_count']}"
    )

    print(
        "  Truncated sections:"
    )

    print(
        f"    Method: "
        f"{summary['method_truncated_count']}"
    )

    print(
        f"    Failing test: "
        f"{summary['test_truncated_count']}"
    )

    print(
        f"    Static evidence: "
        f"{summary['static_truncated_count']}"
    )


def convert_split(
    tokenizer,
    source_file: Path,
    max_seq_length: int,
    method_budget: int,
    test_budget: int,
    static_budget: int,
) -> tuple[
    list[dict],
    list[dict],
]:

    source_records = (
        load_jsonl(
            source_file
        )
    )

    output_records = []

    stats = []

    for index, record in enumerate(
        source_records,
        start=1,
    ):

        packed_record, packed_stats = (
            pack_record(
                tokenizer=tokenizer,
                record=record,
                max_seq_length=(
                    max_seq_length
                ),
                method_budget=(
                    method_budget
                ),
                test_budget=(
                    test_budget
                ),
                static_budget=(
                    static_budget
                ),
            )
        )

        output_records.append(
            packed_record
        )

        stats.append(
            packed_stats
        )

        if index % 100 == 0:

            print(
                f"  Packed "
                f"{index}/"
                f"{len(source_records)}"
            )

    return (
        output_records,
        stats,
    )


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=(
            DEFAULT_MAX_SEQ_LENGTH
        ),
    )

    parser.add_argument(
        "--method-budget",
        type=int,
        default=(
            DEFAULT_METHOD_BUDGET
        ),
    )

    parser.add_argument(
        "--test-budget",
        type=int,
        default=(
            DEFAULT_TEST_BUDGET
        ),
    )

    parser.add_argument(
        "--static-budget",
        type=int,
        default=(
            DEFAULT_STATIC_BUDGET
        ),
    )

    args = (
        parser.parse_args()
    )

    model_snapshot = (
        find_local_model_snapshot()
    )

    print()
    print(
        "Loading tokenizer from:"
    )

    print(
        model_snapshot
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_snapshot,
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    train_source = (
        SOURCE_ROOT
        / "train.jsonl"
    )

    validation_source = (
        SOURCE_ROOT
        / "validation.jsonl"
    )

    print()
    print(
        "Packing training dataset..."
    )

    (
        train_records,
        train_stats,
    ) = convert_split(
        tokenizer=tokenizer,
        source_file=(
            train_source
        ),
        max_seq_length=(
            args.max_seq_length
        ),
        method_budget=(
            args.method_budget
        ),
        test_budget=(
            args.test_budget
        ),
        static_budget=(
            args.static_budget
        ),
    )

    print()
    print(
        "Packing validation dataset..."
    )

    (
        validation_records,
        validation_stats,
    ) = convert_split(
        tokenizer=tokenizer,
        source_file=(
            validation_source
        ),
        max_seq_length=(
            args.max_seq_length
        ),
        method_budget=(
            args.method_budget
        ),
        test_budget=(
            args.test_budget
        ),
        static_budget=(
            args.static_budget
        ),
    )

    train_output = (
        OUTPUT_ROOT
        / "train.jsonl"
    )

    validation_output = (
        OUTPUT_ROOT
        / "valid.jsonl"
    )

    stats_output = (
        OUTPUT_ROOT
        / "dataset_stats.json"
    )

    save_jsonl(
        train_records,
        train_output,
    )

    save_jsonl(
        validation_records,
        validation_output,
    )

    train_summary = (
        summarize_token_lengths(
            stats=(
                train_stats
            ),
            max_seq_length=(
                args.max_seq_length
            ),
        )
    )

    validation_summary = (
        summarize_token_lengths(
            stats=(
                validation_stats
            ),
            max_seq_length=(
                args.max_seq_length
            ),
        )
    )

    combined_stats = (
        train_stats
        + validation_stats
    )

    combined_summary = (
        summarize_token_lengths(
            stats=(
                combined_stats
            ),
            max_seq_length=(
                args.max_seq_length
            ),
        )
    )

    stats_file_data = {
        "model_snapshot": (
            str(
                model_snapshot
            )
        ),

        "packing_policy": {
            "max_seq_length": (
                args.max_seq_length
            ),

            "initial_method_budget": (
                args.method_budget
            ),

            "initial_failing_test_budget": (
                args.test_budget
            ),

            "initial_static_evidence_budget": (
                args.static_budget
            ),

            "priority": [
                "failing_test_context",
                "candidate_method",
                "static_evidence",
            ],

            "candidate_truncation": (
                "head_tail"
            ),

            "failing_test_truncation": (
                "head_tail"
            ),

            "static_evidence_truncation": (
                "head_only"
            ),
        },

        "train": (
            train_summary
        ),

        "validation": (
            validation_summary
        ),

        "combined": (
            combined_summary
        ),

        "per_sample": {
            "train": (
                train_stats
            ),

            "validation": (
                validation_stats
            ),
        },
    }

    save_json(
        stats_file_data,
        stats_output,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "MLX Token-Aware Dataset Conversion"
    )

    print(
        "=" * 100
    )

    print_split_stats(
        name="Train",
        records=(
            train_records
        ),
        token_stats=(
            train_stats
        ),
        max_seq_length=(
            args.max_seq_length
        ),
    )

    print_split_stats(
        name="Validation",
        records=(
            validation_records
        ),
        token_stats=(
            validation_stats
        ),
        max_seq_length=(
            args.max_seq_length
        ),
    )

    print()
    print(
        "Combined"
    )

    print(
        f"  Samples: "
        f"{combined_summary['count']}"
    )

    print(
        f"  Median tokens: "
        f"{combined_summary['median_tokens']:.1f}"
    )

    print(
        f"  P90 tokens: "
        f"{combined_summary['p90_tokens']:.1f}"
    )

    print(
        f"  P95 tokens: "
        f"{combined_summary['p95_tokens']:.1f}"
    )

    print(
        f"  Max tokens: "
        f"{combined_summary['max_tokens']}"
    )

    print(
        f"  > {args.max_seq_length}: "
        f"{combined_summary['over_max_count']}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        train_output
    )

    print(
        validation_output
    )

    print(
        stats_output
    )


if __name__ == "__main__":
    main()