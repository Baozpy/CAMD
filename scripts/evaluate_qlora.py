from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "qwen35_9b_4bit"
)

DEFAULT_ADAPTER = (
    PROJECT_ROOT
    / "adapters"
    / "qwen35_9b_camd_qlora_balanced11_20"
)

DEFAULT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
    / "validation.jsonl"
)

DEFAULT_MLX_DATA = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
    / "mlx"
    / "valid.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qlora"
    / "validation_logprob.json"
)


COMPLETION_TRUE = {
    "is_target_defect": True,
}

COMPLETION_FALSE = {
    "is_target_defect": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base Qwen or a binary QLoRA adapter "
            "using completion log-likelihood."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_ADAPTER,
    )

    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help=(
            "Evaluate the base model without loading "
            "a LoRA/QLoRA adapter."
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=(
            "JSONL file containing candidate metadata "
            "and ground-truth labels."
        ),
    )

    parser.add_argument(
        "--mlx-data",
        type=Path,
        default=DEFAULT_MLX_DATA,
        help=(
            "MLX JSONL file containing chat messages."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Classification threshold for p(true). "
            "Keep at 0.5 for final evaluation."
        ),
    )

    parser.add_argument(
        "--use-mean-logprob",
        action="store_true",
        help=(
            "Use mean completion log-probability "
            "instead of total completion log-probability."
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def json_completion(
    completion: dict[str, Any],
) -> str:
    return json.dumps(
        completion,
        ensure_ascii=False,
    )


def build_full_messages(
    prompt_messages: list[dict[str, str]],
    completion: dict[str, Any],
) -> list[dict[str, str]]:
    return (
        list(prompt_messages)
        + [
            {
                "role": "assistant",
                "content": json_completion(
                    completion
                ),
            }
        ]
    )


def encode_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(
            add_generation_prompt
        ),
    )

    return tokenizer.encode(
        rendered,
        add_special_tokens=False,
    )


def completion_logprob(
    model,
    tokenizer,
    prompt_messages: list[dict[str, str]],
    completion: dict[str, Any],
    *,
    use_mean_logprob: bool,
) -> tuple[float, int]:
    """
    Compute conditional log P(completion | prompt).

    The prompt is rendered with:
        add_generation_prompt=True

    The full conversation is rendered with the assistant
    completion included.

    Only assistant-completion tokens are scored.
    """

    prefix_ids = encode_chat(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )

    full_messages = build_full_messages(
        prompt_messages,
        completion,
    )

    full_ids = encode_chat(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
    )

    prefix_len = len(
        prefix_ids
    )

    # Normally prefix_ids should be an exact prefix.
    # If the tokenizer/chat template introduces a small
    # mismatch, fall back to the longest common prefix.
    if (
        prefix_len > len(full_ids)
        or full_ids[:prefix_len]
        != prefix_ids
    ):
        prefix_len = 0

        max_common = min(
            len(prefix_ids),
            len(full_ids),
        )

        while (
            prefix_len < max_common
            and prefix_ids[prefix_len]
            == full_ids[prefix_len]
        ):
            prefix_len += 1

    if prefix_len <= 0:
        raise RuntimeError(
            "Could not align prompt and completion "
            "tokens under the chat template."
        )

    completion_token_count = (
        len(full_ids)
        - prefix_len
    )

    if completion_token_count <= 0:
        raise RuntimeError(
            "Completion contains no scoreable tokens."
        )

    input_ids = mx.array(
        [full_ids],
        dtype=mx.int32,
    )

    logits = model(
        input_ids
    )

    if isinstance(
        logits,
        tuple,
    ):
        logits = logits[0]

    logits = logits[0]

    log_probs = (
        logits
        - mx.logsumexp(
            logits,
            axis=-1,
            keepdims=True,
        )
    )

    token_scores: list[float] = []

    for position in range(
        prefix_len,
        len(full_ids),
    ):
        prediction_position = (
            position - 1
        )

        if prediction_position < 0:
            continue

        target_token = int(
            full_ids[position]
        )

        score = log_probs[
            prediction_position,
            target_token,
        ]

        mx.eval(
            score
        )

        token_scores.append(
            float(
                score.item()
            )
        )

    if not token_scores:
        raise RuntimeError(
            "No completion token scores were produced."
        )

    total = sum(
        token_scores
    )

    if use_mean_logprob:
        final_score = (
            total
            / len(token_scores)
        )
    else:
        final_score = total

    return (
        final_score,
        len(token_scores),
    )


def stable_binary_softmax(
    score_true: float,
    score_false: float,
) -> tuple[float, float]:
    maximum = max(
        score_true,
        score_false,
    )

    exp_true = math.exp(
        score_true - maximum
    )

    exp_false = math.exp(
        score_false - maximum
    )

    denominator = (
        exp_true
        + exp_false
    )

    return (
        exp_true / denominator,
        exp_false / denominator,
    )


def candidate_identity(
    row: dict[str, Any],
) -> str:
    return (
        f"{row['class_name']}::"
        f"{row['method_name']}@"
        f"{row['start_line']}-"
        f"{row['end_line']}"
    )


def tie_aware_ranks(
    candidate_scores: dict[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """
    Descending ranking with average rank for exact ties.
    """

    sorted_items = sorted(
        candidate_scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    ranks: dict[str, float] = {}

    index = 0

    while index < len(
        sorted_items
    ):
        score = sorted_items[
            index
        ][1]

        end = (
            index + 1
        )

        while end < len(
            sorted_items
        ):
            other_score = (
                sorted_items[
                    end
                ][1]
            )

            if (
                abs(
                    other_score
                    - score
                )
                > tolerance
            ):
                break

            end += 1

        first_rank = (
            index + 1
        )

        last_rank = end

        average_rank = (
            first_rank
            + last_rank
        ) / 2.0

        for tie_index in range(
            index,
            end,
        ):
            identity = (
                sorted_items[
                    tie_index
                ][0]
            )

            ranks[
                identity
            ] = (
                average_rank
            )

        index = end

    return ranks


def evaluate(
    model,
    tokenizer,
    metadata_rows: list[dict[str, Any]],
    mlx_rows: list[dict[str, Any]],
    *,
    threshold: float,
    use_mean_logprob: bool,
) -> dict[str, Any]:
    if len(metadata_rows) != len(
        mlx_rows
    ):
        raise ValueError(
            "Metadata and MLX dataset sizes differ: "
            f"{len(metadata_rows)} vs "
            f"{len(mlx_rows)}"
        )

    sample_results: list[
        dict[str, Any]
    ] = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    bug_scores: dict[
        tuple[str, int],
        dict[str, float],
    ] = defaultdict(dict)

    bug_info: dict[
        tuple[str, int],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    total = len(
        metadata_rows
    )

    for index, (
        metadata,
        mlx_row,
    ) in enumerate(
        zip(
            metadata_rows,
            mlx_rows,
        ),
        start=1,
    ):
        messages = mlx_row[
            "messages"
        ]

        if (
            not messages
            or messages[-1][
                "role"
            ]
            != "assistant"
        ):
            raise ValueError(
                "Expected MLX row to end "
                "with an assistant message."
            )

        prompt_messages = (
            messages[:-1]
        )

        score_true, tokens_true = (
            completion_logprob(
                model,
                tokenizer,
                prompt_messages,
                COMPLETION_TRUE,
                use_mean_logprob=(
                    use_mean_logprob
                ),
            )
        )

        score_false, tokens_false = (
            completion_logprob(
                model,
                tokenizer,
                prompt_messages,
                COMPLETION_FALSE,
                use_mean_logprob=(
                    use_mean_logprob
                ),
            )
        )

        (
            p_true,
            p_false,
        ) = stable_binary_softmax(
            score_true,
            score_false,
        )

        prediction = (
            p_true
            >= threshold
        )

        gold = bool(
            metadata[
                "is_target_defect"
            ]
        )

        if gold and prediction:
            tp += 1
        elif (
            not gold
            and not prediction
        ):
            tn += 1
        elif (
            not gold
            and prediction
        ):
            fp += 1
        else:
            fn += 1

        project = metadata[
            "project"
        ]

        bug_id = int(
            metadata[
                "bug_id"
            ]
        )

        identity = (
            candidate_identity(
                metadata
            )
        )

        bug_key = (
            project,
            bug_id,
        )

        # p_true is used directly as the
        # candidate localization score.
        bug_scores[
            bug_key
        ][identity] = (
            p_true
        )

        bug_info[
            bug_key
        ][identity] = {
            "project": project,
            "bug_id": bug_id,
            "class_name": (
                metadata[
                    "class_name"
                ]
            ),
            "method_name": (
                metadata[
                    "method_name"
                ]
            ),
            "start_line": int(
                metadata[
                    "start_line"
                ]
            ),
            "end_line": int(
                metadata[
                    "end_line"
                ]
            ),
            "is_target_defect": (
                gold
            ),
        }

        result = {
            "index": index,
            "project": project,
            "bug_id": bug_id,
            "identity": identity,
            "class_name": metadata[
                "class_name"
            ],
            "method_name": metadata[
                "method_name"
            ],
            "start_line": int(
                metadata[
                    "start_line"
                ]
            ),
            "end_line": int(
                metadata[
                    "end_line"
                ]
            ),
            "gold": gold,
            "prediction": bool(
                prediction
            ),
            "correct": (
                bool(prediction)
                == gold
            ),
            "logprob_true": (
                score_true
            ),
            "logprob_false": (
                score_false
            ),
            "completion_tokens_true": (
                tokens_true
            ),
            "completion_tokens_false": (
                tokens_false
            ),
            "p_true": p_true,
            "p_false": p_false,
        }

        sample_results.append(
            result
        )

        print(
            f"[{index}/{total}] "
            f"{project}-{bug_id} "
            f"{metadata['method_name']} "
            f"gold={gold} "
            f"p_true={p_true:.4f} "
            f"pred={bool(prediction)}"
        )

    # ---------------------------------------------------------
    # Sample-level classification
    # ---------------------------------------------------------

    total_samples = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn)
        / total_samples
        if total_samples
        else 0.0
    )

    precision = (
        tp
        / (tp + fp)
        if (
            tp + fp
        )
        else 0.0
    )

    recall = (
        tp
        / (tp + fn)
        if (
            tp + fn
        )
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    specificity = (
        tn
        / (tn + fp)
        if (
            tn + fp
        )
        else 0.0
    )

    balanced_accuracy = (
        recall
        + specificity
    ) / 2.0

    sample_metrics = {
        "samples": total_samples,
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (
            balanced_accuracy
        ),
    }

    # ---------------------------------------------------------
    # Bug-level localization
    # ---------------------------------------------------------

    bug_results: list[
        dict[str, Any]
    ] = []

    reciprocal_ranks: list[
        float
    ] = []

    top1 = 0
    top3 = 0
    top5 = 0

    for bug_key in sorted(
        bug_scores
    ):
        project, bug_id = (
            bug_key
        )

        candidate_scores = (
            bug_scores[
                bug_key
            ]
        )

        ranks = tie_aware_ranks(
            candidate_scores
        )

        target_identities = [
            identity
            for (
                identity,
                info,
            ) in bug_info[
                bug_key
            ].items()
            if info[
                "is_target_defect"
            ]
        ]

        if not target_identities:
            continue

        target_ranks = [
            ranks[identity]
            for identity
            in target_identities
            if identity in ranks
        ]

        if not target_ranks:
            continue

        best_target_rank = min(
            target_ranks
        )

        reciprocal_rank = (
            1.0
            / best_target_rank
        )

        reciprocal_ranks.append(
            reciprocal_rank
        )

        if best_target_rank <= 1:
            top1 += 1

        if best_target_rank <= 3:
            top3 += 1

        if best_target_rank <= 5:
            top5 += 1

        ranked_identities = sorted(
            candidate_scores,
            key=lambda identity: (
                ranks[identity],
                identity,
            ),
        )

        ranking = []

        for identity in (
            ranked_identities
        ):
            info = (
                bug_info[
                    bug_key
                ][identity]
            )

            ranking.append(
                {
                    "identity": (
                        identity
                    ),
                    "score": (
                        candidate_scores[
                            identity
                        ]
                    ),
                    "rank": (
                        ranks[
                            identity
                        ]
                    ),
                    "class_name": (
                        info[
                            "class_name"
                        ]
                    ),
                    "method_name": (
                        info[
                            "method_name"
                        ]
                    ),
                    "start_line": (
                        info[
                            "start_line"
                        ]
                    ),
                    "end_line": (
                        info[
                            "end_line"
                        ]
                    ),
                    "is_target_defect": (
                        bool(
                            info[
                                "is_target_defect"
                            ]
                        )
                    ),
                }
            )

        bug_results.append(
            {
                "project": project,
                "bug_id": bug_id,
                "candidate_count": (
                    len(
                        candidate_scores
                    )
                ),
                "target_count": (
                    len(
                        target_identities
                    )
                ),
                "best_target_rank": (
                    best_target_rank
                ),
                "reciprocal_rank": (
                    reciprocal_rank
                ),
                "ranking": ranking,
            }
        )

    bug_count = len(
        reciprocal_ranks
    )

    mrr = (
        sum(
            reciprocal_ranks
        )
        / bug_count
        if bug_count
        else 0.0
    )

    bug_metrics = {
        "bugs": bug_count,
        "mrr": mrr,
        "top1": (
            top1 / bug_count
            if bug_count
            else 0.0
        ),
        "top3": (
            top3 / bug_count
            if bug_count
            else 0.0
        ),
        "top5": (
            top5 / bug_count
            if bug_count
            else 0.0
        ),
    }

    return {
        "sample_metrics": (
            sample_metrics
        ),
        "bug_metrics": (
            bug_metrics
        ),
        "sample_results": (
            sample_results
        ),
        "bug_results": (
            bug_results
        ),
    }


def print_summary(
    results: dict[str, Any],
) -> None:
    sample = results[
        "sample_metrics"
    ]

    bug = results[
        "bug_metrics"
    ]

    print()
    print(
        "=" * 100
    )
    print(
        "QLoRA Completion-LogProb Evaluation"
    )
    print(
        "=" * 100
    )

    print()
    print(
        "Confusion Matrix"
    )

    print(
        f"  TP: "
        f"{sample['tp']}"
    )

    print(
        f"  TN: "
        f"{sample['tn']}"
    )

    print(
        f"  FP: "
        f"{sample['fp']}"
    )

    print(
        f"  FN: "
        f"{sample['fn']}"
    )

    print()
    print(
        "Sample-Level Classification"
    )

    print(
        f"  Accuracy: "
        f"{sample['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{sample['precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{sample['recall']:.4f}"
    )

    print(
        f"  F1: "
        f"{sample['f1']:.4f}"
    )

    print(
        f"  Specificity: "
        f"{sample['specificity']:.4f}"
    )

    print(
        "  Balanced Accuracy: "
        f"{sample['balanced_accuracy']:.4f}"
    )

    print()
    print(
        "Bug-Level Localization"
    )

    print(
        f"  Bugs: "
        f"{bug['bugs']}"
    )

    print(
        f"  MRR: "
        f"{bug['mrr']:.4f}"
    )

    print(
        f"  Top-1: "
        f"{bug['top1']:.4f}"
    )

    print(
        f"  Top-3: "
        f"{bug['top3']:.4f}"
    )

    print(
        f"  Top-5: "
        f"{bug['top5']:.4f}"
    )


def main() -> None:
    args = parse_args()

    if not (
        0.0
        <= args.threshold
        <= 1.0
    ):
        raise ValueError(
            "--threshold must be between "
            "0 and 1."
        )

    metadata_rows = load_jsonl(
        args.metadata
    )

    mlx_rows = load_jsonl(
        args.mlx_data
    )

    print()
    print(
        "Loading QLoRA model..."
    )

    print(
        f"Model: "
        f"{args.model}"
    )

    if args.no_adapter:
        print(
            "Adapter: NONE "
            "(base model evaluation)"
        )

        model, tokenizer = load(
            str(args.model)
        )

    else:
        print(
            f"Adapter: "
            f"{args.adapter}"
        )

        model, tokenizer = load(
            str(args.model),
            adapter_path=(
                str(
                    args.adapter
                )
            ),
        )

    print()
    print(
        f"Evaluation samples: "
        f"{len(metadata_rows)}"
    )

    print(
        f"Metadata: "
        f"{args.metadata}"
    )

    print(
        f"MLX data: "
        f"{args.mlx_data}"
    )

    results = evaluate(
        model,
        tokenizer,
        metadata_rows,
        mlx_rows,
        threshold=(
            args.threshold
        ),
        use_mean_logprob=(
            args.use_mean_logprob
        ),
    )

    results[
        "configuration"
    ] = {
        "model": str(
            args.model
        ),
        "adapter": (
            None
            if args.no_adapter
            else str(
                args.adapter
            )
        ),
        "no_adapter": (
            bool(
                args.no_adapter
            )
        ),
        "metadata": str(
            args.metadata
        ),
        "mlx_data": str(
            args.mlx_data
        ),
        "threshold": (
            args.threshold
        ),
        "scoring": (
            "mean_logprob"
            if args.use_mean_logprob
            else "total_logprob"
        ),
    }

    print_summary(
        results
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved:"
    )

    print(
        args.output
    )


if __name__ == "__main__":
    main()