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
    / "qwen35_9b_camd_pairwise20"
)

DEFAULT_PAIRWISE_METADATA = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
    / "pairwise"
    / "validation.jsonl"
)

DEFAULT_MLX_VALIDATION = (
    PROJECT_ROOT
    / "data"
    / "finetuning"
    / "pairwise_mlx"
    / "valid.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qlora"
    / "validation_pairwise20.json"
)


COMPLETION_A = {
    "preferred_candidate": "A",
}

COMPLETION_B = {
    "preferred_candidate": "B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
            "Evaluate the base model without "
            "loading a LoRA/QLoRA adapter."
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_PAIRWISE_METADATA,
    )

    parser.add_argument(
        "--mlx-validation",
        type=Path,
        default=DEFAULT_MLX_VALIDATION,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--use-mean-logprob",
        action="store_true",
        help=(
            "Use mean completion log-probability "
            "instead of total log-probability."
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows = []

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

    The prompt is rendered with add_generation_prompt=True.
    The full conversation is rendered with the assistant
    completion included.

    Tokens belonging to the assistant completion are scored
    autoregressively.
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

    # Usually prefix_ids is an exact token prefix of full_ids.
    # If the chat template differs slightly, fall back to the
    # longest common prefix.
    prefix_len = len(prefix_ids)

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
        len(full_ids) - prefix_len
    )

    if completion_token_count <= 0:
        raise RuntimeError(
            "Completion contains no scoreable tokens."
        )

    input_ids = mx.array(
        [full_ids],
        dtype=mx.int32,
    )

    logits = model(input_ids)

    # Some model wrappers may return a tuple.
    if isinstance(logits, tuple):
        logits = logits[0]

    # Shape:
    # [batch, sequence, vocabulary]
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

    # Token at full_ids[position] is predicted by
    # logits[position - 1].
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

        mx.eval(score)

        token_scores.append(
            float(score.item())
        )

    if not token_scores:
        raise RuntimeError(
            "No completion token scores were produced."
        )

    total = sum(token_scores)

    if use_mean_logprob:
        final_score = (
            total / len(token_scores)
        )
    else:
        final_score = total

    return (
        final_score,
        len(token_scores),
    )


def stable_binary_softmax(
    score_a: float,
    score_b: float,
) -> tuple[float, float]:
    maximum = max(
        score_a,
        score_b,
    )

    exp_a = math.exp(
        score_a - maximum
    )

    exp_b = math.exp(
        score_b - maximum
    )

    denominator = (
        exp_a + exp_b
    )

    return (
        exp_a / denominator,
        exp_b / denominator,
    )


def tie_aware_ranks(
    candidate_scores: dict[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """
    Descending score ranking with average ranks for ties.

    Example:
        scores = 0.9, 0.8, 0.8, 0.5

        ranks = 1, 2.5, 2.5, 4
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

    while index < len(sorted_items):
        score = sorted_items[
            index
        ][1]

        end = index + 1

        while end < len(sorted_items):
            other_score = sorted_items[
                end
            ][1]

            if (
                abs(
                    other_score - score
                )
                > tolerance
            ):
                break

            end += 1

        # Positions are 1-based.
        first_rank = index + 1
        last_rank = end

        average_rank = (
            first_rank + last_rank
        ) / 2.0

        for tie_index in range(
            index,
            end,
        ):
            identity = sorted_items[
                tie_index
            ][0]

            ranks[identity] = (
                average_rank
            )

        index = end

    return ranks


def candidate_display(
    candidate: dict[str, Any],
) -> str:
    return (
        f"{candidate['class_name']}::"
        f"{candidate['method_name']}@"
        f"{candidate['start_line']}-"
        f"{candidate['end_line']}"
    )


def evaluate_pairwise(
    model,
    tokenizer,
    metadata_rows: list[dict[str, Any]],
    mlx_rows: list[dict[str, Any]],
    *,
    use_mean_logprob: bool,
) -> dict[str, Any]:
    if len(metadata_rows) != len(
        mlx_rows
    ):
        raise ValueError(
            "Metadata and MLX validation sizes differ: "
            f"{len(metadata_rows)} vs "
            f"{len(mlx_rows)}"
        )

    pair_results = []

    correct = 0
    a_correct = 0
    a_total = 0
    b_correct = 0
    b_total = 0

    margins = []

    # bug -> candidate -> accumulated expected wins
    score_sums: dict[
        tuple[str, int],
        dict[str, float],
    ] = defaultdict(
        lambda: defaultdict(float)
    )

    score_counts: dict[
        tuple[str, int],
        dict[str, int],
    ] = defaultdict(
        lambda: defaultdict(int)
    )

    candidate_info: dict[
        tuple[str, int],
        dict[str, dict[str, Any]],
    ] = defaultdict(dict)

    total_pairs = len(
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
                "Expected each MLX row to end "
                "with an assistant message."
            )

        prompt_messages = (
            messages[:-1]
        )

        gold = metadata[
            "preferred_candidate"
        ]

        logprob_a, tokens_a = (
            completion_logprob(
                model,
                tokenizer,
                prompt_messages,
                COMPLETION_A,
                use_mean_logprob=(
                    use_mean_logprob
                ),
            )
        )

        logprob_b, tokens_b = (
            completion_logprob(
                model,
                tokenizer,
                prompt_messages,
                COMPLETION_B,
                use_mean_logprob=(
                    use_mean_logprob
                ),
            )
        )

        p_a, p_b = (
            stable_binary_softmax(
                logprob_a,
                logprob_b,
            )
        )

        prediction = (
            "A"
            if p_a >= p_b
            else "B"
        )

        is_correct = (
            prediction == gold
        )

        if is_correct:
            correct += 1

        if gold == "A":
            a_total += 1

            if is_correct:
                a_correct += 1
        else:
            b_total += 1

            if is_correct:
                b_correct += 1

        gold_probability = (
            p_a
            if gold == "A"
            else p_b
        )

        wrong_probability = (
            p_b
            if gold == "A"
            else p_a
        )

        margin = (
            gold_probability
            - wrong_probability
        )

        margins.append(
            margin
        )

        project = metadata[
            "project"
        ]

        bug_id = int(
            metadata[
                "bug_id"
            ]
        )

        bug_key = (
            project,
            bug_id,
        )

        candidate_a = metadata[
            "candidate_a"
        ]

        candidate_b = metadata[
            "candidate_b"
        ]

        identity_a = candidate_a[
            "identity"
        ]

        identity_b = candidate_b[
            "identity"
        ]

        # Expected win probability.
        score_sums[
            bug_key
        ][identity_a] += p_a

        score_counts[
            bug_key
        ][identity_a] += 1

        score_sums[
            bug_key
        ][identity_b] += p_b

        score_counts[
            bug_key
        ][identity_b] += 1

        candidate_info[
            bug_key
        ][identity_a] = (
            candidate_a
        )

        candidate_info[
            bug_key
        ][identity_b] = (
            candidate_b
        )

        pair_result = {
            "index": index,
            "project": project,
            "bug_id": bug_id,
            "candidate_a": (
                candidate_a
            ),
            "candidate_b": (
                candidate_b
            ),
            "gold": gold,
            "prediction": prediction,
            "correct": is_correct,
            "logprob_a": (
                logprob_a
            ),
            "logprob_b": (
                logprob_b
            ),
            "completion_tokens_a": (
                tokens_a
            ),
            "completion_tokens_b": (
                tokens_b
            ),
            "p_a": p_a,
            "p_b": p_b,
            "gold_margin": margin,
        }

        pair_results.append(
            pair_result
        )

        print(
            f"[{index}/{total_pairs}] "
            f"{project}-{bug_id} "
            f"gold={gold} "
            f"pA={p_a:.4f} "
            f"pB={p_b:.4f} "
            f"pred={prediction} "
            f"{'OK' if is_correct else 'WRONG'}"
        )

    pair_accuracy = (
        correct / total_pairs
        if total_pairs
        else 0.0
    )

    a_accuracy = (
        a_correct / a_total
        if a_total
        else 0.0
    )

    b_accuracy = (
        b_correct / b_total
        if b_total
        else 0.0
    )

    mean_margin = (
        sum(margins)
        / len(margins)
        if margins
        else 0.0
    )

    # ---------------------------------------------------------
    # Bug-level candidate ranking
    # ---------------------------------------------------------

    bug_results = []

    reciprocal_ranks = []

    top1 = 0
    top3 = 0
    top5 = 0

    for bug_key in sorted(
        score_sums
    ):
        project, bug_id = (
            bug_key
        )

        averaged_scores = {}

        for identity, score_sum in (
            score_sums[
                bug_key
            ].items()
        ):
            count = score_counts[
                bug_key
            ][identity]

            averaged_scores[
                identity
            ] = (
                score_sum / count
            )

        ranks = tie_aware_ranks(
            averaged_scores
        )

        targets = [
            identity
            for identity, info
            in candidate_info[
                bug_key
            ].items()
            if info[
                "is_target_defect"
            ]
        ]

        if not targets:
            continue

        target_ranks = [
            ranks[identity]
            for identity in targets
            if identity in ranks
        ]

        if not target_ranks:
            continue

        best_target_rank = min(
            target_ranks
        )

        reciprocal_rank = (
            1.0 / best_target_rank
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

        ranked_candidates = sorted(
            averaged_scores,
            key=lambda identity: (
                ranks[identity],
                identity,
            ),
        )

        ranking_output = []

        for identity in (
            ranked_candidates
        ):
            info = candidate_info[
                bug_key
            ][identity]

            ranking_output.append(
                {
                    "identity":
                        identity,
                    "method":
                        candidate_display(
                            info
                        ),
                    "score":
                        averaged_scores[
                            identity
                        ],
                    "rank":
                        ranks[
                            identity
                        ],
                    "is_target_defect":
                        bool(
                            info[
                                "is_target_defect"
                            ]
                        ),
                    "pair_count":
                        score_counts[
                            bug_key
                        ][identity],
                }
            )

        bug_results.append(
            {
                "project": project,
                "bug_id": bug_id,
                "candidate_count": (
                    len(
                        averaged_scores
                    )
                ),
                "target_count": (
                    len(targets)
                ),
                "best_target_rank": (
                    best_target_rank
                ),
                "reciprocal_rank": (
                    reciprocal_rank
                ),
                "ranking": (
                    ranking_output
                ),
            }
        )

    bug_count = len(
        reciprocal_ranks
    )

    mrr = (
        sum(reciprocal_ranks)
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

    pair_metrics = {
        "pairs": total_pairs,
        "correct": correct,
        "accuracy": pair_accuracy,
        "preferred_a_total": (
            a_total
        ),
        "preferred_a_correct": (
            a_correct
        ),
        "preferred_a_accuracy": (
            a_accuracy
        ),
        "preferred_b_total": (
            b_total
        ),
        "preferred_b_correct": (
            b_correct
        ),
        "preferred_b_accuracy": (
            b_accuracy
        ),
        "mean_gold_margin": (
            mean_margin
        ),
    }

    return {
        "pair_metrics":
            pair_metrics,
        "bug_metrics":
            bug_metrics,
        "pair_results":
            pair_results,
        "bug_results":
            bug_results,
    }


def print_summary(
    results: dict[str, Any],
) -> None:
    pair = results[
        "pair_metrics"
    ]

    bug = results[
        "bug_metrics"
    ]

    print()
    print("=" * 100)
    print(
        "Pairwise QLoRA Evaluation"
    )
    print("=" * 100)

    print()
    print("Pair-Level Preference")

    print(
        f"  Pairs: "
        f"{pair['pairs']}"
    )

    print(
        f"  Correct: "
        f"{pair['correct']}"
    )

    print(
        f"  Accuracy: "
        f"{pair['accuracy']:.4f}"
    )

    print(
        "  Preferred-A Accuracy: "
        f"{pair['preferred_a_accuracy']:.4f}"
        f" "
        f"({pair['preferred_a_correct']}/"
        f"{pair['preferred_a_total']})"
    )

    print(
        "  Preferred-B Accuracy: "
        f"{pair['preferred_b_accuracy']:.4f}"
        f" "
        f"({pair['preferred_b_correct']}/"
        f"{pair['preferred_b_total']})"
    )

    print(
        "  Mean Gold Margin: "
        f"{pair['mean_gold_margin']:.4f}"
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

    metadata_rows = load_jsonl(
        args.metadata
    )

    mlx_rows = load_jsonl(
        args.mlx_validation
    )

    print()
    print("Loading model...")
    print(
        f"Model: {args.model}"
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
                str(args.adapter)
            ),
        )

    print()
    print(
        "Validation pairs: "
        f"{len(metadata_rows)}"
    )

    results = evaluate_pairwise(
        model,
        tokenizer,
        metadata_rows,
        mlx_rows,
        use_mean_logprob=(
            args.use_mean_logprob
        ),
    )

    results[
        "configuration"
    ] = {
        "model":
            str(args.model),
        "adapter": (
            None
            if args.no_adapter
            else str(
                args.adapter
            )
        ),
        "no_adapter":
            bool(
                args.no_adapter
            ),
        "metadata":
            str(args.metadata),
        "mlx_validation":
            str(
                args.mlx_validation
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
    print("Saved:")
    print(args.output)


if __name__ == "__main__":
    main()